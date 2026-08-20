//! Worker Management Module
//!
//! Provides worker lifecycle operations and fan-out request utilities.

use std::{collections::HashMap, sync::Arc, time::Duration};

use axum::response::{IntoResponse, Response};
use futures::{
    future,
    stream::{self, StreamExt},
};
use http::StatusCode;
use serde_json::Value;
use tokio::{
    sync::{watch, Mutex},
    task::JoinHandle,
};
use tracing::{debug, info, warn};

use crate::{
    core::{metrics_aggregator::MetricPack, ConnectionMode, Worker, WorkerRegistry, WorkerType},
    policies::PolicyRegistry,
    protocols::worker_spec::{FlushCacheResult, WorkerLoadInfo, WorkerLoadsResult},
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CONCURRENT: usize = 32;

/// Result of a fan-out request to a single worker
struct WorkerResponse {
    url: String,
    result: Result<reqwest::Response, reqwest::Error>,
}

/// Fan out requests to workers in parallel
async fn fan_out(
    workers: &[Arc<dyn Worker>],
    client: &reqwest::Client,
    endpoint: &str,
    method: reqwest::Method,
) -> Vec<WorkerResponse> {
    let futures: Vec<_> = workers
        .iter()
        .map(|worker| {
            let client = client.clone();
            let url = worker.url().to_string();
            let full_url = format!("{}/{}", url, endpoint);
            let api_key = worker.api_key().clone();
            let method = method.clone();

            async move {
                let mut req = client.request(method, &full_url).timeout(REQUEST_TIMEOUT);
                if let Some(key) = api_key {
                    req = req.bearer_auth(key);
                }
                WorkerResponse {
                    url,
                    result: req.send().await,
                }
            }
        })
        .collect();

    stream::iter(futures)
        .buffer_unordered(MAX_CONCURRENT)
        .collect()
        .await
}

pub enum EngineMetricsResult {
    Ok(String),
    Err(String),
}

impl IntoResponse for EngineMetricsResult {
    fn into_response(self) -> Response {
        match self {
            Self::Ok(text) => (StatusCode::OK, text).into_response(),
            Self::Err(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response(),
        }
    }
}

pub struct WorkerManager;

impl WorkerManager {
    pub fn get_worker_urls(registry: &Arc<WorkerRegistry>) -> Vec<String> {
        registry
            .get_all()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    pub async fn flush_cache_all(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> FlushCacheResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        let http_workers: Vec<_> = workers
            .into_iter()
            .filter(|w| matches!(w.connection_mode(), ConnectionMode::Http))
            .collect();

        if http_workers.is_empty() {
            return FlushCacheResult {
                successful: vec![],
                failed: vec![],
                total_workers,
                http_workers: 0,
                message: "No HTTP workers available for cache flush".to_string(),
            };
        }

        info!(
            "Flushing cache on {} HTTP workers (out of {} total)",
            http_workers.len(),
            total_workers
        );

        let responses = fan_out(&http_workers, client, "flush_cache", reqwest::Method::POST).await;

        let mut successful = Vec::new();
        let mut failed = Vec::new();

        for resp in responses {
            match resp.result {
                Ok(r) if r.status().is_success() => successful.push(resp.url),
                Ok(r) => failed.push((resp.url, format!("HTTP {}", r.status()))),
                Err(e) => failed.push((resp.url, e.to_string())),
            }
        }

        let message = if failed.is_empty() {
            format!(
                "Successfully flushed cache on all {} HTTP workers",
                successful.len()
            )
        } else {
            format!(
                "Cache flush: {} succeeded, {} failed",
                successful.len(),
                failed.len()
            )
        };

        info!("{}", message);

        FlushCacheResult {
            successful,
            failed,
            total_workers,
            http_workers: http_workers.len(),
            message,
        }
    }

    pub async fn get_all_worker_loads(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> WorkerLoadsResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        let futures: Vec<_> = workers
            .iter()
            .map(|worker| {
                let url = worker.url().to_string();
                let api_key = worker.api_key().clone();
                let worker_type = match worker.worker_type() {
                    WorkerType::Regular => None,
                    WorkerType::Prefill { .. } => Some("prefill".to_string()),
                    WorkerType::Decode => Some("decode".to_string()),
                };
                let is_http = matches!(worker.connection_mode(), ConnectionMode::Http);
                let client = client.clone();

                async move {
                    let load = if is_http {
                        Self::parse_load_response(&client, &url, api_key.as_deref()).await
                    } else {
                        -1
                    };
                    WorkerLoadInfo {
                        worker: url,
                        worker_type,
                        load,
                    }
                }
            })
            .collect();

        let loads = future::join_all(futures).await;
        let successful = loads.iter().filter(|l| l.load >= 0).count();
        let failed = loads.iter().filter(|l| l.load < 0).count();

        WorkerLoadsResult {
            loads,
            total_workers,
            successful,
            failed,
        }
    }

    /// Split a worker URL into its addressable base and optional DP rank.
    ///
    /// Under `--dp-aware` the registry stores one worker per DP rank as
    /// `http://host:port@N`. That string is NOT a fetchable URL - `@` is the
    /// userinfo delimiter, so `http://127.0.0.1:8001@0/v1/loads` resolves to
    /// host `0` and the request fails ("Failed to connect to 0.0.0.0 port 80").
    ///
    /// Upstream builds its load-fetch URL straight from `worker.url()`, so every
    /// token-load fetch fails under --dp-aware and every worker reports -1.
    ///
    /// Returns `(base_url, Some(rank))` for a rank worker, `(url, None)` otherwise.
    fn split_worker_url(url: &str) -> (&str, Option<u32>) {
        match url.rsplit_once('@') {
            Some((base, rank)) => match rank.parse::<u32>() {
                Ok(r) => (base, Some(r)),
                Err(_) => (url, None), // an '@' that is not a rank suffix
            },
            None => (url, None),
        }
    }

    async fn parse_load_response(
        client: &reqwest::Client,
        url: &str,
        api_key: Option<&str>,
    ) -> isize {
        let (base, rank) = Self::split_worker_url(url);
        let load_url = format!("{}/v1/loads?include=core", base);
        let mut req = client.get(&load_url).timeout(REQUEST_TIMEOUT);
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }

        match req.send().await {
            Ok(r) if r.status().is_success() => match r.json::<Value>().await {
                Ok(json) => Self::extract_total_tokens(&json, rank),
                _ => -1,
            },
            _ => -1,
        }
    }

    /// Pull a token-level load figure out of a /v1/loads response.
    ///
    /// Upstream reads `json["aggregate"]["total_tokens"]`, but no SGLang engine
    /// emits an `aggregate` key - v0.5.15 and current main both return:
    ///
    /// ```json
    /// { "timestamp": ..., "version": "0.5.15",
    ///   "loads": [ { "dp_rank": 0, "num_used_tokens": 0,
    ///                "num_waiting_uncached_tokens": 0, ... } ] }
    /// ```
    ///
    /// So the upstream lookup always misses and every worker reports -1, which
    /// silently degrades power_of_two to request counts. This reads the schema
    /// the engine actually produces, summing across DP ranks, and keeps the
    /// `aggregate` path first so a future engine that does emit it still wins.
    ///
    /// Per-worker KV saturation in permille (0-1000), from the same payload.
    ///
    /// `extract_total_tokens` gives an absolute token count, which cannot answer
    /// "is this worker full?" on its own. `/v1/loads` also reports
    /// `max_total_num_tokens`, so the ratio is computable without extra calls.
    /// Returns None when capacity is unknown - callers must treat that as
    /// "unknown", never as "saturated", or a parse failure would eject the pool.
    /// `rank = Some(n)` scopes the reading to that DP rank; `None` aggregates.
    ///
    /// Scoping matters: each DP rank owns a SEPARATE KV pool and queue, so a
    /// pool-wide average hides the case this gate exists for. Rank 0 at 100% and
    /// rank 1 at 0% averages to a healthy-looking 500 permille while half the
    /// server is wedged.
    fn extract_saturation_permille(json: &Value, rank: Option<u32>) -> Option<u32> {
        let loads = json.get("loads")?.as_array()?;
        if loads.is_empty() {
            return None;
        }

        let selected: Vec<&Value> = match rank {
            Some(r) => loads
                .iter()
                .filter(|e| e.get("dp_rank").and_then(|v| v.as_u64()) == Some(r as u64))
                .collect(),
            None => loads.iter().collect(),
        };
        // A rank that vanished from the payload is unknown, not saturated.
        if selected.is_empty() {
            return None;
        }

        let mut used: i64 = 0;
        let mut cap: i64 = 0;
        for entry in selected {
            for key in ["num_used_tokens", "num_waiting_uncached_tokens"] {
                used += entry.get(key).and_then(|v| v.as_i64()).unwrap_or(0);
            }
            cap += entry
                .get("max_total_num_tokens")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
        }
        if cap <= 0 {
            return None;
        }
        // Saturate rather than wrap: queued tokens can exceed capacity.
        Some(((used.max(0) as i128 * 1000 / cap as i128).min(u32::MAX as i128)) as u32)
    }

    /// Fetch saturation for every worker. Workers whose value cannot be
    /// determined are omitted, so consumers see absence rather than a wrong 0.
    pub async fn get_all_worker_saturation(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> HashMap<String, u32> {
        let mut out = HashMap::new();
        // Cache responses by base URL: under --dp-aware every rank of a server
        // shares one endpoint, so N ranks would otherwise mean N identical GETs.
        let mut fetched: HashMap<String, Option<Value>> = HashMap::new();

        for worker in worker_registry.get_all() {
            let (base, rank) = Self::split_worker_url(worker.url());

            if !fetched.contains_key(base) {
                let url = format!("{}/v1/loads?include=core", base);
                let body = match client.get(&url).timeout(REQUEST_TIMEOUT).send().await {
                    Ok(r) if r.status().is_success() => r.json::<Value>().await.ok(),
                    _ => None,
                };
                fetched.insert(base.to_string(), body);
            }

            if let Some(Some(json)) = fetched.get(base) {
                if let Some(p) = Self::extract_saturation_permille(json, rank) {
                    // Key by the FULL worker url (with @rank) so the policy can
                    // match it against worker.url() when filtering candidates.
                    out.insert(worker.url().to_string(), p);
                }
            }
        }
        out
    }

    /// Load = tokens already resident + tokens queued but not yet cached, which
    /// is the quantity that predicts KV pressure.
    fn extract_total_tokens(json: &Value, rank: Option<u32>) -> isize {
        // Preferred shape, if any engine ever emits it. Note this is inherently
        // server-wide, so it is only used when no rank was requested.
        if rank.is_none() {
            if let Some(n) = json
                .get("aggregate")
                .and_then(|a| a.get("total_tokens"))
                .and_then(|v| v.as_i64())
            {
                return n as isize;
            }
        }

        // Actual shape: per-DP-rank entries. Scope to one rank when asked,
        // otherwise sum the server.
        if let Some(loads) = json.get("loads").and_then(|v| v.as_array()) {
            if loads.is_empty() {
                return -1;
            }
            let mut total: i64 = 0;
            let mut saw_field = false;
            for entry in loads {
                if let Some(r) = rank {
                    if entry.get("dp_rank").and_then(|v| v.as_u64()) != Some(r as u64) {
                        continue;
                    }
                }
                for key in ["num_used_tokens", "num_waiting_uncached_tokens"] {
                    if let Some(v) = entry.get(key).and_then(|v| v.as_i64()) {
                        total += v;
                        saw_field = true;
                    }
                }
            }
            if saw_field {
                return total as isize;
            }
        }

        -1
    }

    pub async fn get_engine_metrics(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> EngineMetricsResult {
        let workers = worker_registry.get_all();

        if workers.is_empty() {
            return EngineMetricsResult::Err("No available workers".to_string());
        }

        let responses = fan_out(&workers, client, "metrics", reqwest::Method::GET).await;

        let mut metric_packs = Vec::new();
        for resp in responses {
            if let Ok(r) = resp.result {
                if r.status().is_success() {
                    if let Ok(text) = r.text().await {
                        metric_packs.push(MetricPack {
                            labels: vec![("worker_addr".into(), resp.url)],
                            metrics_text: text,
                        });
                    }
                }
            }
        }

        if metric_packs.is_empty() {
            return EngineMetricsResult::Err("All backend requests failed".to_string());
        }

        match crate::core::metrics_aggregator::aggregate_metrics(metric_packs) {
            Ok(text) => EngineMetricsResult::Ok(text),
            Err(e) => EngineMetricsResult::Err(format!("Failed to aggregate metrics: {}", e)),
        }
    }
}

/// Load monitoring service that periodically fetches worker loads
pub struct LoadMonitor {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: reqwest::Client,
    interval: Duration,
    tx: watch::Sender<HashMap<String, isize>>,
    rx: watch::Receiver<HashMap<String, isize>>,
    monitor_handle: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl LoadMonitor {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval_secs: u64,
    ) -> Self {
        let (tx, rx) = watch::channel(HashMap::new());

        Self {
            worker_registry,
            policy_registry,
            client,
            interval: Duration::from_secs(interval_secs),
            tx,
            rx,
            monitor_handle: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn start(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if handle_guard.is_some() {
            debug!("Load monitoring already running");
            return;
        }

        info!(
            "Starting load monitoring with interval: {:?}",
            self.interval
        );

        let worker_registry = Arc::clone(&self.worker_registry);
        let policy_registry = Arc::clone(&self.policy_registry);
        let client = self.client.clone();
        let interval = self.interval;
        let tx = self.tx.clone();

        let handle = tokio::spawn(async move {
            Self::monitor_loop(worker_registry, policy_registry, client, interval, tx).await;
        });

        *handle_guard = Some(handle);
    }

    pub async fn stop(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if let Some(handle) = handle_guard.take() {
            info!("Stopping load monitoring");
            handle.abort();
            let _ = handle.await; // Wait for task to finish
        }
    }

    pub fn subscribe(&self) -> watch::Receiver<HashMap<String, isize>> {
        self.rx.clone()
    }

    async fn monitor_loop(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval: Duration,
        tx: watch::Sender<HashMap<String, isize>>,
    ) {
        let mut interval_timer = tokio::time::interval(interval);

        loop {
            interval_timer.tick().await;

            let power_of_two_policies = policy_registry.get_all_power_of_two_policies();
            // Local addition: cache_aware also consumes load data now (for its
            // saturation gate), so the monitor must run when EITHER is active.
            // Upstream skipped the fetch entirely unless a power_of_two policy
            // existed, which silently starved any other load-aware policy.
            let cache_aware_policies = policy_registry.get_all_cache_aware_policies();

            if power_of_two_policies.is_empty() && cache_aware_policies.is_empty() {
                debug!("No load-aware policies found, skipping load fetch");
                continue;
            }

            let result = WorkerManager::get_all_worker_loads(&worker_registry, &client).await;

            let mut loads = HashMap::new();
            for load_info in result.loads {
                loads.insert(load_info.worker, load_info.load);
            }

            if !loads.is_empty() {
                debug!(
                    "Fetched loads from {} workers, updating {} PowerOfTwo policies",
                    loads.len(),
                    power_of_two_policies.len()
                );
                for policy in &power_of_two_policies {
                    policy.update_loads(&loads);
                }
                let _ = tx.send(loads);
            }

            // Local addition: feed cache_aware its saturation view.
            if !cache_aware_policies.is_empty() {
                let saturation =
                    WorkerManager::get_all_worker_saturation(&worker_registry, &client).await;
                if !saturation.is_empty() {
                    debug!(
                        "Fetched saturation for {} workers, updating {} cache_aware policies",
                        saturation.len(),
                        cache_aware_policies.len()
                    );
                    for policy in &cache_aware_policies {
                        policy.update_saturation(&saturation);
                    }
                }
            } else {
                warn!("No loads fetched from workers");
            }
        }
    }

    pub async fn is_running(&self) -> bool {
        let handle_guard = self.monitor_handle.lock().await;
        handle_guard.is_some()
    }
}

impl Drop for LoadMonitor {
    fn drop(&mut self) {
        if let Ok(mut handle_guard) = self.monitor_handle.try_lock() {
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
        }
    }
}
