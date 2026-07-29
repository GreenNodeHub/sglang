# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""
Cross-DP-instance shared memory (SHM) storage backend for HiCache.

All DP instances on the same node map the same file under /dev/shm so that
KV pages written by any instance are immediately readable by all peers at
host-memory bandwidth.  Both instances read and write symmetrically — there
is no designated producer or consumer.

Why the default write policy works
────────────────────────────────────
The default write_policy is ``write_through_selective``.  Under this policy,
each DP instance's backup thread calls ``batch_set`` shortly after every
L1→L2 write-back, so the shared SHM is populated continuously — not just on
L2 eviction.  The other instance then finds pages in the SHM on its next
prefetch miss instead of recomputing them.

Interface note
──────────────
``cache_controller`` routes through ``_generic_page_set`` / ``_generic_page_get``
for backends that are not in the explicit v1/v2 allow-lists.  Those helpers call
the **v0** interface: ``batch_set(keys, tensors)`` and ``batch_get(keys, hints)``.
The v2 methods (``batch_set_v2`` / ``batch_get_v2``) are kept for completeness
but are not invoked by the main KV path.

SHM file layout
───────────────
[0 : HEADER_SIZE)
    Bytes  0-7  magic        uint64 LE   0xDEAD_BEEF_CAFE_0002
    Bytes  8-15 capacity     uint64 LE   number of page slots
    Bytes 16-23 page_bytes   uint64 LE   bytes per KV page
    Byte   24   ready        uint8       1 once init is complete
    Bytes 25-255 (reserved)

[HEADER_SIZE : HEADER_SIZE + capacity × KEY_SLOT_SIZE)
    Key-slot metadata, KEY_SLOT_SIZE (= 280) bytes per slot:
        Offset 0      valid     uint8     0 = empty, 1 = valid
        Offset 1-7    (pad)
        Offset 8-15   hash64    uint64 LE low-8 bytes of SHA-256(key)
        Offset 16-17  key_len   uint16 LE
        Offset 18-23  (pad)
        Offset 24-279 key_str   uint8[256] UTF-8, NUL-padded

[HEADER_SIZE + capacity × KEY_SLOT_SIZE : )
    Raw page data, page_bytes per slot.
    Slot i lives at offset (HEADER_SIZE + capacity × KEY_SLOT_SIZE) + i × page_bytes.

Eviction
────────
Direct-mapped: slot = hash64(key) % capacity.  A new page evicts whatever
occupied the slot previously.  No probe chains to maintain.

Concurrency
───────────
Writes  : threading.Lock (in-process) + fcntl.LOCK_EX (cross-process).
Reads   : fcntl.LOCK_SH (cross-process shared read lock).
"""
from __future__ import annotations

import fcntl
import hashlib
import logging
import mmap
import os
import struct
import threading
from typing import List, Optional

import numpy as np
import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)

logger = logging.getLogger(__name__)

# ── layout constants ──────────────────────────────────────────────────────────
_MAGIC = 0xDEAD_BEEF_CAFE_0002
_HEADER_SIZE = 256
_MAX_KEY_LEN = 256
_KEY_SLOT_SIZE = 280  # 1+7+8+2+6+256

# Offsets within a key slot
_SLOT_VALID = 0     # uint8
_SLOT_HASH = 8      # uint64 LE
_SLOT_KEYLEN = 16   # uint16 LE
_SLOT_KEYSTR = 24   # uint8[256]

# Offsets within the header
_HDR_MAGIC = 0      # uint64
_HDR_CAP = 8        # uint64
_HDR_PBYTES = 16    # uint64
_HDR_READY = 24     # uint8

_DEFAULT_CAPACITY = 4096


def _low64(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def _build_shm_suffix(cfg: HiCacheStorageConfig) -> str:
    """Derive the SHM name suffix exactly as the file backend does.

    All DP instances with the same TP/PP config produce the same suffix, so
    they map to the same SHM file.
    """
    model_name = "-".join(cfg.model_name.split("/")) if cfg.model_name else ""
    suffix = f"_{model_name}"
    if not cfg.is_mla_model:
        suffix += f"_{cfg.tp_rank}_{cfg.tp_size}"
    if cfg.pp_size > 1:
        suffix += f"_{cfg.pp_size}_{cfg.pp_rank}"
    if cfg.attn_cp_size > 1:
        suffix += f"_cp{cfg.attn_cp_rank}_{cfg.attn_cp_size}"
    return suffix


class HiCacheShmCrossDP(HiCacheStorage):
    """Cross-DP shared memory storage backend.

    Activated by setting ``--hicache-storage-backend shm_cross_dp``.

    Both DP instances write to and read from the same SHM file symmetrically.
    No extra flags are required; the default write_policy (write_through_selective)
    ensures pages are pushed to SHM shortly after every L1→L2 write-back.

    Extra config keys (via ``--hicache-storage-backend-config``):
        cross_dp_capacity  int   number of page slots (default 4096)
        cross_dp_shm_dir   str   directory for SHM files (default /dev/shm)
    """

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        mem_pool_host=None,
    ):
        self._cfg = storage_config
        self._capacity: int = _DEFAULT_CAPACITY
        self._shm_dir: str = "/dev/shm"

        if storage_config.extra_config:
            self._capacity = int(
                storage_config.extra_config.get("cross_dp_capacity", _DEFAULT_CAPACITY)
            )
            self._shm_dir = str(
                storage_config.extra_config.get("cross_dp_shm_dir", "/dev/shm")
            )
        # Fall back to /tmp on systems without /dev/shm (e.g. macOS CI)
        if not os.path.isdir(self._shm_dir):
            import tempfile

            self._shm_dir = tempfile.gettempdir()
            logger.warning(
                "HiCacheShmCrossDP: /dev/shm not found, using %s", self._shm_dir
            )

        suffix = _build_shm_suffix(storage_config)
        base_name = f"sglang_hicache_l2{suffix}"
        self._shm_path = os.path.join(self._shm_dir, base_name + ".bin")
        self._lock_path = self._shm_path + ".lock"

        # Initialized lazily in register_mem_pool_host
        self._page_bytes: Optional[int] = None
        self._mm: Optional[mmap.mmap] = None
        self._lock_fd = None
        self._write_mutex = threading.Lock()
        self._keys_raw: Optional[np.ndarray] = None  # (capacity, KEY_SLOT_SIZE) uint8
        self._data_raw: Optional[np.ndarray] = None  # (capacity, page_bytes) uint8
        self.registered_pools: dict = {}

    # ── pool registration ────────────────────────────────────────────────────

    def register_mem_pool_host(self, mem_pool_host):
        super().register_mem_pool_host(mem_pool_host)
        self._maybe_init(mem_pool_host)

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        super().register_mem_host_pool_v2(host_pool, host_pool_name)
        if host_pool_name == PoolName.KV and self._mm is None:
            self._maybe_init(host_pool)

    def _maybe_init(self, pool) -> None:
        if self._mm is not None:
            return
        page_bytes = int(pool.get_size_per_token() * pool.page_size)
        self._init_shm(page_bytes)

    # ── SHM initialisation ───────────────────────────────────────────────────

    def _init_shm(self, page_bytes: int) -> None:
        self._page_bytes = page_bytes
        key_table_bytes = self._capacity * _KEY_SLOT_SIZE
        data_table_bytes = self._capacity * page_bytes
        total_bytes = _HEADER_SIZE + key_table_bytes + data_table_bytes

        self._lock_fd = open(self._lock_path, "a+b")

        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
        try:
            stale = False
            if os.path.exists(self._shm_path):
                if os.path.getsize(self._shm_path) != total_bytes:
                    stale = True
                else:
                    with open(self._shm_path, "rb") as f:
                        hdr = f.read(32)
                    if len(hdr) >= 24:
                        magic, cap, pbytes = struct.unpack_from("<QQQ", hdr, 0)
                        if magic != _MAGIC or cap != self._capacity or pbytes != page_bytes:
                            stale = True
                    else:
                        stale = True
                if stale:
                    os.remove(self._shm_path)
                    logger.warning(
                        "HiCacheShmCrossDP: stale SHM at %s removed", self._shm_path
                    )

            needs_init = not os.path.exists(self._shm_path)
            fd = os.open(self._shm_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                if needs_init:
                    os.ftruncate(fd, total_bytes)
                self._mm = mmap.mmap(fd, total_bytes, access=mmap.ACCESS_WRITE)
            finally:
                os.close(fd)

            if needs_init:
                self._mm[:total_bytes] = b"\x00" * total_bytes
                struct.pack_into(
                    "<QQQ", self._mm, _HDR_MAGIC, _MAGIC, self._capacity, page_bytes
                )
                self._mm[_HDR_READY] = 1
                self._mm.flush()
                logger.info(
                    "HiCacheShmCrossDP: created %s  capacity=%d  page_bytes=%d  total=%.1f MB",
                    self._shm_path,
                    self._capacity,
                    page_bytes,
                    total_bytes / (1024**2),
                )
            else:
                logger.info(
                    "HiCacheShmCrossDP: attached to existing SHM %s", self._shm_path
                )
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

        # Build numpy views — they share physical memory with the mmap
        self._keys_raw = np.ndarray(
            shape=(self._capacity, _KEY_SLOT_SIZE),
            dtype=np.uint8,
            buffer=self._mm,
            offset=_HEADER_SIZE,
        )
        self._data_raw = np.ndarray(
            shape=(self._capacity, page_bytes),
            dtype=np.uint8,
            buffer=self._mm,
            offset=_HEADER_SIZE + key_table_bytes,
        )

    def _is_ready(self) -> bool:
        return self._mm is not None

    # ── key-slot helpers ─────────────────────────────────────────────────────

    def _slot_valid(self, slot: int) -> int:
        return int(self._keys_raw[slot, _SLOT_VALID])

    def _slot_hash64(self, slot: int) -> int:
        return int.from_bytes(
            bytes(self._keys_raw[slot, _SLOT_HASH: _SLOT_HASH + 8]), "little"
        )

    def _slot_key(self, slot: int) -> str:
        klen = int.from_bytes(
            bytes(self._keys_raw[slot, _SLOT_KEYLEN: _SLOT_KEYLEN + 2]), "little"
        )
        raw = bytes(self._keys_raw[slot, _SLOT_KEYSTR: _SLOT_KEYSTR + klen])
        return raw.decode("utf-8", errors="replace")

    def _find(self, key: str) -> Optional[int]:
        """Return slot index if key is present, else None.  Lock-free read."""
        h = _low64(key)
        slot = h % self._capacity
        if (
            self._slot_valid(slot) == 1
            and self._slot_hash64(slot) == h
            and self._slot_key(slot) == key
        ):
            return slot
        return None

    def _write_slot(self, slot: int, key: str, page_np: np.ndarray) -> None:
        """Overwrite slot with new key and page data.  Caller must hold write lock."""
        h = _low64(key)
        kb = key.encode("utf-8")[:_MAX_KEY_LEN]
        klen = len(kb)

        row = self._keys_raw[slot]
        # Invalidate first so readers see a clean empty slot during the write
        row[_SLOT_VALID] = 0
        row[_SLOT_HASH: _SLOT_HASH + 8] = np.frombuffer(
            h.to_bytes(8, "little"), dtype=np.uint8
        )
        row[_SLOT_KEYLEN: _SLOT_KEYLEN + 2] = np.frombuffer(
            klen.to_bytes(2, "little"), dtype=np.uint8
        )
        row[_SLOT_KEYSTR: _SLOT_KEYSTR + klen] = np.frombuffer(kb, dtype=np.uint8)
        # Write page data
        np.copyto(self._data_raw[slot], page_np)
        # Mark valid last — acts as a visibility fence for readers
        row[_SLOT_VALID] = 1

    # ── core v0 read/write ────────────────────────────────────────────────────

    def get(
        self,
        key: str,
        target_location=None,
        target_sizes=None,
    ) -> Optional[torch.Tensor]:
        """Return flat page tensor if key is in SHM, else None.

        ``target_location`` is used as a dtype hint (consistent with the file
        backend): if provided, the returned tensor is viewed as that dtype so
        that ``set_from_flat_data_page`` receives the expected element type.
        """
        if not self._is_ready():
            return None
        fcntl.flock(self._lock_fd, fcntl.LOCK_SH)
        try:
            slot = self._find(key)
            if slot is None:
                return None
            # Copy while holding the shared lock to avoid a partial overwrite
            page_bytes_copy = bytes(self._data_raw[slot])
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

        flat = torch.frombuffer(bytearray(page_bytes_copy), dtype=torch.uint8)
        if target_location is not None:
            flat = flat.view(target_location.dtype)
        return flat

    def set(
        self,
        key: str,
        value=None,
        target_location=None,
        target_sizes=None,
    ) -> bool:
        """Store a flat page tensor in SHM.

        ``value`` is the tensor returned by ``host_pool.get_data_page(idx, flat=True)``.
        The raw bytes are written to the slot at ``hash64(key) % capacity``,
        evicting whatever was there previously (direct-mapped).
        """
        if not self._is_ready() or value is None:
            return False
        if isinstance(value, torch.Tensor):
            page_np = value.contiguous().view(torch.uint8).numpy()
        else:
            page_np = np.frombuffer(bytes(value), dtype=np.uint8)

        if page_np.nbytes != self._page_bytes:
            logger.warning(
                "HiCacheShmCrossDP: page size mismatch %d vs %d for key %s, skipping",
                page_np.nbytes,
                self._page_bytes,
                key,
            )
            return False

        h = _low64(key)
        slot = h % self._capacity
        with self._write_mutex:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            try:
                self._write_slot(slot, key, page_np.reshape(-1))
            finally:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        return True

    def batch_get(
        self,
        keys: List[str],
        target_locations=None,
        target_sizes=None,
    ) -> List[Optional[torch.Tensor]]:
        """Called by ``_generic_page_get`` for reads."""
        targets = target_locations if target_locations is not None else [None] * len(keys)
        return [self.get(k, t) for k, t in zip(keys, targets)]

    def batch_set(
        self,
        keys: List[str],
        values=None,
        target_locations=None,
        target_sizes=None,
    ) -> bool:
        """Called by ``_generic_page_set`` for writes."""
        if values is None:
            return False
        return all(self.set(k, v) for k, v in zip(keys, values))

    def exists(self, key: str) -> bool:
        if not self._is_ready():
            return False
        fcntl.flock(self._lock_fd, fcntl.LOCK_SH)
        try:
            return self._find(key) is not None
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    # ── v2 interface (kept for completeness; not called by the main KV path) ──

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers=None,
        extra_info=None,
    ) -> PoolTransferResult:
        if not self._is_ready():
            return PoolTransferResult.empty()

        # Auxiliary pools (Mamba, SWA, …) are not stored in SHM.
        # Conservative: return empty if any auxiliary pool is required.
        for transfer in pool_transfers or []:
            if transfer.name != PoolName.KV:
                return PoolTransferResult.empty()

        fcntl.flock(self._lock_fd, fcntl.LOCK_SH)
        try:
            kv_pages = 0
            for k in keys:
                if self._find(k) is None:
                    break
                kv_pages += 1
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

        if kv_pages == 0:
            return PoolTransferResult.empty()
        return PoolTransferResult(kv_pages, {})

    def _write_page_v2(
        self, pool_name, key: str, host_pool, page_offset: int
    ) -> bool:
        if pool_name != PoolName.KV:
            return True
        page_tensor = host_pool.get_data_page(page_offset, flat=True)
        return self.set(key, page_tensor)

    def _read_page_v2(
        self, pool_name, key: str, host_pool, page_offset: int
    ) -> bool:
        if pool_name != PoolName.KV:
            return False
        dummy = host_pool.get_dummy_flat_data_page()
        page_tensor = self.get(key, target_location=dummy)
        if page_tensor is None:
            return False
        host_pool.set_from_flat_data_page(page_offset, page_tensor)
        return True

    def _batch_io_v2(
        self, transfers: List[PoolTransfer], op_fn
    ) -> dict[str, List[bool]]:
        results: dict = {}
        for transfer in transfers:
            name = transfer.name
            pool = self.registered_pools.get(name) or getattr(
                self, "mem_pool_host", None
            )
            if pool is None:
                results[str(name)] = [False] * len(transfer.keys or [])
                continue
            keys = transfer.keys or []
            page_size = getattr(pool, "page_size", 1) or 1
            host_indices = transfer.host_indices
            if host_indices is None or host_indices.numel() != len(keys) * page_size:
                results[str(name)] = [False] * len(keys)
                continue
            results[str(name)] = [
                op_fn(name, key, pool, int(host_indices[i * page_size].item()))
                for i, key in enumerate(keys)
            ]
        return results

    def batch_get_v2(self, transfers, extra_info=None):
        return self._batch_io_v2(transfers, self._read_page_v2)

    def batch_set_v2(self, transfers, extra_info=None):
        return self._batch_io_v2(transfers, self._write_page_v2)

    # ── misc ──────────────────────────────────────────────────────────────────

    def clear(self) -> bool:
        if not self._is_ready():
            return True
        with self._write_mutex:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            try:
                self._keys_raw[:, _SLOT_VALID] = 0
                self._mm.flush()
            finally:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        return True

    def __del__(self):
        try:
            if self._lock_fd is not None:
                self._lock_fd.close()
            if self._mm is not None:
                self._mm.close()
        except Exception:
            pass
