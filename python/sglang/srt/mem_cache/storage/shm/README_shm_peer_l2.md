# HiCache `shm_peer_l2` — Cross-DP Peer L2 Cache

## Tổng quan

`shm_peer_l2` cho phép các DP rank trên cùng một node chia sẻ L2 host KV cache mà không cần copy data qua L3 store. Cả hai rank map cùng physical SHM pages; khi Rank A compute xong một prefix và flush lên L2, Rank B có thể đọc trực tiếp từ buffer đó chỉ bằng một `memcpy` trong SHM.

```
Không dùng shm_peer_l2:
  Rank A: [GPU L1] → [L2 buffer A]  ← private, Rank B không thấy
  Rank B: [GPU L1] → [L2 buffer B]  ← phải recompute prefix

Dùng shm_peer_l2:
  Rank A: [GPU L1] → [shared L2 SHM buffer]
  Rank B: [GPU L1] ←── memcpy ──── [shared L2 SHM buffer][slot của Rank A]
                         ↑ không cần recompute
```

### So sánh với `shm_cross_dp`

| | `shm_cross_dp` | `shm_peer_l2` |
|---|---|---|
| L2 buffer | Shared physical pages | Shared physical pages |
| L3 data store | SHM file chứa full page data (copy thêm 1 lần) | Không có |
| Write path | GPU → L2 → copy vào SHM L3 file | GPU → SHM L2 → ghi 288B pointer |
| Read path | SHM L3 → CPU buf → L2 → GPU | memcpy trong SHM → GPU |
| Memory dùng | shared L2 + SHM data copy | shared L2 + tiny index (~vài chục MB) |
| Interface | v0 (`batch_set` / `batch_get`) | v1 (`batch_set_v1` / `batch_get_v1`) |

Cả hai đều dùng chung `shm_shared_l2` allocator cho L2 buffer. `shm_peer_l2` không cần khai báo `shm_cross_dp` riêng — chỉ cần một flag là đủ.

---

## Điều kiện tiên quyết

- `--dp-size >= 2`
- Tất cả DP rank chạy trên **cùng node** — `/dev/shm` là tmpfs node-local; backend này không hoạt động cross-node
- `/dev/shm` đủ dung lượng: L2 SHM buffer + index + gen array (xem tính toán bên dưới)
- `--hicache-write-policy write_through` — `write_back` chỉ flush L2 khi evict khỏi L1, nên Rank B sẽ không thấy data kịp thời

---

## Launch command

```bash
python -m sglang.launch_server \
    --model-path <model> \
    --tp-size <tp> \
    --dp-size 2 \
    --hicache-size <L2_GB> \
    --hicache-write-policy write_through \
    --hicache-io-backend kernel \
    --hicache-mem-layout page_first \
    --hicache-storage-backend shm_peer_l2
```

**H200 / CUDA 12.8+ có `cudaMemcpyBatchAsync`:**

```bash
    --hicache-io-backend direct \
    --hicache-mem-layout page_first_direct \
```

**Tuỳ chỉnh index capacity và thư mục SHM:**

```bash
    --hicache-storage-backend-extra-config '{"peer_l2_capacity": 262144, "peer_l2_shm_dir": "/dev/shm"}'
```

> Nếu bỏ qua `peer_l2_capacity`, server tự tính theo `page_num × 4` (load factor 25%) và ghi log tại startup.

---

## Tính `peer_l2_capacity`

Index là direct-mapped hash table: mỗi key chiếm đúng 1 slot (288 byte). Quá nhỏ → collision → entry cũ bị overwrite → index miss dù data vẫn còn trong L2.

### Công thức

```
# MHA / GQA model:
size_per_token  = head_dim × num_kv_heads × num_layers × dtype_bytes × 2  (K+V)

# MLA model (DeepSeek):
size_per_token  = (kv_lora_rank + qk_rope_head_dim) × dtype_bytes × num_layers

page_size_bytes = size_per_token × page_size_tokens   # page_size_tokens default = 32

page_num        = hicache_size_GB × 1e9 / page_size_bytes

peer_l2_capacity = page_num × load_factor_inv
```

**`load_factor_inv`:**

| Giá trị | Load factor | Khi nào dùng |
|---|---|---|
| 2 | 50% | RAM tight |
| 4 | 25% | **Recommended** — mặc định tự tính |
| 8 | 12.5% | Prefix rất dài, cần hit rate tối đa |

### Script tính nhanh

```python
def calc_peer_l2_capacity(
    hicache_size_gb: float,
    page_size_tokens: int = 32,
    num_layers: int = 80,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    dtype_bytes: int = 2,       # bf16=2, fp8=1
    load_factor_inv: int = 4,
):
    size_per_token  = head_dim * num_kv_heads * num_layers * dtype_bytes * 2
    page_size_bytes = size_per_token * page_size_tokens
    page_num        = int(hicache_size_gb * 1e9 / page_size_bytes)
    capacity        = page_num * load_factor_inv
    index_mb        = capacity * 288 / 1024**2
    gen_mb          = page_num * 4 / 1024**2
    print(f"page_size       : {page_size_bytes / 1024**2:.2f} MB")
    print(f"page_num        : {page_num:,}")
    print(f"peer_l2_capacity: {capacity:,}  →  round up to next power of 2: {2 ** (capacity - 1).bit_length():,}")
    print(f"index file size : {index_mb:.1f} MB")
    print(f"gen array size  : {gen_mb:.2f} MB")
    print(f"/dev/shm usage  : L2_buffer + {index_mb + gen_mb:.1f} MB overhead")
```

Ví dụ — Llama 70B (GQA 8 heads), bf16, L2=475 GB:

```
page_size       : 10.00 MB
page_num        : 47,683
peer_l2_capacity: 190,732  →  262,144 (2^18)
index file size : 54.8 MB
gen array size  : 0.18 MB
/dev/shm usage  : 475 GB + ~55 MB overhead
```

---

## Files được tạo trong `/dev/shm`

```
/dev/shm/
├── sglang_l2_<model_slug>_pp<pp_rank>_<pp_size>.mmap       # L2 buffer (shared vật lý)
├── sglang_l2_<model_slug>_pp<pp_rank>_<pp_size>.mmap.lock  # flock serialize creation
├── sglang_l2_idx_<model_slug>_pp<pp_rank>_<pp_size>.bin    # index hash table (288B/entry)
├── sglang_l2_idx_<model_slug>_pp<pp_rank>_<pp_size>.bin.lock
├── sglang_l2_gen_<model_slug>_pp<pp_rank>_<pp_size>.bin    # generation counter (4B/page)
└── sglang_l2_gen_<model_slug>_pp<pp_rank>_<pp_size>.bin.lock
```

`<model_slug>` = model path với `/` thay bằng `-`, ví dụ: `deepseek-ai-DeepSeek-V3`.

Kiểm tra sau khi start:

```bash
# Phải có 1 file .mmap lớn (L2 buffer) + 2 file .bin nhỏ (index + gen)
ls -lh /dev/shm/sglang_l2*

# Kiểm tra dung lượng SHM còn trống
df -h /dev/shm
```

---

## Luồng hoạt động

### Request 1 → Rank A (cache miss, compute và ghi L2)

```
Rank A: match_prefix() → L1 MISS
      → prefetch_from_storage() → index MISS
      → GPU compute prefix tokens
      → write_backup(): GPU→CPU DMA → shared L2 SHM [slot thuộc partition của Rank A]
      → backup_thread: batch_set_v1()
          → _write_entry(key, dp_rank=A, slot_start)   ← 288B vào index SHM
```

### Request 2 → Rank B (cùng prefix, đọc từ Rank A)

```
Rank B: match_prefix() → L1 MISS
      → prefetch_from_storage() → batch_get_v1()
          Phase 1 (LOCK_SH):
            index lookup → dp_rank=A ≠ self.dp_rank=B → HIT
            check gen counter: current == stored → data còn valid
            peer_slot = entry.slot_start
          Phase 2 (no lock):
            memcpy(L2[own_slot], L2[peer_slot], page_bytes)
      → GPU DMA: own_slot → L1 GPU Rank B
      → SKIP compute
```

**Rank B chỉ tốn: 1 index lookup + 1 memcpy trong SHM + 1 DMA. Không recompute.**

### Generation counter — tránh đọc stale data

Mỗi lần Rank A cấp phát lại một slot (evict page cũ), counter của slot đó tăng lên 1. Rank B so sánh counter hiện tại với giá trị lưu trong index entry; nếu khác nhau → Rank A đã overwrite slot đó → bỏ qua, coi như miss thay vì copy data sai.

---

## Extra config

Truyền qua `--hicache-storage-backend-extra-config` dưới dạng JSON:

| Key | Default | Mô tả |
|---|---|---|
| `peer_l2_capacity` | auto (`page_num × 4`) | Số entry trong index hash table |
| `peer_l2_shm_dir` | `/dev/shm` | Thư mục chứa index và gen array files |

---

## Docker / container

Khi chạy trong container, cần mount `/dev/shm` với kích thước đủ lớn:

```yaml
# docker-compose.yaml
services:
  sglang:
    shm_size: "500gb"          # phải >= L2_buffer + vài chục MB overhead
    volumes:
      - /dev/shm:/dev/shm      # share /dev/shm giữa các container cùng node
```

Hoặc khi dùng `docker run`:

```bash
docker run --shm-size=500g -v /dev/shm:/dev/shm ...
```

Tất cả container chạy DP rank trên cùng node phải dùng chung `/dev/shm` của host (mount `-v /dev/shm:/dev/shm`).

---

## Những điểm cần chú ý

**1. `write_through` bắt buộc**
`write_back` chỉ flush L2 khi evict khỏi L1. Rank B sẽ không thấy data cho đến khi eviction xảy ra — có thể hàng phút sau. Luôn dùng `write_through` với `shm_peer_l2`.

**2. Rank B phải đợi Rank A flush xong**
Nếu Rank A chưa chạy xong `backup_thread`, Rank B sẽ miss index và phải compute lại. Đây là behavior đúng — không phải bug. Với `write_through`, khoảng trễ này rất nhỏ.

**3. Stale file sau khi restart**
Các file SHM tồn tại sau khi stop server. SGLang tự detect stale (kiểm tra magic, capacity, file size) và recreate khi khởi động lại. Không cần xóa thủ công, nhưng nếu muốn giải phóng `/dev/shm` ngay:

```bash
rm -f /dev/shm/sglang_l2*
```

**4. `exists()` chỉ return True khi `dp_rank != self.dp_rank`**
Mỗi rank tự quản lý L2 local của mình qua cache controller thông thường. Index chỉ dùng để tìm data của rank **khác**.

**5. Chỉ hoạt động trong cùng node**
`/dev/shm` là tmpfs node-local. Nếu DP rank nằm trên node khác nhau, dùng `mooncake` hoặc `nixl`.

**6. Không dùng `shm_cross_dp` kèm `shm_peer_l2`**
`shm_peer_l2` đã tự thiết lập shared L2 allocator (`shm_shared_l2`). Chỉ cần một flag `--hicache-storage-backend shm_peer_l2`.

---

## Verify hoạt động

```bash
# 1. Kiểm tra SHM files sau startup
ls -lh /dev/shm/sglang_l2*
# Kỳ vọng: 1 file .mmap lớn (~L2 size) + 4 file nhỏ (.bin, .bin.lock × 2)

# 2. Log startup — cả 2 rank phải báo cùng 1 index file
grep "HiCacheShmPeerL2" /path/to/sglang.log
# Rank 0: "HiCacheShmPeerL2: created index /dev/shm/sglang_l2_idx_... capacity=262144"
# Rank 1: "HiCacheShmPeerL2: attached to existing index /dev/shm/sglang_l2_idx_..."

# 3. Auto-computed capacity log (nếu không truyền peer_l2_capacity)
grep "auto-computed peer_l2_capacity" /path/to/sglang.log
# HiCacheShmPeerL2: auto-computed peer_l2_capacity=190732 (page_num=47683 × 4, index size=54.8 MB)

# 4. Collision rate log (mỗi 10,000 writes)
grep "collision rate" /path/to/sglang.log
# Warning nếu > 10%: tăng peer_l2_capacity

# 5. Kiểm tra dung lượng
df -h /dev/shm
```

---

## Tham khảo

- Implementation: `hicache_shm_peer_l2.py` (cùng thư mục)
- Shared allocator: `pool_host/common.py` — `ShmHostTensorAllocator`, `get_allocator_type()`
- SHM file allocation: `storage/mmap/mmap_allocator.py` — `alloc_shm`, `_open_named_shm_fd`
- L2 pool base: `pool_host/base.py` — slot partitioning, `dp_rank`, `dp_size`, `local_size`, `register_alloc_callback`
- Cache controller: `hiradix_cache.py` — `write_backup`, `_inc_hit_count`, `prefetch_from_storage`
