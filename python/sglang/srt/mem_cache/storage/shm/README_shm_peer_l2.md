# HiCache `shm_peer_l2` — Cross-DP Shared L2 Cache

## Tổng quan

`shm_peer_l2` cho phép các DP rank trên cùng một node chia sẻ L2 host KV cache mà không cần copy data qua L3 store. Cả hai rank map cùng physical SHM pages; khi Rank A compute xong một prefix và flush lên L2, Rank B có thể đọc trực tiếp từ buffer đó chỉ bằng một `memcpy` trong SHM.

```
Không dùng shm_peer_l2:
  Rank A: [L1 GPU] → [L2 mmap A]   ← private, Rank B không thấy
  Rank B: [L1 GPU] → [L2 mmap B]   ← private, phải recompute prefix

Dùng shm_peer_l2:
  Rank A: [L1 GPU] → [shared L2 SHM] ← cùng physical pages
  Rank B: [L1 GPU] ←─ memcpy ─────── [shared L2 SHM][slot của Rank A]
                        ↑ không cần recompute
```

### So sánh với `shm_cross_dp`

| | `shm_cross_dp` | `shm_peer_l2` |
|---|---|---|
| L2 buffer | Private per rank | Shared physical pages |
| L3 data store | SHM file chứa full page data | Không có |
| Write path | GPU → L2 → copy vào SHM L3 | GPU → SHM L2 → ghi 288B pointer |
| Read path | SHM L3 → CPU buf → L2 → GPU | memcpy trong SHM → GPU |
| Memory dùng | L2_A + L2_B + SHM_data ≈ 3× | Shared L2 + tiny index ≈ 1× |
| Interface | v0 (`batch_set` / `batch_get`) | v1 (`batch_set_v1` / `batch_get_v1`) |

---

## Điều kiện tiên quyết

- `dp_size >= 2`
- Tất cả DP rank chạy trên **cùng một node** (`/dev/shm` là local memory)
- `/dev/shm` đủ dung lượng: L2 SHM buffer + index file
- `--hicache-write-policy write_through` (bắt buộc — `write_back` không flush L2 ngay, Rank B sẽ không thấy)

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
    --hicache-storage-backend shm_peer_l2 \
    --hicache-storage-backend-extra-config '{"peer_l2_capacity": <capacity>}'
```

Trên H200 với CUDA 12.8+ (`cudaMemcpyBatchAsync` available):

```bash
    --hicache-io-backend direct \
    --hicache-mem-layout page_first_direct \
```

---

## Tính `peer_l2_capacity`

`peer_l2_capacity` là số entry của index hash table (direct-mapped). Quá nhỏ → collision → entry cũ bị overwrite → index miss dù data vẫn còn trong L2.

### Công thức

```
# MHA model (GQA/MHA):
size_per_token  = head_dim × num_kv_heads × num_layers × dtype_bytes × 2

# MLA model (DeepSeek):
size_per_token  = (kv_lora_rank + qk_rope_head_dim) × dtype_bytes × num_layers

page_size_bytes = size_per_token × page_size_tokens   # page_size_tokens default = 32

page_num        = hicache_size_GB × 1e9 / page_size_bytes

peer_l2_capacity = page_num × load_factor_inv
```

**`load_factor_inv`**:

| Giá trị | Load factor | Khi nào dùng |
|---|---|---|
| 2 | 50% | RAM tight, chấp nhận miss cao |
| 4 | 25% | **Recommended** cho production |
| 8 | 12.5% | Prefix dài, cần hit rate cao |

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
    print(f"page_size       : {page_size_bytes / 1024**2:.2f} MB")
    print(f"page_num        : {page_num:,}")
    print(f"peer_l2_capacity: {capacity:,}")
    print(f"index file size : {index_mb:.1f} MB")
```

Ví dụ — GLM 70B, bf16, L2=475GB:

```
page_size       : 10.00 MB
page_num        : 47,683
peer_l2_capacity: 190,732  →  làm tròn lên 2^18 = 262144
index file size : 52.5 MB
```

---

## Files được tạo trong `/dev/shm`

```
/dev/shm/
├── sglang_l2_<model>_pp<rank>_<size>.mmap       # L2 buffer (shared physical pages)
├── sglang_l2_<model>_pp<rank>_<size>.mmap.lock  # flock serialize creation
├── sglang_l2_idx_<model>_pp<rank>_<size>.bin    # index hash table (288B/entry)
└── sglang_l2_idx_<model>_pp<rank>_<size>.bin.lock
```

Kiểm tra sau khi start:

```bash
# Phải là 1 file .mmap dùng chung, không phải 2 file riêng
ls -lh /dev/shm/sglang_l2_*.mmap

# Kiểm tra dung lượng SHM còn trống
df -h /dev/shm
```

---

## Luồng hoạt động

### Request 1 → Rank A (cache miss)

```
Rank A: match_prefix() → L1 MISS
      → prefetch_from_storage() → batch_exists() → index MISS
      → GPU compute prefix
      → KV → L1 (device HBM)
      → _inc_hit_count() → hit_count >= write_through_threshold (=1)
      → write_backup(): GPU→CPU DMA → shared L2 SHM slot [Rank A partition]
      → backup_thread: batch_set_v1()
          → _write_entry(idx_slot, key, dp_rank=A, slot_start)   ← 288B vào index SHM
```

### Request 2 → Rank B (cùng prefix)

```
Rank B: match_prefix() → L1 MISS, L2 local MISS
      → prefetch_from_storage() → batch_exists()
          → index lookup: dp_rank=A ≠ self.dp_rank=B → HIT
      → _page_get_zero_copy() → batch_get_v1()
          Phase 1 (LOCK_SH): peer_slot = entry.slot_idx  ← Rank A's slot pointer
          Phase 2 (no lock):  memcpy(L2[own_slot], L2[peer_slot], page_bytes)
      → GPU DMA: shared L2 own_slot → L1 GPU Rank B
      → SKIP compute
```

**Rank B chỉ tốn: 1 memcpy trong SHM + 1 DMA, không recompute prefix.**

---

## Extra config

Truyền qua `--hicache-storage-backend-extra-config` dưới dạng JSON:

| Key | Default | Mô tả |
|---|---|---|
| `peer_l2_capacity` | `4096` | Số entry trong index hash table. Nên đặt theo công thức trên |
| `peer_l2_shm_dir` | `/dev/shm` | Thư mục chứa index SHM file |

```bash
--hicache-storage-backend-extra-config '{"peer_l2_capacity": 262144, "peer_l2_shm_dir": "/dev/shm"}'
```

---

## Những điểm cần chú ý

**1. `write_through` bắt buộc**
`write_back` chỉ flush L2 khi evict khỏi L1 → Rank B không thấy data cho đến khi eviction xảy ra. Dùng `write_through` để L2 được populate ngay sau khi compute.

**2. `peer_l2_capacity` quá nhỏ**
Index là direct-mapped: khi collision, entry cũ bị overwrite. Data vẫn còn trong L2 SHM nhưng không tìm được qua index → miss. Không có warning log; chỉ thấy hit rate thấp bất thường.

**3. Rank B phải đợi Rank A flush xong**
Nếu Rank A chưa chạy xong `backup_thread`, Rank B sẽ miss index và phải compute lại. Đây là behavior đúng — không phải bug.

**4. `exists()` chỉ return True khi `dp_rank != self.dp_rank`**
Mỗi rank tự đọc L2 local của mình qua L1/L2 path thông thường. Index chỉ dùng để tìm data của rank khác.

**5. Stale file sau khi restart**
Các file SHM tồn tại sau khi stop server. SGLang tự detect stale (kiểm tra kích thước không khớp) và recreate ở lần khởi động tiếp theo. Không cần xóa thủ công trừ khi muốn giải phóng `/dev/shm`.

**6. Chỉ hoạt động trên cùng node**
`/dev/shm` là tmpfs node-local. Nếu các DP rank nằm trên node khác nhau, dùng `mooncake` hoặc `nixl` thay thế.

---

## Verify hoạt động

```bash
# 1. Kiểm tra SHM file dùng chung
ls -lh /dev/shm/sglang_l2_*
# Kỳ vọng: 1 file .mmap lớn (~L2 size) + 1 file index .bin nhỏ (~vài chục MB)

# 2. Kiểm tra dung lượng
df -h /dev/shm

# 3. Prometheus metrics — cross-rank hit xuất hiện khi Rank B đọc được của Rank A
# hicache_host_used_tokens tăng nhanh hơn nếu sharing hoạt động

# 4. Log tại startup (cả 2 rank phải thấy cùng 1 SHM file)
grep "HiCacheShmPeerL2" /path/to/sglang.log
# Rank 0: "created index ... capacity=262144"
# Rank 1: "attached to existing index ..."
```

---

## Tham khảo

- Implementation: `hicache_shm_peer_l2.py` (cùng thư mục)
- Allocator: `pool_host/common.py` — `ShmHostTensorAllocator`, `get_allocator_type()`
- L2 pool base: `pool_host/base.py` — slot partitioning, `local_size`, `free_slots`
- Cache controller: `managers/cache_controller.py` — `_page_set_zero_copy`, `_page_get_zero_copy`, `backup_thread_func`
- Radix cache: `mem_cache/hiradix_cache.py` — `write_backup`, `_inc_hit_count`, `prefetch_from_storage`
