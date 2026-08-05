# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""
Cross-DP peer-L2 storage backend for HiCache.

When the L2 host pool is backed by shared memory (allocator_type="shm_shared_l2"),
all DP instances map the same physical SHM buffer.  HiCacheShmPeerL2 exploits
this by maintaining a lightweight shared index that maps

    key  →  (dp_rank, slot_start)

so each DP instance can read pages written by its peer directly out of the
shared L2 buffer without going all the way to an L3 disk/network store.

Index SHM file layout  (/dev/shm/sglang_l2_idx_*.bin)
───────────────────────────────────────────────────────
[0 : HEADER_SIZE)
    Bytes  0-7   magic      uint64 LE  0xCA_FE_FEED_0001
    Bytes  8-15  capacity   uint64 LE  number of index slots
    Byte   16    ready      uint8      1 once init is complete
    Bytes 17-255 reserved

[HEADER_SIZE : HEADER_SIZE + capacity × ENTRY_SIZE)
    Per-slot (ENTRY_SIZE = 288 bytes):
        Offset 0      valid     uint8     0=empty, 1=valid
        Offset 1      dp_rank   uint8     which DP rank wrote this entry
        Offset 2-7    pad
        Offset 8-15   slot_idx  uint64 LE first token index in host pool
        Offset 16-23  hash64    uint64 LE low-8 bytes of SHA-256(key)
        Offset 24-25  key_len   uint16 LE
        Offset 26-31  pad
        Offset 32-287 key_str   uint8[256]

Eviction: direct-mapped (slot = hash64(key) % capacity).
Locking:
    Index writes : threading.Lock (in-process) + fcntl.LOCK_EX (cross-process)
    Index reads  : fcntl.LOCK_SH (cross-process shared read lock)
    L2 buffer    : unprotected — stale reads are accepted (best-effort cache)
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
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)

logger = logging.getLogger(__name__)

_MAGIC = 0xCA_FE_FEED_0002  # bumped from 0001 when _E_GEN field was added
_HEADER_SIZE = 256
_ENTRY_SIZE = 288
_MAX_KEY_LEN = 256

# Byte offsets within a single index entry (total 288 bytes)
_E_VALID  = 0   # uint8
_E_DPRANK = 1   # uint8
_E_GEN    = 2   # uint32 LE  generation counter at write time (was pad)
# offset 6-7: pad (2 bytes)
_E_SLOT   = 8   # uint64 LE  first token index in host pool
_E_HASH   = 16  # uint64 LE  low-8 bytes of SHA-256(key)
_E_KEYLEN = 24  # uint16 LE
# offset 26-31: pad (6 bytes)
_E_KEYSTR = 32  # uint8[256]

_DEFAULT_CAPACITY = 4096
_COLLISION_LOG_INTERVAL = 10_000  # log collision stats every N writes


def _low64(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


class HiCacheShmPeerL2(HiCacheStorage):
    """Peer-L2 storage backend for cross-DP shared memory pools.

    Activated by ``--hicache-storage-backend shm_peer_l2``.  Requires the L2
    host pool to have been created with ``allocator_type="shm_shared_l2"``
    (i.e., ``--hicache-storage-backend shm_cross_dp``).

    Uses the V1 (zero-copy) interface: ``batch_set_v1`` records each key's
    host-pool slot index in a shared SHM index; ``batch_get_v1`` looks up the
    peer's slot and reads the page directly from the shared L2 buffer.

    Extra config keys (via ``--hicache-storage-backend-config``):
        peer_l2_capacity  int  index hash-table capacity (default 4096)
        peer_l2_shm_dir   str  directory for SHM files (default /dev/shm)
    """

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        mem_pool_host=None,
    ):
        self._cfg = storage_config
        self._capacity: int = _DEFAULT_CAPACITY
        self._capacity_user_set: bool = False
        self._shm_dir: str = "/dev/shm"
        if storage_config.extra_config:
            if "peer_l2_capacity" in storage_config.extra_config:
                self._capacity = int(storage_config.extra_config["peer_l2_capacity"])
                self._capacity_user_set = True
            self._shm_dir = str(
                storage_config.extra_config.get("peer_l2_shm_dir", "/dev/shm")
            )
        if not os.path.isdir(self._shm_dir):
            import tempfile

            self._shm_dir = tempfile.gettempdir()
            logger.warning(
                "HiCacheShmPeerL2: /dev/shm not found, using %s", self._shm_dir
            )

        # Deterministic index file name — same for all DP ranks.
        model_slug = "-".join((storage_config.model_name or "").split("/"))
        suffix = f"_{model_slug}_pp{storage_config.pp_rank}_{storage_config.pp_size}"
        self._idx_path = os.path.join(
            self._shm_dir, f"sglang_l2_idx{suffix}.bin"
        )
        self._lock_path = self._idx_path + ".lock"

        # Populated in register_mem_pool_host
        self._dp_rank: int = 0
        self._dp_size: int = 1
        self._page_size: int = 1
        self._mm: Optional[mmap.mmap] = None
        self._lock_fd = None
        self._write_mutex = threading.Lock()
        self._entries: Optional[np.ndarray] = None  # (capacity, ENTRY_SIZE) uint8
        self.registered_pools: dict = {}

        # Generation array SHM — tracks slot reuse to detect stale index entries.
        # _gen_arr[page_idx] is incremented each time that page slot is reallocated.
        self._gen_mm: Optional[mmap.mmap] = None
        self._gen_arr: Optional[np.ndarray] = None  # uint32[page_num], in shared memory
        self._gen_path: Optional[str] = None

        # Collision tracking — local to this process, not cross-rank.
        self._write_count: int = 0
        self._collision_count: int = 0

        if mem_pool_host is not None:
            self.register_mem_pool_host(mem_pool_host)

    # ── pool registration ────────────────────────────────────────────────────

    def register_mem_pool_host(self, mem_pool_host):
        super().register_mem_pool_host(mem_pool_host)
        self._page_size = mem_pool_host.page_size
        self._dp_rank = mem_pool_host.dp_rank
        self._dp_size = mem_pool_host.dp_size
        if not self._capacity_user_set:
            # Auto-compute: all DP ranks write to the same index, so total
            # active keys ≈ page_num (both partitions combined). ×4 gives a
            # 25% load factor on the direct-mapped table, keeping collision
            # rate low without burning too much /dev/shm.
            self._capacity = mem_pool_host.page_num * 4
            logger.info(
                "HiCacheShmPeerL2: auto-computed peer_l2_capacity=%d "
                "(page_num=%d × 4, index size=%.1f MB); "
                "override with peer_l2_capacity in --hicache-storage-backend-extra-config",
                self._capacity,
                mem_pool_host.page_num,
                self._capacity * _ENTRY_SIZE / 1024**2,
            )
        if self._mm is None:
            self._init_shm()
        if self._gen_mm is None:
            self._init_gen_shm(mem_pool_host.page_num)
        mem_pool_host.register_alloc_callback(self._alloc_callback)

    def _init_shm(self) -> None:
        total_bytes = _HEADER_SIZE + self._capacity * _ENTRY_SIZE

        self._lock_fd = open(self._lock_path, "a+b")
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
        try:
            stale = False
            if os.path.exists(self._idx_path):
                if os.path.getsize(self._idx_path) != total_bytes:
                    stale = True
                else:
                    with open(self._idx_path, "rb") as f:
                        hdr = f.read(16)
                    if len(hdr) >= 16:
                        magic, cap = struct.unpack_from("<QQ", hdr, 0)
                        if magic != _MAGIC or cap != self._capacity:
                            stale = True
                    else:
                        stale = True
                if stale:
                    os.remove(self._idx_path)
                    logger.warning(
                        "HiCacheShmPeerL2: stale index at %s removed", self._idx_path
                    )

            needs_init = not os.path.exists(self._idx_path)
            fd = os.open(self._idx_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                if needs_init:
                    os.ftruncate(fd, total_bytes)
                self._mm = mmap.mmap(fd, total_bytes, access=mmap.ACCESS_WRITE)
            finally:
                os.close(fd)

            if needs_init:
                self._mm[:total_bytes] = b"\x00" * total_bytes
                struct.pack_into("<QQ", self._mm, 0, _MAGIC, self._capacity)
                self._mm[16] = 1
                self._mm.flush()
                logger.info(
                    "HiCacheShmPeerL2: created index %s  capacity=%d  size=%.1f kB",
                    self._idx_path,
                    self._capacity,
                    total_bytes / 1024,
                )
            else:
                logger.info(
                    "HiCacheShmPeerL2: attached to existing index %s",
                    self._idx_path,
                )
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

        self._entries = np.ndarray(
            shape=(self._capacity, _ENTRY_SIZE),
            dtype=np.uint8,
            buffer=self._mm,
            offset=_HEADER_SIZE,
        )

    def _init_gen_shm(self, page_num: int) -> None:
        """Create or attach to the generation-counter SHM array (uint32[page_num])."""
        self._gen_path = self._idx_path.replace("sglang_l2_idx", "sglang_l2_gen")
        gen_lock_path = self._gen_path + ".lock"
        size_bytes = page_num * 4  # one uint32 per page

        gen_lock_fd = open(gen_lock_path, "a+b")
        fcntl.flock(gen_lock_fd, fcntl.LOCK_EX)
        try:
            stale = (
                os.path.exists(self._gen_path)
                and os.path.getsize(self._gen_path) != size_bytes
            )
            if stale:
                os.remove(self._gen_path)
                logger.warning(
                    "HiCacheShmPeerL2: stale gen array at %s removed", self._gen_path
                )

            needs_init = not os.path.exists(self._gen_path)
            fd = os.open(self._gen_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                if needs_init:
                    os.ftruncate(fd, size_bytes)
                self._gen_mm = mmap.mmap(fd, size_bytes, access=mmap.ACCESS_WRITE)
            finally:
                os.close(fd)

            if needs_init:
                self._gen_mm[:size_bytes] = b"\x00" * size_bytes
                self._gen_mm.flush()
                logger.info(
                    "HiCacheShmPeerL2: created gen array %s  pages=%d  size=%.1f kB",
                    self._gen_path,
                    page_num,
                    size_bytes / 1024,
                )
            else:
                logger.info(
                    "HiCacheShmPeerL2: attached to existing gen array %s", self._gen_path
                )
        finally:
            fcntl.flock(gen_lock_fd, fcntl.LOCK_UN)
            gen_lock_fd.close()

        self._gen_arr = np.ndarray(
            shape=(page_num,), dtype=np.uint32, buffer=self._gen_mm
        )

    def _alloc_callback(self, select_index: "torch.Tensor") -> None:
        """Increment the generation for each reallocated page slot.

        Called by HostKVCache.alloc() just before returning the new indices.
        Incrementing here (before new data is written) ensures Rank B sees a
        mismatched generation and treats the slot as a miss instead of reading
        data from the previous occupant.
        """
        if self._gen_arr is None:
            return
        page_size = self._page_size
        # select_index holds token indices; pages start every page_size entries.
        page_starts = select_index[::page_size].cpu().numpy()
        page_idxs = page_starts // page_size
        for idx in page_idxs:
            self._gen_arr[idx] += 1  # uint32 write to aligned addr is atomic on x86/ARM

    # ── index entry helpers ──────────────────────────────────────────────────

    def _entry_valid(self, slot: int) -> bool:
        return int(self._entries[slot, _E_VALID]) == 1

    def _entry_dp_rank(self, slot: int) -> int:
        return int(self._entries[slot, _E_DPRANK])

    def _entry_slot_idx(self, slot: int) -> int:
        return int.from_bytes(
            bytes(self._entries[slot, _E_SLOT : _E_SLOT + 8]), "little"
        )

    def _entry_hash64(self, slot: int) -> int:
        return int.from_bytes(
            bytes(self._entries[slot, _E_HASH : _E_HASH + 8]), "little"
        )

    def _entry_generation(self, slot: int) -> int:
        return int.from_bytes(
            bytes(self._entries[slot, _E_GEN : _E_GEN + 4]), "little"
        )

    def _entry_key(self, slot: int) -> str:
        klen = int.from_bytes(
            bytes(self._entries[slot, _E_KEYLEN : _E_KEYLEN + 2]), "little"
        )
        return bytes(
            self._entries[slot, _E_KEYSTR : _E_KEYSTR + klen]
        ).decode("utf-8", errors="replace")

    def _find(self, key: str) -> Optional[int]:
        """Return slot index if *key* is present, else None.  Assumes shared lock held."""
        h = _low64(key)
        slot = h % self._capacity
        if (
            self._entry_valid(slot)
            and self._entry_hash64(slot) == h
            and self._entry_key(slot) == key
        ):
            return slot
        return None

    def _write_entry(
        self, slot: int, key: str, dp_rank: int, slot_idx: int, generation: int
    ) -> None:
        """Overwrite one index entry.  Caller must hold write lock."""
        h = _low64(key)
        kb = key.encode("utf-8")[:_MAX_KEY_LEN]
        klen = len(kb)
        row = self._entries[slot]
        row[_E_VALID] = 0  # invalidate first so readers see a clean slot
        row[_E_DPRANK] = dp_rank
        row[_E_GEN : _E_GEN + 4] = np.frombuffer(
            generation.to_bytes(4, "little"), dtype=np.uint8
        )
        row[_E_SLOT : _E_SLOT + 8] = np.frombuffer(
            slot_idx.to_bytes(8, "little"), dtype=np.uint8
        )
        row[_E_HASH : _E_HASH + 8] = np.frombuffer(
            h.to_bytes(8, "little"), dtype=np.uint8
        )
        row[_E_KEYLEN : _E_KEYLEN + 2] = np.frombuffer(
            klen.to_bytes(2, "little"), dtype=np.uint8
        )
        row[_E_KEYSTR : _E_KEYSTR + klen] = np.frombuffer(kb, dtype=np.uint8)
        row[_E_VALID] = 1  # mark valid last — acts as a visibility fence

    # ── V1 (zero-copy slot-index) interface ─────────────────────────────────

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info=None,
    ) -> List[bool]:
        """Record each key's first-token slot index in the shared index.

        Called by the backup thread after pages are written into L2.  Only
        records the slot pointer — the page data stays in the shared L2 buffer.
        """
        if self._mm is None:
            return [False] * len(keys)
        results: List[bool] = []
        with self._write_mutex:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            try:
                for i, key in enumerate(keys):
                    try:
                        slot_start = int(host_indices[i * self._page_size].item())
                        h = _low64(key)
                        idx_slot = h % self._capacity
                        self._write_count += 1
                        if (
                            self._entry_valid(idx_slot)
                            and self._entry_hash64(idx_slot) != h
                        ):
                            self._collision_count += 1
                        page_idx = slot_start // self._page_size
                        gen = int(self._gen_arr[page_idx]) if self._gen_arr is not None else 0
                        self._write_entry(idx_slot, key, self._dp_rank, slot_start, gen)
                        results.append(True)
                    except Exception as e:
                        logger.warning(
                            "HiCacheShmPeerL2 set error for key %s: %s", key, e
                        )
                        results.append(False)
            finally:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        self._maybe_log_collision_stats()
        return results

    def _maybe_log_collision_stats(self) -> None:
        if self._write_count == 0 or self._write_count % _COLLISION_LOG_INTERVAL != 0:
            return
        rate = self._collision_count / self._write_count
        if rate > 0.10:
            logger.warning(
                "HiCacheShmPeerL2: high index collision rate %.1f%% "
                "(%d/%d writes) — increase peer_l2_capacity (current=%d) "
                "to reduce cache misses",
                rate * 100,
                self._collision_count,
                self._write_count,
                self._capacity,
            )
        else:
            logger.debug(
                "HiCacheShmPeerL2: collision rate %.1f%% (%d/%d writes)",
                rate * 100,
                self._collision_count,
                self._write_count,
            )

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info=None,
    ) -> List[bool]:
        """Load each key from the peer's L2 partition into our local slots.

        Two-phase:
        1. Index look-up under LOCK_SH — find (peer_dp_rank, peer_slot_start).
        2. Data copy without lock — memcpy within the shared SHM buffer.
           A concurrent peer eviction of that slot is accepted as a best-effort
           cache miss (corrupted data would appear as a miss on the next decode).
        """
        if self._mm is None:
            return [False] * len(keys)

        pool = self.mem_pool_host

        # Phase 1: read all peer slot indices under a single shared lock.
        # For each hit, also verify the generation counter matches: if the peer
        # has reallocated that L2 slot since the index was written, the stored
        # generation will differ from the current one in the gen array, and we
        # treat the entry as a miss rather than copying stale data.
        peer_slots: List[Optional[int]] = []
        fcntl.flock(self._lock_fd, fcntl.LOCK_SH)
        try:
            for key in keys:
                idx_slot = self._find(key)
                if idx_slot is None or self._entry_dp_rank(idx_slot) == self._dp_rank:
                    peer_slots.append(None)
                    continue
                peer_slot = self._entry_slot_idx(idx_slot)
                if self._gen_arr is not None:
                    stored_gen = self._entry_generation(idx_slot)
                    current_gen = int(self._gen_arr[peer_slot // self._page_size])
                    if stored_gen != current_gen:
                        peer_slots.append(None)  # slot was reused — stale entry
                        continue
                peer_slots.append(peer_slot)
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

        # Phase 2: copy pages within the shared L2 buffer.
        results: List[bool] = []
        for i, peer_slot in enumerate(peer_slots):
            if peer_slot is None:
                results.append(False)
                continue
            try:
                page_data = pool.get_data_page(peer_slot, flat=True)
                local_slot = int(host_indices[i * self._page_size].item())
                pool.set_from_flat_data_page(local_slot, page_data)
                results.append(True)
            except Exception as e:
                logger.warning(
                    "HiCacheShmPeerL2 read error for key %s: %s", keys[i], e
                )
                results.append(False)
        return results

    # ── V0 stubs — required by the abstract base class ───────────────────────

    def get(self, key, target_location=None, target_sizes=None):
        """Not used — HiCacheShmPeerL2 operates exclusively via the V1 path."""
        return None

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        return [None] * len(keys)

    def set(self, key, value=None, target_location=None, target_sizes=None):
        return False

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None):
        return False

    # ── existence check ──────────────────────────────────────────────────────

    def exists(self, key: str) -> bool:
        """Return True if the peer dp_rank has an index entry for *key*."""
        if self._mm is None:
            return False
        fcntl.flock(self._lock_fd, fcntl.LOCK_SH)
        try:
            idx_slot = self._find(key)
            return idx_slot is not None and self._entry_dp_rank(idx_slot) != self._dp_rank
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers=None,
        extra_info=None,
    ) -> PoolTransferResult:
        """Only KV pages are tracked; auxiliary pools are not supported."""
        for transfer in pool_transfers or []:
            if transfer.name != PoolName.KV:
                return PoolTransferResult.empty()

        if self._mm is None:
            return PoolTransferResult.empty()

        fcntl.flock(self._lock_fd, fcntl.LOCK_SH)
        try:
            kv_pages = 0
            for key in keys:
                idx_slot = self._find(key)
                if idx_slot is None or self._entry_dp_rank(idx_slot) == self._dp_rank:
                    break
                kv_pages += 1
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

        if kv_pages == 0:
            return PoolTransferResult.empty()
        return PoolTransferResult(kv_pages, {})

    # ── misc ──────────────────────────────────────────────────────────────────

    def clear(self) -> bool:
        if self._mm is None:
            return True
        with self._write_mutex:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            try:
                self._entries[:, _E_VALID] = 0
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
