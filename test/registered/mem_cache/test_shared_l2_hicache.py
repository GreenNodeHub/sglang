"""
Unit tests for the shared-L2 HiCache implementation (Steps 1–6).

Each test class guards a concrete failure mode:

1. TestNamedShmAllocation   — alloc_shm(shm_name=…) creates a file whose
   physical pages are shared by any number of mmap objects; MAP_PRIVATE or a
   wrong fallback flag (e.g. MAP_NOCACHE on macOS) would turn these red.

2. TestShmAllocatorFactory  — ShmHostTensorAllocator and get_allocator_type
   wiring; if the factory ignored shm_name/dp_rank or the "shm_cross_dp" →
   "shm_shared_l2" mapping were dropped, these fail.

3. TestSlotPartitioning     — HostKVCache free_slots partitioning invariants
   (disjoint ranges, contiguous boundary, equal sizes); a copy-paste error in
   the slot_start / local_size formula would break them.

4. TestHiCacheShmPeerL2    — index-SHM CRUD: set records entries, exists()
   returns True only for peer dp_rank, get copies pages within the shared
   buffer, clear invalidates all entries; any one of these failing silently
   would make the peer-L2 lookup return wrong data or always miss.

5. TestCrossProcessSharedL2 — two independent OS processes share one named
   SHM file; the writer's batch_set_v1 is visible to the reader's
   batch_get_v1; guards the OS-level MAP_SHARED + fcntl serialisation.
"""

import glob
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
import uuid
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig, PoolName
from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.pool_host.common import (
    ShmHostTensorAllocator,
    get_allocator_from_storage,
    get_allocator_type,
)
from sglang.srt.mem_cache.storage.mmap import _named_shm_dir, alloc_shm
from sglang.srt.mem_cache.storage.shm.hicache_shm_peer_l2 import HiCacheShmPeerL2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=45, suite="base-a-test-cpu")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Absolute path to the repo's python/ directory — used to make sglang
# importable in subprocesses when running from source.
_SGLANG_PYTHON = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python")
)


def _make_storage_config(model_tag: str) -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=model_tag,
    )


def _cleanup_shm(shm_dir: str, *patterns: str) -> None:
    for pat in patterns:
        for p in glob.glob(os.path.join(shm_dir, pat)):
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Minimal HostKVCache subclass (no GPU required)
# ---------------------------------------------------------------------------

class _MinimalHostKVCache(HostKVCache):
    """Concrete HostKVCache that allocates nothing on the device.

    Lets us exercise the slot-partitioning logic in base.py without
    instantiating MHATokenToKVPoolHost or touching GPU memory.
    """

    _BYTES_PER_TOKEN = 64

    def get_size_per_token(self) -> int:
        return self._BYTES_PER_TOKEN

    def init_kv_buffer(self) -> torch.Tensor:
        # Return a throwaway CPU tensor — partitioning math lives in clear().
        return torch.zeros(4)

    def load_to_device_per_layer(self, *a, **kw) -> None:
        pass

    def backup_from_device_all_layer(self, *a, **kw) -> None:
        pass

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        n = self.page_size * self._BYTES_PER_TOKEN // 4
        return torch.zeros(n)

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(self.page_size * self._BYTES_PER_TOKEN // 4)

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        pass


def _make_device_pool() -> MagicMock:
    pool = MagicMock()
    pool.store_dtype = torch.float16
    pool.size = 200
    pool.start_layer = 0
    pool.end_layer = 2
    pool.layer_shard_enabled = False
    return pool


def _make_partitioned_cache(dp_rank: int, dp_size: int, page_size: int = 4):
    """Instantiate _MinimalHostKVCache with the given DP partition parameters."""
    with patch("psutil.virtual_memory") as mock_mem:
        mock_mem.return_value.available = 100 * (1024**3)  # 100 GB
        return _MinimalHostKVCache(
            device_pool=_make_device_pool(),
            host_to_device_ratio=1.0,
            host_size=0,
            page_size=page_size,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
            dp_rank=dp_rank,
            dp_size=dp_size,
        )


# ---------------------------------------------------------------------------
# Mock pool host for HiCacheShmPeerL2 tests
# ---------------------------------------------------------------------------

class _FlatBufPool:
    """Minimal pool host backed by a flat CPU tensor."""

    def __init__(self, buf: torch.Tensor, dp_rank: int, page_size: int):
        self._buf = buf
        self.dp_rank = dp_rank
        self.dp_size = 2
        self.page_size = page_size
        self._feat = buf.shape[1] if buf.dim() == 2 else buf.numel() // 32
        n_tokens = buf.shape[0] if buf.dim() == 2 else buf.numel() // self._feat
        self.page_num = n_tokens // page_size

    def get_data_page(self, idx: int, flat: bool = True) -> torch.Tensor:
        d = self._buf[idx : idx + self.page_size, :]
        return d.flatten() if flat else d

    def set_from_flat_data_page(self, idx: int, page: torch.Tensor) -> None:
        self._buf[idx : idx + self.page_size, :] = page.reshape(
            self.page_size, self._feat
        )

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(self.page_size * self._feat)


# ============================================================================
# Test 1 — Named SHM allocation
# ============================================================================

class TestNamedShmAllocation(CustomTestCase):
    """alloc_shm(shm_name=…) creates a file backed by OS shared memory."""

    def setUp(self):
        self._name = f"ut_shm_{uuid.uuid4().hex[:8]}"
        self._fds: list[int] = []

    def tearDown(self):
        for fd in self._fds:
            try:
                os.close(fd)
            except OSError:
                pass
        _cleanup_shm(
            _named_shm_dir(),
            f"{self._name}.mmap",
            f"{self._name}.mmap.lock",
        )

    def test_named_file_is_created_on_disk(self):
        """alloc_shm creates a persistent file under /dev/shm (or /tmp)."""
        _, fd, _ = alloc_shm((10,), torch.float32, shm_name=self._name)
        self._fds.append(fd)
        expected = os.path.join(_named_shm_dir(), f"{self._name}.mmap")
        self.assertTrue(
            os.path.exists(expected),
            f"Expected SHM file not found at {expected}",
        )

    def test_two_mappings_of_same_name_share_physical_pages(self):
        """Writes through one mapping are immediately visible through the other.

        Guards against MAP_PRIVATE or a wrong flag (e.g. MAP_NOCACHE on macOS
        that would give ENOMEM or silently create a private copy).
        """
        dims = (20,)
        t1, fd1, _ = alloc_shm(dims, torch.float32, shm_name=self._name)
        t2, fd2, _ = alloc_shm(dims, torch.float32, shm_name=self._name)
        self._fds.extend([fd1, fd2])

        t1.fill_(3.14)
        self.assertTrue(
            torch.allclose(t2, t1),
            "t2 did not reflect t1's write — memory is not shared (check MAP_SHARED)",
        )
        t2[0] = 99.0
        self.assertAlmostEqual(t1[0].item(), 99.0, places=3)

    def test_anonymous_shm_backward_compat(self):
        """alloc_shm(shm_name=None) still returns a usable anonymous mmap."""
        t, fd, _ = alloc_shm((8,), torch.float32, shm_name=None)
        self._fds.append(fd)
        t.fill_(7.0)
        self.assertAlmostEqual(t.sum().item(), 56.0, places=3)

    def test_stale_file_wrong_size_is_replaced(self):
        """alloc_shm recreates the SHM file when it has an unexpected size."""
        path = os.path.join(_named_shm_dir(), f"{self._name}.mmap")
        # Write a too-small file so alloc_shm considers it stale.
        with open(path, "wb") as f:
            f.write(b"\x00" * 16)

        _, fd, _ = alloc_shm((100,), torch.float32, shm_name=self._name)
        self._fds.append(fd)
        # File must have been replaced with the correct size.
        self.assertGreater(os.path.getsize(path), 16)


# ============================================================================
# Test 2 — ShmHostTensorAllocator and factory
# ============================================================================

class TestShmAllocatorFactory(CustomTestCase):
    """ShmHostTensorAllocator creation, factory helpers, and type mapping."""

    def setUp(self):
        self._name = f"ut_alloc_{uuid.uuid4().hex[:8]}"

    def tearDown(self):
        d = _named_shm_dir()
        _cleanup_shm(
            d,
            f"{self._name}.mmap",
            f"{self._name}.mmap.lock",
            f"{self._name}_f.mmap",
            f"{self._name}_f.mmap.lock",
        )

    def test_two_named_allocators_share_memory(self):
        """Two allocators with the same shm_name map the same physical pages."""
        dims = (16, 4)
        a0 = ShmHostTensorAllocator(shm_name=self._name, dp_rank=0, dp_size=2)
        a1 = ShmHostTensorAllocator(shm_name=self._name, dp_rank=1, dp_size=2)
        t0 = a0.allocate(dims, dtype=torch.float16, device="cpu")
        t1 = a1.allocate(dims, dtype=torch.float16, device="cpu")

        t0.fill_(1.5)
        self.assertTrue(
            torch.allclose(t1.float(), t0.float()),
            "Allocators with same shm_name do not share physical memory",
        )

    def test_factory_shm_shared_l2_passes_kwargs(self):
        """get_allocator_from_storage('shm_shared_l2') preserves shm_name/dp_rank/dp_size."""
        a = get_allocator_from_storage(
            "shm_shared_l2",
            shm_name=self._name + "_f",
            dp_rank=1,
            dp_size=3,
        )
        self.assertIsInstance(a, ShmHostTensorAllocator)
        self.assertEqual(a.shm_name, self._name + "_f")
        self.assertEqual(a.dp_rank, 1)
        self.assertEqual(a.dp_size, 3)

    def test_factory_shm_creates_anonymous_allocator(self):
        """get_allocator_from_storage('shm') creates an anonymous allocator."""
        a = get_allocator_from_storage("shm")
        self.assertIsInstance(a, ShmHostTensorAllocator)
        self.assertIsNone(a.shm_name)

    def test_shm_cross_dp_maps_to_shm_shared_l2(self):
        """'shm_cross_dp' storage backend maps to 'shm_shared_l2' allocator type.

        Guards the wiring between --hicache-storage-backend shm_cross_dp and
        the L2 allocator; if this mapping breaks, cross-DP L2 sharing silently
        falls back to private memory.
        """

        class _Args:
            hicache_storage_backend = "shm_cross_dp"
            hicache_storage_backend_extra_config = None

        self.assertEqual(get_allocator_type(_Args()), "shm_shared_l2")

    def test_shm_peer_l2_maps_to_shm_shared_l2(self):
        """'shm_peer_l2' storage backend maps to 'shm_shared_l2' allocator type.

        Guards the wiring between --hicache-storage-backend shm_peer_l2 and
        the shared L2 allocator. Without this mapping, both DP ranks get
        separate private mmap regions — batch_get_v1 reads from the wrong
        buffer and silently returns stale data.
        """

        class _Args:
            hicache_storage_backend = "shm_peer_l2"
            hicache_storage_backend_extra_config = None

        self.assertEqual(get_allocator_type(_Args()), "shm_shared_l2")

    def test_shm_backend_stays_shm(self):
        """'shm' backend must NOT be silently upgraded to 'shm_shared_l2'."""

        class _Args:
            hicache_storage_backend = "shm"
            hicache_storage_backend_extra_config = None

        self.assertEqual(get_allocator_type(_Args()), "shm")

    def test_shm_peer_l2_is_registered_in_backend_factory(self):
        """shm_peer_l2 must appear in StorageBackendFactory._registry.

        Guards against adding the class but forgetting to register it; a
        missing registration produces a runtime ValueError on startup.
        """
        from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory

        self.assertIn(
            "shm_peer_l2",
            StorageBackendFactory._registry,
            "shm_peer_l2 not in StorageBackendFactory._registry — "
            "add the register_backend() call to backend_factory.py",
        )


# ============================================================================
# Test 3 — Slot partitioning
# ============================================================================

class TestSlotPartitioning(CustomTestCase):
    """HostKVCache.clear() partitions free_slots by dp_rank."""

    def test_dp_size_1_owns_all_local_slots(self):
        """With dp_size=1, free_slots covers the full local pool from slot 0."""
        c = _make_partitioned_cache(dp_rank=0, dp_size=1, page_size=4)
        self.assertEqual(c.slot_start, 0)
        self.assertEqual(c.local_size, c.size)
        self.assertEqual(len(c.free_slots), c.size)
        self.assertEqual(c.free_slots[0].item(), 0)

    def test_dp_size_2_partitions_are_disjoint(self):
        """Rank 0 and rank 1 receive non-overlapping slot sets.

        Guards against an off-by-one or sign error in slot_start / local_size;
        overlapping partitions would cause double-alloc corruption.
        """
        c0 = _make_partitioned_cache(dp_rank=0, dp_size=2, page_size=4)
        c1 = _make_partitioned_cache(dp_rank=1, dp_size=2, page_size=4)
        overlap = set(c0.free_slots.tolist()) & set(c1.free_slots.tolist())
        self.assertEqual(len(overlap), 0, f"Ranks 0 and 1 share slots: {overlap}")

    def test_dp_size_2_boundary_is_contiguous(self):
        """Rank 1's first slot immediately follows rank 0's last slot."""
        c0 = _make_partitioned_cache(dp_rank=0, dp_size=2, page_size=4)
        c1 = _make_partitioned_cache(dp_rank=1, dp_size=2, page_size=4)
        self.assertEqual(
            c1.slot_start,
            c0.slot_start + c0.local_size,
            "Rank 1's window does not start where rank 0's ends",
        )

    def test_dp_size_2_equal_and_page_aligned_local_sizes(self):
        """Both ranks receive the same local_size, which is a multiple of page_size."""
        c0 = _make_partitioned_cache(dp_rank=0, dp_size=2, page_size=4)
        c1 = _make_partitioned_cache(dp_rank=1, dp_size=2, page_size=4)
        self.assertEqual(c0.local_size, c1.local_size)
        self.assertEqual(c0.local_size % c0.page_size, 0)

    def test_slot_used_spans_full_pool_for_cross_dp_double_free_detection(self):
        """slot_used covers self.size (not just local_size).

        A peer's double-free (freeing a slot it does not own) must be caught;
        shrinking slot_used to local_size would miss those bugs.
        """
        c0 = _make_partitioned_cache(dp_rank=0, dp_size=2, page_size=4)
        self.assertEqual(len(c0.slot_used), c0.size)

    def test_alloc_returns_slots_within_own_partition(self):
        """alloc() only yields slot indices in [slot_start, slot_start+local_size)."""
        c0 = _make_partitioned_cache(dp_rank=0, dp_size=2, page_size=4)
        slots = c0.alloc(c0.page_size)
        self.assertIsNotNone(slots)
        for s in slots.tolist():
            self.assertGreaterEqual(s, c0.slot_start)
            self.assertLess(s, c0.slot_start + c0.local_size)

    def test_available_size_equals_local_size(self):
        """available_size() reflects only this rank's capacity, not the full pool."""
        c0 = _make_partitioned_cache(dp_rank=0, dp_size=2, page_size=4)
        self.assertEqual(c0.available_size(), c0.local_size)


# ============================================================================
# Test 4 — HiCacheShmPeerL2 index operations
# ============================================================================

class TestHiCacheShmPeerL2(CustomTestCase):
    """Index-SHM CRUD: set_v1 / exists / get_v1 / clear."""

    _PAGE_SIZE = 2
    _N_TOKENS = 32
    _FEAT = 8

    def setUp(self):
        self._tag = f"ut_pl2_{uuid.uuid4().hex[:8]}"
        self._cfg = _make_storage_config(f"model_{self._tag}")
        self._buf = torch.zeros(self._N_TOKENS, self._FEAT, dtype=torch.float32)

    def tearDown(self):
        d = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
        _cleanup_shm(d, f"sglang_l2_idx*{self._tag}*")

    def _make_backends(self):
        b0 = HiCacheShmPeerL2(self._cfg)
        b0.register_mem_pool_host(_FlatBufPool(self._buf, dp_rank=0, page_size=self._PAGE_SIZE))
        b1 = HiCacheShmPeerL2(self._cfg)
        b1.register_mem_pool_host(_FlatBufPool(self._buf, dp_rank=1, page_size=self._PAGE_SIZE))
        return b0, b1

    def _host_idx(self, *slot_pairs) -> torch.Tensor:
        """Build host_indices: each slot_pair (start, start) repeated page_size times."""
        indices = []
        for s in slot_pairs:
            indices.extend([s] * self._PAGE_SIZE)
        return torch.tensor(indices, dtype=torch.int64)

    def test_capacity_auto_computed_from_page_num(self):
        """Without peer_l2_capacity in extra_config, capacity = page_num × 4.

        Guards the auto-compute path: if register_mem_pool_host stops setting
        _capacity from page_num, the default of 4096 silently returns, causing
        near-100% collision rate on any real L2 pool.
        """
        b = HiCacheShmPeerL2(self._cfg)
        pool = _FlatBufPool(self._buf, dp_rank=0, page_size=self._PAGE_SIZE)
        b.register_mem_pool_host(pool)
        self.assertEqual(b._capacity, pool.page_num * 4)
        self.assertFalse(b._capacity_user_set)

    def test_capacity_user_value_overrides_auto(self):
        """peer_l2_capacity in extra_config must pin _capacity and set _capacity_user_set.

        Guards the override path: a regression that ignores extra_config would
        silently replace the user value with the auto-computed one.
        """
        import dataclasses

        user_capacity = 512
        cfg = dataclasses.replace(
            _make_storage_config(f"model_{self._tag}_override"),
            extra_config={"peer_l2_capacity": user_capacity},
        )
        b = HiCacheShmPeerL2(cfg)
        pool = _FlatBufPool(self._buf, dp_rank=0, page_size=self._PAGE_SIZE)
        b.register_mem_pool_host(pool)
        self.assertEqual(b._capacity, user_capacity)
        self.assertTrue(b._capacity_user_set)

    def test_index_shm_is_created_after_registration(self):
        b0, _ = self._make_backends()
        self.assertIsNotNone(b0._mm)

    def test_batch_set_v1_returns_all_true(self):
        b0, _ = self._make_backends()
        r = b0.batch_set_v1(["key_A", "key_B"], self._host_idx(0, 2))
        self.assertTrue(all(r))

    def test_exists_false_for_own_entries(self):
        """exists() must not match entries written by the same dp_rank.

        Guards a bug where a rank might read its own in-flight pages back
        through the peer-L2 path, causing double-load.
        """
        b0, _ = self._make_backends()
        b0.batch_set_v1(["key_A"], self._host_idx(0))
        self.assertFalse(b0.exists("key_A"))

    def test_exists_true_for_peer_entries(self):
        b0, b1 = self._make_backends()
        b0.batch_set_v1(["key_A", "key_B"], self._host_idx(0, 2))
        self.assertTrue(b1.exists("key_A"))
        self.assertTrue(b1.exists("key_B"))

    def test_exists_false_for_unknown_key(self):
        _, b1 = self._make_backends()
        self.assertFalse(b1.exists("no_such_key"))

    def test_batch_get_v1_copies_peer_page_into_local_slot(self):
        """batch_get_v1 copies the peer's page into our target slot.

        The guard: if get_data_page / set_from_flat_data_page are swapped or
        the slot index is wrong, the target slot will not match the source.
        """
        b0, b1 = self._make_backends()
        self._buf[0:2, :] = 42.0   # DP0's page A
        self._buf[2:4, :] = 99.0   # DP0's page B
        b0.batch_set_v1(["key_A", "key_B"], self._host_idx(0, 2))

        target = self._host_idx(10, 12)
        results = b1.batch_get_v1(["key_A", "key_B"], target)
        self.assertTrue(all(results))
        self.assertAlmostEqual(self._buf[10:12, :].mean().item(), 42.0, places=3)
        self.assertAlmostEqual(self._buf[12:14, :].mean().item(), 99.0, places=3)

    def test_batch_get_v1_returns_false_for_own_or_missing_key(self):
        b0, b1 = self._make_backends()
        # DP1 writes its own key — should not be returned by DP1's get
        b1.batch_set_v1(["own_key"], self._host_idx(0))
        target = self._host_idx(10)
        self.assertFalse(b1.batch_get_v1(["own_key"], target)[0])
        self.assertFalse(b1.batch_get_v1(["missing"], target)[0])

    def test_batch_exists_v2_reports_correct_consecutive_prefix(self):
        b0, b1 = self._make_backends()
        b0.batch_set_v1(["k0", "k1", "k2"], self._host_idx(0, 2, 4))

        full = b1.batch_exists_v2(["k0", "k1", "k2"])
        self.assertEqual(full.kv_hit_pages, 3)

        partial = b1.batch_exists_v2(["k0", "missing", "k2"])
        self.assertEqual(partial.kv_hit_pages, 1)

    def test_clear_invalidates_all_entries_for_peer(self):
        """After clear(), the peer no longer sees any entries."""
        b0, b1 = self._make_backends()
        b0.batch_set_v1(["key_A"], self._host_idx(0))
        self.assertTrue(b1.exists("key_A"))
        b0.clear()
        self.assertFalse(b1.exists("key_A"))

    def test_second_attach_sees_existing_index(self):
        """A second HiCacheShmPeerL2 attaches to the existing index, not a new one."""
        b0, b1 = self._make_backends()
        b0.batch_set_v1(["key_A"], self._host_idx(0))

        # Instantiate a fresh b1' — must see key_A written by b0
        b1_fresh = HiCacheShmPeerL2(self._cfg)
        b1_fresh.register_mem_pool_host(
            _FlatBufPool(self._buf, dp_rank=1, page_size=self._PAGE_SIZE)
        )
        self.assertTrue(b1_fresh.exists("key_A"))


# ============================================================================
# Test 5 — Cross-process shared L2
# ============================================================================

# Inline scripts run in subprocesses. They import sglang independently so they
# exercise the actual OS-level MAP_SHARED behaviour rather than in-process mocks.

_WRITER_SCRIPT = textwrap.dedent(
    """
import sys, os
sys.path.insert(0, {sglang_path!r})
os.environ.setdefault("SGLANG_SUPPRESS_WARNINGS", "1")

from sglang.srt.mem_cache.storage.mmap import alloc_shm
from sglang.srt.mem_cache.storage.shm.hicache_shm_peer_l2 import HiCacheShmPeerL2
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
import torch, json

SHM_NAME   = {shm_name!r}
MODEL_TAG  = {model_tag!r}
PAGE_SIZE  = 2
N          = 24
FEAT       = 8

tensor, fd, mm = alloc_shm((N, FEAT), torch.float32, shm_name=SHM_NAME)
tensor[0:PAGE_SIZE, :]            = 42.0   # page A at slot 0
tensor[PAGE_SIZE:PAGE_SIZE*2, :] = 99.0   # page B at slot PAGE_SIZE

cfg = HiCacheStorageConfig(
    tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
    attn_cp_rank=0, attn_cp_size=1,
    is_mla_model=False, enable_storage_metrics=False,
    is_page_first_layout=True, model_name=MODEL_TAG,
)

class Pool:
    page_size = PAGE_SIZE
    page_num  = N // PAGE_SIZE
    dp_rank   = 0
    dp_size   = 2
    def get_data_page(self, idx, flat=True):
        d = tensor[idx:idx+PAGE_SIZE, :]
        return d.flatten() if flat else d
    def set_from_flat_data_page(self, idx, p):
        tensor[idx:idx+PAGE_SIZE, :] = p.reshape(PAGE_SIZE, FEAT)

b = HiCacheShmPeerL2(cfg)
b.register_mem_pool_host(Pool())
ok = b.batch_set_v1(
    ["key_A", "key_B"],
    torch.tensor([0, 0, PAGE_SIZE, PAGE_SIZE], dtype=torch.int64),
)
print(json.dumps({{"batch_set_ok": all(ok)}}))
os.close(fd)
"""
)

_READER_SCRIPT = textwrap.dedent(
    """
import sys, os
sys.path.insert(0, {sglang_path!r})
os.environ.setdefault("SGLANG_SUPPRESS_WARNINGS", "1")

from sglang.srt.mem_cache.storage.mmap import alloc_shm
from sglang.srt.mem_cache.storage.shm.hicache_shm_peer_l2 import HiCacheShmPeerL2
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
import torch, json

SHM_NAME  = {shm_name!r}
MODEL_TAG = {model_tag!r}
PAGE_SIZE = 2
N         = 24
FEAT      = 8

tensor, fd, mm = alloc_shm((N, FEAT), torch.float32, shm_name=SHM_NAME)

cfg = HiCacheStorageConfig(
    tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
    attn_cp_rank=0, attn_cp_size=1,
    is_mla_model=False, enable_storage_metrics=False,
    is_page_first_layout=True, model_name=MODEL_TAG,
)

class Pool:
    page_size = PAGE_SIZE
    page_num  = N // PAGE_SIZE
    dp_rank   = 1
    dp_size   = 2
    def get_data_page(self, idx, flat=True):
        d = tensor[idx:idx+PAGE_SIZE, :]
        return d.flatten() if flat else d
    def set_from_flat_data_page(self, idx, p):
        tensor[idx:idx+PAGE_SIZE, :] = p.reshape(PAGE_SIZE, FEAT)

b = HiCacheShmPeerL2(cfg)
b.register_mem_pool_host(Pool())

ex_A    = b.exists("key_A")
ex_B    = b.exists("key_B")
ex_none = b.exists("no_such_key")

# Read peer's pages into our slots 10 and 12
target = torch.tensor([10, 10, 10 + PAGE_SIZE, 10 + PAGE_SIZE], dtype=torch.int64)
get_ok = b.batch_get_v1(["key_A", "key_B"], target)

val_A = float(tensor[10 : 10+PAGE_SIZE, :].mean())
val_B = float(tensor[10+PAGE_SIZE : 10+PAGE_SIZE*2, :].mean())

print(json.dumps({{
    "exists_A":    ex_A,
    "exists_B":    ex_B,
    "exists_none": ex_none,
    "get_ok":      get_ok,
    "val_A":       val_A,
    "val_B":       val_B,
}}))
os.close(fd)
"""
)


class TestCrossProcessSharedL2(CustomTestCase):
    """Two separate processes share one named SHM; peer index lookup bridges them."""

    def setUp(self):
        self._shm_name = f"ut_xp_{uuid.uuid4().hex[:8]}"
        self._model_tag = f"xp_{self._shm_name}"

    def tearDown(self):
        d = _named_shm_dir()
        _cleanup_shm(
            d,
            f"{self._shm_name}.mmap",
            f"{self._shm_name}.mmap.lock",
            f"sglang_l2_idx*{self._shm_name}*",
        )

    def _run_script(self, template: str, timeout: int = 30) -> dict:
        code = template.format(
            sglang_path=_SGLANG_PYTHON,
            shm_name=self._shm_name,
            model_tag=self._model_tag,
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            self.fail(
                f"Subprocess failed (rc={result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        # Last non-empty line must be valid JSON
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        self.assertTrue(lines, f"No output from script:\nstderr: {result.stderr}")
        return json.loads(lines[-1])

    def test_writer_entries_visible_to_reader_via_peer_index(self):
        """DP0 writes pages → index SHM → DP1 reads correct values.

        Guards:
        - MAP_SHARED makes pages persistent between two process lifetimes
        - fcntl serialisation allows attach without data race
        - batch_get_v1 reads the correct slot (not index-slot confusion)
        """
        w = self._run_script(_WRITER_SCRIPT)
        self.assertTrue(
            w.get("batch_set_ok"),
            f"Writer batch_set_v1 reported failure: {w}",
        )

        r = self._run_script(_READER_SCRIPT)
        self.assertTrue(r.get("exists_A"), "DP1 could not find DP0's key_A in index")
        self.assertTrue(r.get("exists_B"), "DP1 could not find DP0's key_B in index")
        self.assertFalse(r.get("exists_none"), "exists() returned True for unknown key")
        self.assertTrue(
            all(r.get("get_ok", [])),
            f"batch_get_v1 reported failure: {r.get('get_ok')}",
        )
        self.assertAlmostEqual(
            r["val_A"], 42.0, places=2,
            msg=f"DP1 got wrong value for page_A: {r['val_A']} (expected 42.0)",
        )
        self.assertAlmostEqual(
            r["val_B"], 99.0, places=2,
            msg=f"DP1 got wrong value for page_B: {r['val_B']} (expected 99.0)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
