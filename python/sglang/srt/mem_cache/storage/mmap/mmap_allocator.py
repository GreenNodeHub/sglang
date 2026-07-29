import ctypes
import ctypes.util
import fcntl
import logging
import math
import mmap
import os
import uuid
import weakref
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Load libc once at module level so munmap is callable safely at GC/shutdown time.
# Resolve the SONAME via find_library so the allocator also works on systems
# whose libc is not named "libc.so.6" (e.g. musl / Alpine).
try:
    _libc_name = ctypes.util.find_library("c") or "libc.so.6"
    _libc = ctypes.CDLL(_libc_name, use_errno=True)
    _libc.mmap.restype = ctypes.c_void_p
    _libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    _libc.munmap.restype = ctypes.c_int
    _libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
except OSError:
    _libc = None

# MAP_POPULATE is in Python's mmap module only since 3.11.
# Fall back to 0 (not the Linux constant 0x08000) so macOS and other
# non-Linux platforms don't accidentally use MAP_NOCACHE (same bits).
_MAP_POPULATE = getattr(mmap, "MAP_POPULATE", 0)
# MAP_HUGETLB and MAP_HUGE_* are Linux-specific and not in Python's mmap module.
_MAP_HUGETLB = 0x40000
_MAP_HUGE_2MB = 21 << 26  # 0x1400000
_MAP_HUGE_1GB = 30 << 26  # 0x78000000
_MAP_FAILED = ctypes.c_void_p(-1).value
_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)


def _alloc_hugepage(n_bytes: int, alloc_bytes: int, extra_flags: int) -> ctypes.Array:
    """Call mmap via libc with hugepage flags and return an owning ctypes array.

    munmap fires automatically via weakref.finalize when the array is
    garbage-collected (i.e. when the tensor that wraps it is freed).
    """
    ptr = _libc.mmap(
        None,
        alloc_bytes,
        mmap.PROT_READ | mmap.PROT_WRITE,
        mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | _MAP_POPULATE | extra_flags,
        -1,
        0,
    )
    if ptr is None or ptr == _MAP_FAILED:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    array = (ctypes.c_uint8 * n_bytes).from_address(ptr)
    weakref.finalize(array, _libc.munmap, ctypes.c_void_p(ptr), alloc_bytes)
    return array


def alloc_mmap(dims: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Allocate a host tensor via anonymous mmap. Set SGLANG_HUGEPAGE_SIZE=2MB or 1GB for hugepages.

    MAP_SHARED + MAP_POPULATE are both required so cudaHostRegister pins real,
    pre-faulted physical pages (otherwise pinning can race with COW or page
    faults and the device ends up reading stale data).

    The tensor owns the mapping; munmap fires when the tensor is freed.
    """
    # Re-read per call (not cached) so that envs.SGLANG_HUGEPAGE_SIZE.override()
    # works correctly in tests.
    hugepage_size = (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip().upper()
    n_bytes = math.prod(dims) * torch.empty([], dtype=dtype).element_size()

    if hugepage_size == "":
        page_size, extra_flags = mmap.PAGESIZE, 0
    elif hugepage_size == "2MB":
        page_size, extra_flags = 2 * 1024 * 1024, _MAP_HUGETLB | _MAP_HUGE_2MB
    elif hugepage_size == "1GB":
        page_size, extra_flags = 1024 * 1024 * 1024, _MAP_HUGETLB | _MAP_HUGE_1GB
    else:
        logger.warning(
            "Unrecognized SGLANG_HUGEPAGE_SIZE=%r; expected '2MB' or '1GB'. "
            "Falling back to plain page-size mmap.",
            envs.SGLANG_HUGEPAGE_SIZE.get(),
        )
        page_size, extra_flags = mmap.PAGESIZE, 0

    alloc_bytes = math.ceil(n_bytes / page_size) * page_size

    if extra_flags:
        if _libc is None:
            logger.error(
                "Hugepage mmap requested but libc.so.6 could not be loaded; "
                "falling back to plain mmap. SGLANG_HUGEPAGE_SIZE=%s will be ignored.",
                hugepage_size,
            )
        else:
            try:
                array = _alloc_hugepage(n_bytes, alloc_bytes, extra_flags)
                return torch.frombuffer(
                    array, dtype=dtype, count=math.prod(dims)
                ).reshape(dims)
            except OSError as e:
                logger.error(
                    "Hugepage mmap via libc failed (%s); falling back to plain mmap. "
                    "SGLANG_HUGEPAGE_SIZE=%s will be ignored.",
                    e,
                    hugepage_size,
                )
        alloc_bytes = math.ceil(n_bytes / mmap.PAGESIZE) * mmap.PAGESIZE

    # Plain mmap path -- used directly when no hugepages requested, or as fallback.
    # torch.frombuffer keeps a reference to mm inside the tensor storage, so mm
    # stays alive until the tensor is freed and mmap.mmap.__del__ calls munmap.
    mm = mmap.mmap(
        -1,
        alloc_bytes,
        flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | _MAP_POPULATE,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    try:
        # MADV_POPULATE_WRITE guarantees pages are populated and writable,
        # throwing an error on failure (e.g. out of memory).
        mm.madvise(_MADV_POPULATE_WRITE)
    except OSError:
        # Fall back to MAP_POPULATE if MADV_POPULATE_WRITE is not supported (<5.14 kernel).
        pass
    return torch.frombuffer(mm, dtype=dtype, count=math.prod(dims)).reshape(dims)


def _named_shm_dir() -> str:
    """Return a directory suitable for named shared-memory files."""
    if os.path.isdir("/dev/shm"):
        return "/dev/shm"
    import tempfile

    return tempfile.gettempdir()


def _open_anonymous_shm_fd(alloc_bytes: int) -> int:
    """Open a private anonymous SHM fd and truncate it to alloc_bytes.

    Uses memfd_create when available (Linux 3.17+); falls back to an
    immediately-unlinked file under /dev/shm.  The resulting fd is not
    visible to other processes.
    """
    fd = None
    try:
        # MFD_CLOEXEC is standard on Linux 3.17+
        fd = os.memfd_create(
            f"sglang_host_pool_{uuid.uuid4().hex}",
            flags=getattr(os, "MFD_CLOEXEC", 1),
        )
    except (AttributeError, OSError):
        # Fallback to creating a file in /dev/shm if memfd_create is not supported
        shm_path = f"/dev/shm/sglang_host_pool_{uuid.uuid4().hex}.mmap"
        try:
            fd = os.open(shm_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
            try:
                os.unlink(shm_path)
            except OSError:
                pass
        except Exception as e:
            raise OSError(f"Failed to create shm file: {e}")
    os.ftruncate(fd, alloc_bytes)
    return fd


def _open_named_shm_fd(alloc_bytes: int, shm_name: str) -> int:
    """Open or create a named SHM file under /dev/shm, serialised with flock.

    The first caller creates the file and truncates it to alloc_bytes; later
    callers simply open the existing file.  A stale file (wrong size) is
    removed and recreated.  The file is NOT unlinked on close — the caller
    owns lifecycle and cleanup.
    """
    shm_dir = _named_shm_dir()
    shm_path = os.path.join(shm_dir, f"{shm_name}.mmap")
    lock_path = shm_path + ".lock"

    # Serialise creation so only one process truncates.
    lock_fd = open(lock_path, "a+b")
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        is_new = not os.path.exists(shm_path)
        if not is_new and os.path.getsize(shm_path) != alloc_bytes:
            logger.warning(
                "alloc_shm: stale named SHM %s (size %d != expected %d), recreating.",
                shm_path,
                os.path.getsize(shm_path),
                alloc_bytes,
            )
            os.remove(shm_path)
            is_new = True

        fd = os.open(shm_path, os.O_CREAT | os.O_RDWR, 0o600)
        if is_new:
            os.ftruncate(fd, alloc_bytes)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    logger.info(
        "alloc_shm: %s named SHM %s (%.1f MB)",
        "created" if is_new else "attached to",
        shm_path,
        alloc_bytes / (1024**2),
    )
    return fd


def alloc_shm(
    dims: tuple,
    dtype: torch.dtype,
    shm_name: Optional[str] = None,
) -> tuple[torch.Tensor, int, mmap.mmap]:
    """Allocate a host tensor via shared memory.

    shm_name=None (default)
        Anonymous private mapping (memfd_create + random name, or an
        immediately-unlinked /dev/shm file).  Not shareable across
        unrelated processes.  Preserves the original behaviour.

    shm_name=<str>
        Named mapping at /dev/shm/<shm_name>.mmap.  All processes that
        call alloc_shm with the same shm_name and dims/dtype attach to
        the same physical pages.  The first caller creates and truncates
        the file; subsequent callers map the existing file.  The file is
        NOT unlinked on close — the caller owns cleanup.

    Returns (tensor, fd, mm).  The caller must keep fd open for the
    lifetime of the mapping.
    """
    if (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip() != "":
        logger.warning(
            "Hugepages are not supported with SHM allocator. "
            "Falling back to plain page-size mmap."
        )

    n_bytes = math.prod(dims) * torch.empty([], dtype=dtype).element_size()
    alloc_bytes = math.ceil(n_bytes / mmap.PAGESIZE) * mmap.PAGESIZE

    fd = (
        _open_anonymous_shm_fd(alloc_bytes)
        if shm_name is None
        else _open_named_shm_fd(alloc_bytes, shm_name)
    )

    try:
        mm = mmap.mmap(
            fd,
            alloc_bytes,
            flags=mmap.MAP_SHARED | _MAP_POPULATE,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        try:
            # MADV_POPULATE_WRITE guarantees pages are populated and writable,
            # throwing an error on failure (e.g. out of memory).
            mm.madvise(_MADV_POPULATE_WRITE)
        except OSError:
            # Fall back to MAP_POPULATE if MADV_POPULATE_WRITE is not supported (<5.14 kernel).
            pass
    except Exception as e:
        os.close(fd)
        raise e

    tensor = torch.frombuffer(mm, dtype=dtype, count=math.prod(dims)).reshape(dims)
    return tensor, fd, mm
