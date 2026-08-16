"""Runtime hardware profiling for AXION throughput."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class HardwareProfile:
    cpu_count: int
    device: str
    vram_mb: float
    score_batch_size: int
    dataloader_num_workers: int
    torch_num_threads: int


_CONFIGURED = False


def cpu_count() -> int:
    return int(os.cpu_count() or 1)


def vram_mb(device: Optional[torch.device] = None) -> float:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    try:
        free_b, total_b = torch.cuda.mem_get_info(device)
        # Prefer free for batch sizing; fall back to total
        return float(free_b) / (1024.0 * 1024.0)
    except Exception:
        try:
            props = torch.cuda.get_device_properties(device)
            return float(props.total_memory) / (1024.0 * 1024.0)
        except Exception:
            return 0.0


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def configure_torch_runtime(*, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Idempotent: threads, cudnn.benchmark, TF32.

    Honors OMP_NUM_THREADS / TORCH_NUM_THREADS when set by run_final.sh so Mode B
    max-HW (nproc-2) is not silently overridden.
    """
    global _CONFIGURED
    n = cpu_count()
    env_threads = _env_int("TORCH_NUM_THREADS") or _env_int("OMP_NUM_THREADS")
    if env_threads is not None and env_threads > 0:
        compute_threads = max(1, min(n, env_threads))
        workers_budget = max(0, n - compute_threads)
    else:
        # Leave room for DataLoader workers on multi-core boxes
        workers_budget = min(8, max(0, n // 3))
        compute_threads = max(1, n - workers_budget)
    torch.set_num_threads(compute_threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, compute_threads // 2)))
    except RuntimeError:
        pass  # already set
    info: Dict[str, Any] = {
        "cpu_count": n,
        "torch_num_threads": compute_threads,
        "dataloader_workers_budget": workers_budget,
    }
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_available():
        # g5-faithful: leave cudnn/TF32 at PyTorch defaults (benchmark/TF32
        # throughput toggles shifted classical tiny-n ranking on glass).
        torch.backends.cudnn.benchmark = False
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        info["cudnn_benchmark"] = False
        info["tf32"] = False
        info["vram_mb"] = vram_mb(device)
    _CONFIGURED = True
    return info


def recommend_score_batch_size(
    d: int,
    *,
    device: Optional[torch.device] = None,
    override: int = 0,
    vram: Optional[float] = None,
) -> int:
    """Pick a large scoring batch from VRAM and feature dim."""
    if int(override) > 0:
        return int(override)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        # CPU: still use large batches for fewer Python trips
        return int(min(8192, max(1024, 65536 // max(1, d))))
    mb = float(vram if vram is not None else vram_mb(device))
    # Tiers: 24GB+ / 16GB-class (~12-16k free) / 12GB / smaller
    # Heuristic: classical low-d can push huge batches; high-d PCA needs caution
    if d <= 16:
        if mb >= 18000:
            base = 32768
        elif mb >= 10000:
            base = 16384
        elif mb >= 6000:
            base = 8192
        else:
            base = 4096
    elif d <= 64:
        if mb >= 18000:
            base = 16384
        elif mb >= 10000:
            base = 12288
        elif mb >= 6000:
            base = 8192
        else:
            base = 4096
    elif d <= 256:
        if mb >= 18000:
            base = 8192
        elif mb >= 10000:
            base = 4096
        elif mb >= 6000:
            base = 2048
        else:
            base = 1024
    else:
        if mb >= 18000:
            base = 4096
        elif mb >= 10000:
            base = 2048
        elif mb >= 6000:
            base = 1024
        else:
            base = 512
    return int(max(512, base))


def recommend_dataloader_workers(override: int = -1) -> int:
    if int(override) >= 0:
        return int(override)
    n = cpu_count()
    # Cap at 8: host RAM on rented 16GB-GPU boxes is often ~21 GiB
    return int(min(8, max(0, n // 3)))


def build_profile(
    d: int,
    *,
    device: Optional[torch.device] = None,
    score_batch_size: int = 0,
    dataloader_num_workers: int = -1,
) -> HardwareProfile:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = configure_torch_runtime(device=device)
    workers = recommend_dataloader_workers(dataloader_num_workers)
    # Cap workers so compute threads stay meaningful
    n = cpu_count()
    workers = min(workers, max(0, n // 2))
    sbs = recommend_score_batch_size(
        d, device=device, override=score_batch_size, vram=info.get("vram_mb")
    )
    return HardwareProfile(
        cpu_count=n,
        device=str(device),
        vram_mb=float(info.get("vram_mb", 0.0)),
        score_batch_size=sbs,
        dataloader_num_workers=workers,
        torch_num_threads=int(info.get("torch_num_threads", max(1, n - workers))),
    )
