"""Nested progress helpers + verbose step logging for AXION runs.

Outer bars: dataset x setting (and seeds / variants).
Inner bars: epochs / MCS score chunks / extension steps.

Uses tqdm when available; otherwise plain stderr lines.

Env:
  AXION_NO_PROGRESS=1      disable bars
  AXION_FORCE_PROGRESS=1   force bars even when non-TTY (tee / nohup / final)
  AXION_VERBOSE=1          print every stage/step banner (default on for final)
  AXION_VERBOSE_EVERY=1    plain-mode: log every bar update (not just ~10%)
  AXION_EPOCH_BATCH_BAR=1  show batch i/N on epoch bar postfix
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None  # type: ignore


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def verbose_enabled() -> bool:
    if _truthy("AXION_NO_VERBOSE"):
        return False
    if _truthy("AXION_VERBOSE"):
        return True
    # Final / forced-progress runs always want stage banners
    return _truthy("AXION_FORCE_PROGRESS")


def progress_enabled() -> bool:
    if _truthy("AXION_NO_PROGRESS"):
        return False
    if _truthy("AXION_FORCE_PROGRESS"):
        return True
    if verbose_enabled():
        return True
    return bool(sys.stderr.isatty() or sys.stdout.isatty())


def step_log(msg: str, *, level: str = "STEP") -> None:
    """Always-flush stage banner when verbose (or forced progress)."""
    if not (verbose_enabled() or progress_enabled()):
        return
    ts = time.strftime("%H:%M:%S")
    print(f"[{level} {ts}] {msg}", file=sys.stderr, flush=True)


class ProgressBar:
    """Thin wrapper so call sites stay identical with/without tqdm."""

    def __init__(
        self,
        total: int,
        *,
        desc: str = "",
        leave: bool = True,
        unit: str = "it",
        position: Optional[int] = None,
        mininterval: Optional[float] = None,
    ) -> None:
        self.total = max(0, int(total))
        self.desc = str(desc)
        self.n = 0
        self._plain = _tqdm is None or not progress_enabled()
        self._bar = None
        self._every = _truthy("AXION_VERBOSE_EVERY") or verbose_enabled()
        if mininterval is None:
            mininterval = 0.05 if verbose_enabled() else 0.2
        if not self._plain:
            kwargs: dict[str, Any] = dict(
                total=self.total,
                desc=self.desc,
                leave=leave,
                unit=unit,
                mininterval=float(mininterval),
                dynamic_ncols=True,
                file=sys.stderr,
                smoothing=0.05,
            )
            if position is not None:
                kwargs["position"] = int(position)
            self._bar = _tqdm(**kwargs)
        elif progress_enabled() and self.desc:
            print(
                f"[progress] start {self.desc} total={self.total}",
                file=sys.stderr,
                flush=True,
            )

    def update(self, n: int = 1) -> None:
        self.n += int(n)
        if self._bar is not None:
            self._bar.update(int(n))
        elif progress_enabled() and self.total > 0:
            if self._every or self.n == self.total or self.n % max(1, self.total // 20) == 0:
                print(
                    f"[progress] {self.desc} {self.n}/{self.total}",
                    file=sys.stderr,
                    flush=True,
                )

    def set_description(self, desc: str) -> None:
        self.desc = str(desc)
        if self._bar is not None:
            self._bar.set_description(self.desc)

    def set_postfix(self, **kwargs: Any) -> None:
        if self._bar is not None:
            self._bar.set_postfix(**kwargs, refresh=False)
        elif progress_enabled() and kwargs and self._every:
            bits = " ".join(f"{k}={v}" for k, v in kwargs.items())
            print(
                f"[progress] {self.desc} {self.n}/{self.total} {bits}",
                file=sys.stderr,
                flush=True,
            )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        elif progress_enabled() and self.desc:
            print(
                f"[progress] done {self.desc} {self.n}/{self.total}",
                file=sys.stderr,
                flush=True,
            )

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@contextmanager
def progress(
    total: int,
    *,
    desc: str = "",
    leave: bool = True,
    unit: str = "it",
    position: Optional[int] = None,
    mininterval: Optional[float] = None,
) -> Iterator[ProgressBar]:
    bar = ProgressBar(
        total,
        desc=desc,
        leave=leave,
        unit=unit,
        position=position,
        mininterval=mininterval,
    )
    try:
        yield bar
    finally:
        bar.close()


@contextmanager
def stage(name: str, *, detail: str = "") -> Iterator[float]:
    """Time a named stage and emit start/done step_log lines."""
    msg = f"{name}" + (f" | {detail}" if detail else "")
    step_log(f"BEGIN {msg}")
    t0 = time.perf_counter()
    try:
        yield t0
    finally:
        dt = time.perf_counter() - t0
        step_log(f"END   {msg} ({dt:.2f}s)")


def env_cpu_threads_60pct(nproc: Optional[int] = None) -> int:
    """Return ``max(1, int(0.6 * nproc))`` for OMP/MKL/TORCH thread caps."""
    if nproc is None:
        nproc = int(os.cpu_count() or 1)
    return max(1, int(0.6 * int(nproc)))
