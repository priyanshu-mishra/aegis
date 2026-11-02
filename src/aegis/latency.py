"""Wall-clock timing for pipeline stages and per-call latency percentiles."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np


class StageTimer:
    """Accumulates durations (seconds) per named stage; ``stage`` can be entered repeatedly."""

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.samples[name].append(time.perf_counter() - t0)

    def record(self, name: str, seconds: float) -> None:
        self.samples[name].append(seconds)

    def summary(self) -> dict[str, dict[str, float]]:
        """Per stage: call count, total seconds, and p50/p95/max in milliseconds."""
        out = {}
        for name, xs in self.samples.items():
            a = np.asarray(xs) * 1000
            out[name] = {
                "calls": len(xs),
                "total_s": float(a.sum() / 1000),
                "p50_ms": float(np.percentile(a, 50)),
                "p95_ms": float(np.percentile(a, 95)),
                "max_ms": float(a.max()),
            }
        return out


def time_calls(fn: Callable[[int], object], n: int, timer: StageTimer, name: str) -> None:
    """Call ``fn(i)`` for i in range(n), recording each call's latency under ``name``."""
    for i in range(n):
        t0 = time.perf_counter()
        fn(i)
        timer.record(name, time.perf_counter() - t0)
