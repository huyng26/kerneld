from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

from kerneld.ops.rmsnorm.reference import rmsnorm_ref
from kerneld.ops.rmsnorm.verify import _resolve_device, _resolve_dtype, _tolerances
from kerneld.schemas import MicrobenchResult, RMSNormOpSpec


def benchmark_candidate_fn(
    *,
    candidate_id: str,
    kernel_fn: Callable[..., Any],
    spec: RMSNormOpSpec,
    warmup_iters: int = 20,
    measured_iters: int = 100,
) -> MicrobenchResult:
    try:
        if warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative")
        if measured_iters <= 0:
            raise ValueError("measured_iters must be positive")
        torch = _import_torch()
        device = _resolve_device(torch, spec.device)
        dtype = _resolve_dtype(torch, spec.dtype)
        x = torch.randn(spec.input_shape, device=device, dtype=dtype)
        weight = torch.randn((spec.hidden_size,), device=device, dtype=dtype)
        atol, rtol = _tolerances(spec.dtype)

        with torch.no_grad():
            expected = rmsnorm_ref(x, weight, spec.eps)
            actual = kernel_fn(x, weight, spec.eps)
        _synchronize(torch, device)
        if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
            return MicrobenchResult(
                candidate_id=candidate_id,
                passed=False,
                warmup_iters=warmup_iters,
                measured_iters=measured_iters,
                error="candidate failed allclose before benchmarking",
            )

        baseline_stats = _time_callable(
            torch,
            device,
            lambda: rmsnorm_ref(x, weight, spec.eps),
            warmup_iters=warmup_iters,
            measured_iters=measured_iters,
        )
        candidate_stats = _time_callable(
            torch,
            device,
            lambda: kernel_fn(x, weight, spec.eps),
            warmup_iters=warmup_iters,
            measured_iters=measured_iters,
        )
        baseline_ms = baseline_stats.median_ms
        candidate_ms = candidate_stats.median_ms
        speedup_pct = ((baseline_ms - candidate_ms) / baseline_ms) * 100.0 if baseline_ms > 0 else None
        return MicrobenchResult(
            candidate_id=candidate_id,
            passed=True,
            baseline_ms=baseline_ms,
            candidate_ms=candidate_ms,
            baseline_mean_ms=baseline_stats.mean_ms,
            candidate_mean_ms=candidate_stats.mean_ms,
            speedup_pct=speedup_pct,
            warmup_iters=warmup_iters,
            measured_iters=measured_iters,
            metadata={
                "shape": list(spec.input_shape),
                "dtype": spec.dtype,
                "device": str(device),
                "baseline": "rmsnorm_ref",
                "latency_ms": "median",
            },
        )
    except Exception as exc:
        return MicrobenchResult(
            candidate_id=candidate_id,
            passed=False,
            warmup_iters=warmup_iters,
            measured_iters=measured_iters,
            error=str(exc),
        )


@dataclass(frozen=True)
class _TimingStats:
    median_ms: float
    mean_ms: float
    samples_ms: list[float]


def _time_callable(
    torch: Any,
    device: Any,
    fn: Callable[[], Any],
    *,
    warmup_iters: int,
    measured_iters: int,
) -> _TimingStats:
    with torch.no_grad():
        for _ in range(warmup_iters):
            fn()
        _synchronize(torch, device)

        if str(device).startswith("cuda"):
            timings = []
            for _ in range(measured_iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                timings.append(float(start.elapsed_time(end)))
            _synchronize(torch, device)
            return _TimingStats(
                median_ms=float(statistics.median(timings)),
                mean_ms=float(statistics.fmean(timings)),
                samples_ms=timings,
            )

        timings = []
        for _ in range(measured_iters):
            start = time.perf_counter()
            fn()
            timings.append((time.perf_counter() - start) * 1000.0)
        return _TimingStats(
            median_ms=float(statistics.median(timings)),
            mean_ms=float(statistics.fmean(timings)),
            samples_ms=timings,
        )


def _synchronize(torch: Any, device: Any) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for RMSNorm benchmarking") from exc
    return torch
