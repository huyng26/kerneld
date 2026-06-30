from __future__ import annotations

from typing import Any, Callable

from kerneld.ops.rmsnorm.reference import rmsnorm_ref
from kerneld.schemas import RMSNormOpSpec, VerificationResult

_REL_ERROR_DENOM_MIN = 1e-3


def verify_candidate_fn(
    *,
    candidate_id: str,
    kernel_fn: Callable[..., Any],
    spec: RMSNormOpSpec,
) -> VerificationResult:
    try:
        torch = _import_torch()
        device = _resolve_device(torch, spec.device)
        dtype = _resolve_dtype(torch, spec.dtype)
        atol, rtol = _tolerances(spec.dtype)
        cases = []
        max_abs = 0.0
        max_rel = 0.0

        for shape in _verification_shapes(spec.input_shape, spec.hidden_size):
            x = torch.randn(shape, device=device, dtype=dtype)
            weight = torch.randn((spec.hidden_size,), device=device, dtype=dtype)
            expected = rmsnorm_ref(x, weight, spec.eps)
            actual = kernel_fn(x, weight, spec.eps)
            if actual.shape != expected.shape:
                raise AssertionError(f"shape mismatch: expected {tuple(expected.shape)}, got {tuple(actual.shape)}")
            if actual.dtype != expected.dtype:
                raise AssertionError(f"dtype mismatch: expected {expected.dtype}, got {actual.dtype}")
            torch.cuda.synchronize() if str(device).startswith("cuda") else None
            abs_error = (actual - expected).abs().max().item()
            rel_error = _stable_relative_error(torch, actual, expected)
            passed = bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
            cases.append(
                {
                    "shape": list(shape),
                    "dtype": spec.dtype,
                    "passed": passed,
                    "max_abs_error": abs_error,
                    "max_rel_error": rel_error,
                }
            )
            max_abs = max(max_abs, float(abs_error))
            max_rel = max(max_rel, float(rel_error))
            if not passed:
                return VerificationResult(
                    candidate_id=candidate_id,
                    passed=False,
                    max_abs_error=max_abs,
                    max_rel_error=max_rel,
                    atol=atol,
                    rtol=rtol,
                    cases=cases,
                    error=f"candidate output failed allclose for shape {shape}",
                )
        return VerificationResult(
            candidate_id=candidate_id,
            passed=True,
            max_abs_error=max_abs,
            max_rel_error=max_rel,
            atol=atol,
            rtol=rtol,
            cases=cases,
        )
    except Exception as exc:
        return VerificationResult(candidate_id=candidate_id, passed=False, error=str(exc))


def _stable_relative_error(torch: Any, actual: Any, expected: Any) -> float:
    denom = torch.maximum(actual.abs(), expected.abs()).clamp_min(_REL_ERROR_DENOM_MIN)
    return float(((actual - expected).abs() / denom).max().item())


def _verification_shapes(input_shape: tuple[int, ...], hidden_size: int) -> list[tuple[int, ...]]:
    primary = tuple(input_shape)
    shapes = [primary]
    if len(primary) >= 3:
        shapes.append((1, min(primary[-2], 8), hidden_size))
    elif len(primary) == 2:
        shapes.append((min(primary[0], 8), hidden_size))
    else:
        shapes.append((hidden_size,))
    deduped = []
    for shape in shapes:
        if shape[-1] != hidden_size:
            shape = (*shape[:-1], hidden_size)
        if shape not in deduped:
            deduped.append(shape)
    return deduped


def _tolerances(dtype: str) -> tuple[float, float]:
    normalized = dtype.lower()
    if normalized in {"float16", "fp16", "half"}:
        return 1e-2, 1e-2
    if normalized in {"bfloat16", "bf16"}:
        return 2e-2, 2e-2
    return 1e-5, 1e-5


def _resolve_dtype(torch: Any, dtype: str):
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype for verification: {dtype!r}") from exc


def _resolve_device(torch: Any, requested: str):
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA verification device, but torch.cuda.is_available() is false")
    return torch.device(requested)


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for RMSNorm verification") from exc
    return torch
