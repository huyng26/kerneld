from __future__ import annotations

import gc
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from kerneld.ops.registry import OpHandler, get_op_handler, validate_op_spec
from kerneld.pipeline.candidates import load_candidate_info
from kerneld.pipeline.integrator import patch_model
from kerneld.run_state import RunState
from kerneld.schemas import ModelbenchResult, OpSpec

_MODEL_OUTPUT_REL_DENOM_MIN = 1e-3
_DEFAULT_MODEL_INPUT_SEED = 0
_MODEL_OUTPUT_TOLERANCES = {
    "float32": (1e-4, 1e-5),
    "fp32": (1e-4, 1e-5),
    "float16": (5e-1, 5e-2),
    "fp16": (5e-1, 5e-2),
    "bfloat16": (5e-1, 5e-2),
    "bf16": (5e-1, 5e-2),
}


def modelbench_run(
    run_dir: Path,
    *,
    candidate_id: str,
    warmup_iters: int = 5,
    measured_iters: int = 20,
    input_seed: int = _DEFAULT_MODEL_INPUT_SEED,
) -> ModelbenchResult:
    state = RunState.load(run_dir)
    if state.config is None:
        result = ModelbenchResult(candidate_id=candidate_id, passed=False, error="missing config.json")
        state.write_json(f"modelbench/{candidate_id}.json", result)
        return result
    try:
        spec_payload = state.read_json("op_spec.json")
        handler, spec = validate_op_spec(spec_payload)
        candidate = load_candidate_info(state, candidate_id)
        result = _benchmark_model_pair(
            model_id=state.config.model_id,
            input_shape=state.config.input_shape,
            dtype_label=state.config.dtype,
            device_label=state.config.device,
            spec=spec,
            handler=handler,
            candidate=candidate,
            candidate_id=candidate_id,
            warmup_iters=warmup_iters,
            measured_iters=measured_iters,
            input_seed=input_seed,
        )
    except Exception as exc:
        result = ModelbenchResult(
            candidate_id=candidate_id,
            passed=False,
            warmup_iters=warmup_iters,
            measured_iters=measured_iters,
            metadata={"input_seed": input_seed},
            error=str(exc),
        )
    state.write_json(f"modelbench/{candidate_id}.json", result)
    return result


def _benchmark_model_pair(
    *,
    model_id: str,
    input_shape: tuple[int, ...],
    dtype_label: str,
    device_label: str,
    spec: OpSpec,
    handler: OpHandler | None = None,
    candidate=None,
    candidate_id: str,
    warmup_iters: int,
    measured_iters: int,
    input_seed: int = _DEFAULT_MODEL_INPUT_SEED,
) -> ModelbenchResult:
    if handler is None:
        handler = get_op_handler(spec.op_type)
    torch = _import_torch()
    transformers = _import_transformers()
    device = _resolve_device(torch, device_label)
    dtype = _resolve_dtype(torch, dtype_label)
    input_ids = None
    baseline_logits = None

    baseline_model = _load_model(transformers, model_id, dtype, device)
    input_ids = _make_input_ids(torch, baseline_model, input_shape, device, input_seed=input_seed)
    with torch.no_grad():
        baseline_logits = baseline_model(input_ids=input_ids).logits.detach().float().cpu()
    baseline_ms = _time_forward(
        torch,
        device,
        lambda model=baseline_model: model(input_ids=input_ids),
        warmup_iters=warmup_iters,
        measured_iters=measured_iters,
    )
    del baseline_model
    _cleanup(torch, device)

    patched_model = _load_model(transformers, model_id, dtype, device)
    patch_result = patch_model(patched_model, spec, candidate, scope="compatible")
    patch_metadata = _patch_metadata(handler, spec, patch_result)
    with torch.no_grad():
        patched_logits_tensor = patched_model(input_ids=input_ids).logits.detach().float().cpu()
    patched_ms = _time_forward(
        torch,
        device,
        lambda model=patched_model: model(input_ids=input_ids),
        warmup_iters=warmup_iters,
        measured_iters=measured_iters,
    )
    output_metrics = _model_output_metrics(torch, baseline_logits, patched_logits_tensor)
    max_abs_tol, mean_abs_tol = _model_output_tolerances(dtype_label)
    output_passed = bool(
        output_metrics["max_abs_error"] <= max_abs_tol
        and output_metrics["mean_abs_error"] <= mean_abs_tol
        and output_metrics["argmax_match"]
    )
    speedup_pct = ((baseline_ms - patched_ms) / baseline_ms) * 100.0 if baseline_ms > 0 else None
    del patched_model
    _cleanup(torch, device)

    error = None
    if not output_passed:
        error = (
            "model output drift exceeded tolerance: "
            f"max_abs={output_metrics['max_abs_error']:.6g} <= {max_abs_tol:.6g}, "
            f"mean_abs={output_metrics['mean_abs_error']:.6g} <= {mean_abs_tol:.6g}, "
            f"argmax_match={output_metrics['argmax_match']}"
        )

    return ModelbenchResult(
        candidate_id=candidate_id,
        passed=output_passed,
        baseline_ms=baseline_ms,
        patched_ms=patched_ms,
        speedup_pct=speedup_pct,
        warmup_iters=warmup_iters,
        measured_iters=measured_iters,
        output_max_abs_error=float(output_metrics["max_abs_error"]),
        output_mean_abs_error=float(output_metrics["mean_abs_error"]),
        output_max_rel_error=float(output_metrics["max_rel_error"]),
        output_argmax_match=bool(output_metrics["argmax_match"]),
        metadata={
            "model_id": model_id,
            "input_shape": list(input_shape),
            "input_seed": input_seed,
            "dtype": dtype_label,
            "device": device_label,
            "output_max_abs_tolerance": max_abs_tol,
            "output_mean_abs_tolerance": mean_abs_tol,
            "output_rel_denominator_min": _MODEL_OUTPUT_REL_DENOM_MIN,
            **patch_metadata,
        },
        error=error,
    )


def _patch_metadata(handler: OpHandler, spec: OpSpec, patch_result: Any) -> dict[str, Any]:
    if handler.patch_metadata is None:
        return {"patch_scope": getattr(patch_result, "patch_scope", "unknown")}
    return handler.patch_metadata(spec, patch_result)


def _model_output_metrics(torch: Any, baseline_logits: Any, patched_logits: Any) -> dict[str, float | bool]:
    diff = (patched_logits - baseline_logits).abs()
    denom = torch.maximum(baseline_logits.abs(), patched_logits.abs()).clamp_min(_MODEL_OUTPUT_REL_DENOM_MIN)
    baseline_next_token = baseline_logits[:, -1, :].argmax(dim=-1)
    patched_next_token = patched_logits[:, -1, :].argmax(dim=-1)
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "max_rel_error": float((diff / denom).max().item()),
        "argmax_match": bool(torch.equal(baseline_next_token, patched_next_token)),
    }


def _model_output_tolerances(dtype_label: str) -> tuple[float, float]:
    return _MODEL_OUTPUT_TOLERANCES.get(dtype_label.lower(), (1e-4, 1e-5))


def _load_model(transformers: Any, model_id: str, dtype: Any, device: Any):
    if "Qwen3.5" in model_id:
        model = transformers.AutoModelForMultimodalLM.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=True,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=True,
        )

    model.to(device)
    model.eval()
    return model


def _time_forward(torch: Any, device: Any, fn: Callable[[], Any], *, warmup_iters: int, measured_iters: int) -> float:
    if warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")
    if measured_iters <= 0:
        raise ValueError("measured_iters must be positive")
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
            return float(statistics.median(timings))

        timings = []
        for _ in range(measured_iters):
            start = time.perf_counter()
            fn()
            timings.append((time.perf_counter() - start) * 1000.0)
        return float(statistics.median(timings))


def _make_input_ids(
    torch: Any,
    model: Any,
    input_shape: tuple[int, ...],
    device: Any,
    *,
    input_seed: int = _DEFAULT_MODEL_INPUT_SEED,
):
    if len(input_shape) == 1:
        batch_size, seq_len = 1, input_shape[0]
    elif len(input_shape) >= 2:
        batch_size, seq_len = input_shape[0], input_shape[1]
    else:
        raise ValueError("input_shape must contain at least a sequence length")
    vocab_size = int(getattr(model.config, "vocab_size", 32000))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(input_seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator, device="cpu")
    return input_ids.to(device)


def _resolve_device(torch: Any, requested: str):
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA modelbench device, but torch.cuda.is_available() is false")
    return torch.device(requested)


def _resolve_dtype(torch: Any, dtype: str):
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype for modelbench: {dtype!r}") from exc


def _synchronize(torch: Any, device: Any) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _cleanup(torch: Any, device: Any) -> None:
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for modelbench") from exc
    return torch


def _import_transformers():
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError("transformers is required for modelbench") from exc
    return transformers
