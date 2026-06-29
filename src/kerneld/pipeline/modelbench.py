from __future__ import annotations

import gc
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from kerneld.pipeline.candidates import load_candidate_info
from kerneld.pipeline.integrator import patch_model
from kerneld.run_state import RunState
from kerneld.schemas import ModelbenchResult, RMSNormOpSpec


def modelbench_run(
    run_dir: Path,
    *,
    candidate_id: str,
    warmup_iters: int = 5,
    measured_iters: int = 20,
) -> ModelbenchResult:
    state = RunState.load(run_dir)
    if state.config is None:
        result = ModelbenchResult(candidate_id=candidate_id, passed=False, error="missing config.json")
        state.write_json(f"modelbench/{candidate_id}.json", result)
        return result
    try:
        spec_payload = state.read_json("op_spec.json")
        if spec_payload.get("op_type") != "rmsnorm":
            raise ValueError(f"unsupported op spec type: {spec_payload.get('op_type')!r}")
        spec = RMSNormOpSpec.model_validate(spec_payload)
        candidate = load_candidate_info(state, candidate_id)
        result = _benchmark_model_pair(
            model_id=state.config.model_id,
            input_shape=state.config.input_shape,
            dtype_label=state.config.dtype,
            device_label=state.config.device,
            spec=spec,
            candidate=candidate,
            candidate_id=candidate_id,
            warmup_iters=warmup_iters,
            measured_iters=measured_iters,
        )
    except Exception as exc:
        result = ModelbenchResult(
            candidate_id=candidate_id,
            passed=False,
            warmup_iters=warmup_iters,
            measured_iters=measured_iters,
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
    spec: RMSNormOpSpec,
    candidate,
    candidate_id: str,
    warmup_iters: int,
    measured_iters: int,
) -> ModelbenchResult:
    torch = _import_torch()
    transformers = _import_transformers()
    device = _resolve_device(torch, device_label)
    dtype = _resolve_dtype(torch, dtype_label)
    input_ids = None
    baseline_logits = None

    baseline_model = _load_model(transformers, model_id, dtype, device)
    input_ids = _make_input_ids(torch, baseline_model, input_shape, device)
    with torch.no_grad():
        baseline_logits = baseline_model(input_ids=input_ids).logits.detach().float().cpu()
    baseline_ms = _time_forward(
        torch,
        device,
        lambda: baseline_model(input_ids=input_ids),
        warmup_iters=warmup_iters,
        measured_iters=measured_iters,
    )
    del baseline_model
    _cleanup(torch, device)

    patched_model = _load_model(transformers, model_id, dtype, device)
    patch_result = patch_model(patched_model, spec, candidate, scope="compatible")
    with torch.no_grad():
        patched_logits_tensor = patched_model(input_ids=input_ids).logits.detach().float().cpu()
    patched_ms = _time_forward(
        torch,
        device,
        lambda: patched_model(input_ids=input_ids),
        warmup_iters=warmup_iters,
        measured_iters=measured_iters,
    )
    max_abs = (patched_logits_tensor - baseline_logits).abs().max().item()
    max_rel = ((patched_logits_tensor - baseline_logits).abs() / baseline_logits.abs().clamp_min(1e-8)).max().item()
    speedup_pct = ((baseline_ms - patched_ms) / baseline_ms) * 100.0 if baseline_ms > 0 else None
    del patched_model
    _cleanup(torch, device)

    return ModelbenchResult(
        candidate_id=candidate_id,
        passed=True,
        baseline_ms=baseline_ms,
        patched_ms=patched_ms,
        speedup_pct=speedup_pct,
        warmup_iters=warmup_iters,
        measured_iters=measured_iters,
        output_max_abs_error=float(max_abs),
        output_max_rel_error=float(max_rel),
        metadata={
            "model_id": model_id,
            "input_shape": list(input_shape),
            "dtype": dtype_label,
            "device": device_label,
            "module_path": spec.module_path,
            "patch_scope": patch_result.patch_scope,
            "patched_module_paths": patch_result.patched_module_paths,
            "num_patched_modules": len(patch_result.patched_module_paths),
            "skipped_modules": [
                {"module_path": decision.module_path, "reason": decision.reason}
                for decision in patch_result.skipped_modules
            ],
        },
    )


def _load_model(transformers: Any, model_id: str, dtype: Any, device: Any):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
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


def _make_input_ids(torch: Any, model: Any, input_shape: tuple[int, ...], device: Any):
    if len(input_shape) == 1:
        batch_size, seq_len = 1, input_shape[0]
    elif len(input_shape) >= 2:
        batch_size, seq_len = input_shape[0], input_shape[1]
    else:
        raise ValueError("input_shape must contain at least a sequence length")
    vocab_size = int(getattr(model.config, "vocab_size", 32000))
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


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
