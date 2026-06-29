from __future__ import annotations

from pathlib import Path
from typing import Any

from kerneld.ops.rmsnorm.spec import build_rmsnorm_spec, find_rmsnorm_modules
from kerneld.run_state import RunState
from kerneld.schemas import RMSNormOpSpec


def extract_run(run_dir: Path, *, module_path: str | None = None) -> RMSNormOpSpec:
    state = RunState.load(run_dir)
    if state.config is None:
        raise FileNotFoundError(f"missing config.json in run directory {state.run_dir}")
    spec = extract_op_spec(state, module_path=module_path)
    state.write_json("op_spec.json", spec)
    return spec


def extract_op_spec(state: RunState, *, module_path: str | None = None) -> RMSNormOpSpec:
    config = state.config
    if config is None:
        raise ValueError("RunState must have a loaded RunConfig")
    if config.op != "rmsnorm":
        raise ValueError(f"unsupported op for extraction: {config.op!r}")

    torch = _import_torch()
    transformers = _import_transformers()
    device = _resolve_device(torch, config.device)
    dtype = _resolve_dtype(torch, config.dtype)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    modules = find_rmsnorm_modules(model)
    if not modules:
        raise RuntimeError(f"no RMSNorm-like modules found in {config.model_id}")
    target_path, target_module = _select_module(modules, module_path)

    captured: dict[str, Any] = {}

    def hook(_module, inputs, _output):
        if inputs:
            captured["input"] = inputs[0].detach()

    handle = target_module.register_forward_hook(hook)
    try:
        input_ids = _make_input_ids(torch, model, config.input_shape, device)
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        handle.remove()

    input_tensor = captured.get("input")
    if input_tensor is None:
        raise RuntimeError(f"RMSNorm module {target_path!r} did not receive an input tensor")
    return build_rmsnorm_spec(
        model_id=config.model_id,
        module_path=target_path,
        module=target_module,
        input_tensor=input_tensor,
    )


def _select_module(modules: list[tuple[str, Any]], module_path: str | None) -> tuple[str, Any]:
    if module_path is None:
        return modules[0]
    for path, module in modules:
        if path == module_path:
            return path, module
    available = ", ".join(path for path, _module in modules[:10])
    raise ValueError(f"RMSNorm module {module_path!r} not found; first matches: {available}")


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
        raise RuntimeError("requested CUDA extraction device, but torch.cuda.is_available() is false")
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
        raise ValueError(f"unsupported dtype for extraction: {dtype!r}") from exc


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for `kerneld extract`; install the hf/triton extras") from exc
    return torch


def _import_transformers():
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError("transformers is required for `kerneld extract`; install the hf extra") from exc
    return transformers
