from __future__ import annotations

from typing import Any

from kerneld.schemas import RMSNormOpSpec

RMSNORM_CLASS_MARKERS = ("rmsnorm", "rms_norm")
EPS_ATTRS = ("variance_epsilon", "eps", "epsilon")


def is_rmsnorm_module(module: Any) -> bool:
    class_name = module.__class__.__name__.lower()
    has_name_marker = any(marker in class_name for marker in RMSNORM_CLASS_MARKERS)
    return has_name_marker and hasattr(module, "weight") and any(
        hasattr(module, attr) for attr in EPS_ATTRS
    )


def find_rmsnorm_modules(model: Any) -> list[tuple[str, Any]]:
    return [(name, module) for name, module in model.named_modules() if is_rmsnorm_module(module)]


def get_module_by_path(model: Any, module_path: str) -> Any:
    current = model
    if not module_path:
        return current
    for part in module_path.split("."):
        if part.isdigit() and hasattr(current, "__getitem__"):
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def get_module_epsilon(module: Any) -> float:
    for attr in EPS_ATTRS:
        if hasattr(module, attr):
            return float(getattr(module, attr))
    raise ValueError(f"module {module.__class__.__name__} does not expose an RMSNorm epsilon")


def build_rmsnorm_spec(
    *,
    model_id: str,
    module_path: str,
    module: Any,
    input_tensor: Any,
    baseline_label: str = "hf_module",
) -> RMSNormOpSpec:
    weight = getattr(module, "weight", None)
    if weight is None:
        raise ValueError(f"module {module_path!r} does not expose a weight parameter")
    input_shape = tuple(int(dim) for dim in input_tensor.shape)
    input_stride = tuple(int(dim) for dim in input_tensor.stride())
    weight_shape = tuple(int(dim) for dim in weight.shape)
    if not input_shape:
        raise ValueError("RMSNorm input tensor must have at least one dimension")
    hidden_size = int(input_shape[-1])
    if weight_shape and int(weight_shape[-1]) != hidden_size:
        raise ValueError(
            f"RMSNorm weight hidden size {weight_shape[-1]} does not match input hidden size {hidden_size}"
        )
    return RMSNormOpSpec(
        model_id=model_id,
        module_path=module_path,
        module_class=module.__class__.__name__,
        input_shape=input_shape,
        input_stride=input_stride,
        hidden_size=hidden_size,
        weight_shape=weight_shape,
        dtype=_dtype_label(input_tensor.dtype),
        eps=get_module_epsilon(module),
        device=str(input_tensor.device),
        baseline_label=baseline_label,
    )


def _dtype_label(dtype: Any) -> str:
    label = str(dtype)
    if label.startswith("torch."):
        return label.removeprefix("torch.")
    return label
