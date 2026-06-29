from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable

from kerneld.ops.rmsnorm.spec import (
    get_module_by_path,
    get_module_epsilon,
    is_rmsnorm_module,
)
from kerneld.schemas import RMSNormOpSpec


@dataclass(frozen=True)
class RMSNormPatchDecision:
    module_path: str
    reason: str


@dataclass(frozen=True)
class RMSNormPatchResult:
    model: Any
    patch_scope: str
    patched_module_paths: list[str] = field(default_factory=list)
    skipped_modules: list[RMSNormPatchDecision] = field(default_factory=list)


def patch_model(
    model: Any,
    spec: RMSNormOpSpec,
    kernel_fn: Callable[..., Any],
    *,
    scope: str = "compatible",
) -> RMSNormPatchResult:
    if scope not in {"single", "compatible"}:
        raise ValueError(f"unsupported RMSNorm patch scope: {scope!r}")

    if scope == "single":
        module = get_module_by_path(model, spec.module_path)
        reason = rmsnorm_incompatibility_reason(module, spec)
        if reason is not None:
            raise ValueError(f"target RMSNorm module {spec.module_path!r} is incompatible: {reason}")
        replacement = PatchedRMSNorm(module, kernel_fn, hidden_size=spec.hidden_size)
        replace_module_by_path(model, spec.module_path, replacement)
        return RMSNormPatchResult(
            model=model,
            patch_scope=scope,
            patched_module_paths=[spec.module_path],
        )

    compatible, skipped = find_compatible_rmsnorm_modules(model, spec)
    if not compatible:
        raise ValueError(f"no compatible RMSNorm modules found for {spec.module_path!r}")

    for module_path, module in compatible:
        replacement = PatchedRMSNorm(module, kernel_fn, hidden_size=spec.hidden_size)
        replace_module_by_path(model, module_path, replacement)

    return RMSNormPatchResult(
        model=model,
        patch_scope=scope,
        patched_module_paths=[module_path for module_path, _ in compatible],
        skipped_modules=skipped,
    )


def find_compatible_rmsnorm_modules(
    model: Any,
    spec: RMSNormOpSpec,
) -> tuple[list[tuple[str, Any]], list[RMSNormPatchDecision]]:
    compatible: list[tuple[str, Any]] = []
    skipped: list[RMSNormPatchDecision] = []
    for module_path, module in model.named_modules():
        if not module_path or isinstance(module, PatchedRMSNorm) or not is_rmsnorm_module(module):
            continue
        reason = rmsnorm_incompatibility_reason(module, spec)
        if reason is None:
            compatible.append((module_path, module))
        else:
            skipped.append(RMSNormPatchDecision(module_path=module_path, reason=reason))
    return compatible, skipped


def rmsnorm_incompatibility_reason(module: Any, spec: RMSNormOpSpec) -> str | None:
    if not is_rmsnorm_module(module):
        return "not an RMSNorm module"
    if module.__class__.__name__ != spec.module_class:
        return f"class {module.__class__.__name__!r} does not match {spec.module_class!r}"

    weight = getattr(module, "weight", None)
    if weight is None:
        return "missing weight"
    weight_shape = tuple(int(dim) for dim in getattr(weight, "shape", ()))
    if weight_shape != tuple(spec.weight_shape):
        return f"weight shape {weight_shape!r} does not match {tuple(spec.weight_shape)!r}"
    if not weight_shape or int(weight_shape[-1]) != int(spec.hidden_size):
        return f"hidden size does not match {spec.hidden_size}"

    eps = get_module_epsilon(module)
    if not math.isclose(eps, float(spec.eps), rel_tol=0.0, abs_tol=1e-12):
        return f"epsilon {eps!r} does not match {spec.eps!r}"
    if not _is_floating_weight(weight):
        return "weight dtype is not floating point"
    return None


def _nn_module_base():
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("torch is required for RMSNorm patching") from exc
    return nn.Module


class PatchedRMSNorm(_nn_module_base()):
    def __init__(
        self,
        original_module: Any,
        kernel_fn: Callable[..., Any],
        *,
        hidden_size: int,
        fallback_on_error: bool = False,
    ) -> None:
        super().__init__()
        self.original_module = original_module
        self.kernel_fn = kernel_fn
        self.hidden_size = int(hidden_size)
        self.eps = get_module_epsilon(original_module)
        self.fallback_on_error = fallback_on_error
        if hasattr(original_module, "variance_epsilon"):
            self.variance_epsilon = self.eps
        if hasattr(original_module, "epsilon"):
            self.epsilon = self.eps

    @property
    def weight(self):
        return self.original_module.weight

    def forward(self, hidden_states, *args, **kwargs):
        if args or kwargs:
            return self.original_module(hidden_states, *args, **kwargs)
        if not hasattr(hidden_states, "shape") or int(hidden_states.shape[-1]) != self.hidden_size:
            return self.original_module(hidden_states)
        try:
            return self.kernel_fn(hidden_states, self.weight, self.eps)
        except Exception:
            if self.fallback_on_error:
                return self.original_module(hidden_states)
            raise


def replace_module_by_path(model: Any, module_path: str, replacement: Any) -> None:
    parent_path, _, child_name = module_path.rpartition(".")
    parent = get_module_by_path(model, parent_path) if parent_path else model
    if child_name.isdigit() and hasattr(parent, "__setitem__"):
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


def _is_floating_weight(weight: Any) -> bool:
    is_floating_point = getattr(weight, "is_floating_point", None)
    if callable(is_floating_point):
        return bool(is_floating_point())
    dtype = str(getattr(weight, "dtype", ""))
    return any(label in dtype for label in ("float", "bfloat", "half"))
