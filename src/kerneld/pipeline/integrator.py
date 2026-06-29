from __future__ import annotations

from typing import Any

from kerneld.backends.triton import TritonBackend
from kerneld.ops.rmsnorm.patchers import RMSNormPatchResult, patch_model as patch_rmsnorm_model
from kerneld.schemas import CandidateInfo, RMSNormOpSpec


def patch_model(
    model: Any,
    spec: RMSNormOpSpec,
    candidate: CandidateInfo,
    *,
    scope: str = "compatible",
) -> RMSNormPatchResult:
    if spec.op_type != "rmsnorm":
        raise ValueError(f"unsupported op type for patching: {spec.op_type!r}")
    if candidate.backend != "triton":
        raise ValueError(f"unsupported backend for v1 patching: {candidate.backend!r}")
    kernel_fn = TritonBackend().load_entrypoint(candidate)
    return patch_rmsnorm_model(model, spec, kernel_fn, scope=scope)
