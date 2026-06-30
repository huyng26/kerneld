from __future__ import annotations

from typing import Any

from kerneld.backends.triton import TritonBackend
from kerneld.ops.registry import get_op_handler
from kerneld.schemas import CandidateInfo, OpSpec


def patch_model(
    model: Any,
    spec: OpSpec,
    candidate: CandidateInfo,
    *,
    scope: str = "compatible",
) -> Any:
    handler = get_op_handler(spec.op_type)
    if handler.patch_model is None:
        raise ValueError(f"op type {spec.op_type!r} does not support model patching")
    if candidate.backend != "triton":
        raise ValueError(f"unsupported backend for v1 patching: {candidate.backend!r}")
    kernel_fn = TritonBackend().load_entrypoint(candidate)
    return handler.patch_model(model, spec, kernel_fn, scope=scope)
