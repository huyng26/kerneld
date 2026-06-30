from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from kerneld.schemas import CandidateInfo, MicrobenchResult, OpSpec, VerificationResult


@dataclass(frozen=True)
class OpHandler:
    op_type: str
    spec_model: type[OpSpec]
    generate_candidates: Callable[..., list[CandidateInfo]] | None = None
    verify_kernel: Callable[..., VerificationResult] | None = None
    benchmark_kernel: Callable[..., MicrobenchResult] | None = None
    patch_model: Callable[..., Any] | None = None
    patch_metadata: Callable[..., dict[str, Any]] | None = None


def get_op_handler(op_type: str) -> OpHandler:
    handlers = _op_handlers()
    try:
        return handlers[op_type]
    except KeyError as exc:
        supported = ", ".join(sorted(handlers)) or "<none>"
        raise ValueError(f"unsupported op type {op_type!r}; supported ops: {supported}") from exc


def validate_op_spec(payload: dict[str, Any]) -> tuple[OpHandler, OpSpec]:
    op_type = payload.get("op_type")
    if not isinstance(op_type, str):
        raise ValueError("op spec is missing string field 'op_type'")
    handler = get_op_handler(op_type)
    return handler, handler.spec_model.model_validate(payload)


@lru_cache(maxsize=1)
def _op_handlers() -> dict[str, OpHandler]:
    from kerneld.ops.rmsnorm.handler import RMSNORM_HANDLER

    return {
        RMSNORM_HANDLER.op_type: RMSNORM_HANDLER,
    }
