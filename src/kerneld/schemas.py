from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

BackendName = Literal["triton", "torch", "cutlass", "cute_dsl"]


class KerneldModel(BaseModel):
    """Base model with predictable path serialization."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class RunConfig(KerneldModel):
    run_id: str
    run_dir: Path
    model_id: str
    op: str
    input_shape: tuple[int, ...]
    dtype: str
    device: str = "cuda"
    max_candidates: int = Field(gt=0)
    min_model_speedup_pct: float = 0.0

    @field_validator("input_shape", mode="before")
    @classmethod
    def _coerce_input_shape(cls, value: Any) -> tuple[int, ...]:
        return _coerce_int_tuple(value, "input_shape")


class Plan(KerneldModel):
    run_id: str
    model_id: str
    op: str
    input_shape: tuple[int, ...]
    dtype: str
    device: str
    max_candidates: int
    min_model_speedup_pct: float
    primary_metric: str = "full_model_latency_ms"
    acceptance_rules: list[str] = Field(
        default_factory=lambda: [
            "verification_passed",
            "microbench_candidate_faster_than_baseline",
            "modelbench_speedup_at_or_above_threshold",
        ]
    )
    agent_enabled: bool = False

    @field_validator("input_shape", mode="before")
    @classmethod
    def _coerce_input_shape(cls, value: Any) -> tuple[int, ...]:
        return _coerce_int_tuple(value, "input_shape")


class OpSpec(KerneldModel):
    op_type: str
    model_id: str
    input_shape: tuple[int, ...]
    dtype: str
    device: str

    @field_validator("input_shape", mode="before")
    @classmethod
    def _coerce_input_shape(cls, value: Any) -> tuple[int, ...]:
        return _coerce_int_tuple(value, "input_shape")


class RMSNormOpSpec(OpSpec):
    op_type: Literal["rmsnorm"] = "rmsnorm"
    module_path: str
    module_class: str
    input_stride: tuple[int, ...]
    hidden_size: int = Field(gt=0)
    weight_shape: tuple[int, ...]
    eps: float
    baseline_label: str

    @field_validator("input_stride", "weight_shape", mode="before")
    @classmethod
    def _coerce_tuple_fields(cls, value: Any) -> tuple[int, ...]:
        return _coerce_int_tuple(value, "tuple field")


class CandidateInfo(KerneldModel):
    candidate_id: str
    backend: BackendName
    entrypoint: str
    source_files: list[Path]
    build_required: bool
    build_artifact: Path | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationResult(KerneldModel):
    candidate_id: str
    passed: bool
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    atol: float | None = None
    rtol: float | None = None
    cases: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class MicrobenchResult(KerneldModel):
    candidate_id: str
    passed: bool
    baseline_ms: float | None = None
    candidate_ms: float | None = None
    baseline_mean_ms: float | None = None
    candidate_mean_ms: float | None = None
    speedup_pct: float | None = None
    warmup_iters: int | None = None
    measured_iters: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ModelbenchResult(KerneldModel):
    candidate_id: str
    passed: bool
    baseline_ms: float | None = None
    patched_ms: float | None = None
    speedup_pct: float | None = None
    warmup_iters: int | None = None
    measured_iters: int | None = None
    output_max_abs_error: float | None = None
    output_mean_abs_error: float | None = None
    output_max_rel_error: float | None = None
    output_argmax_match: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class SelectionResult(KerneldModel):
    accepted: bool
    candidate_id: str | None = None
    reason: str
    considered_candidates: list[str] = Field(default_factory=list)
    model_speedup_pct: float | None = None
    microbench_speedup_pct: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class RunSummary(KerneldModel):
    run_id: str
    run_dir: Path
    model_id: str
    op: str
    status: str
    selected_candidate_id: str | None = None
    artifacts: dict[str, Path] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _coerce_int_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError(f"{field_name} cannot be empty")
        return tuple(int(part) for part in parts)
    if isinstance(value, list):
        return tuple(int(part) for part in value)
    if isinstance(value, tuple):
        return tuple(int(part) for part in value)
    raise TypeError(f"{field_name} must be a comma-separated string or sequence of ints")
