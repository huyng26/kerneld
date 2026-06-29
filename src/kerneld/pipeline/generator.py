from __future__ import annotations

import json
import shutil
from pathlib import Path
from string import Template
from typing import Any

from kerneld.run_state import RunState
from kerneld.schemas import CandidateInfo, RMSNormOpSpec

_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "ops" / "rmsnorm" / "backends" / "triton" / "template.py.j2"


def generate_run(run_dir: Path) -> list[CandidateInfo]:
    state = RunState.load(run_dir)
    op_spec_payload = state.read_json("op_spec.json")
    if op_spec_payload.get("op_type") != "rmsnorm":
        raise ValueError(f"unsupported op spec type: {op_spec_payload.get('op_type')!r}")
    spec = RMSNormOpSpec.model_validate(op_spec_payload)
    candidates = generate_rmsnorm_triton_candidates(state, spec)
    state.write_json("candidates.json", {"candidates": [c.model_dump(mode="json") for c in candidates]})
    if candidates:
        shutil.copyfile(candidates[0].source_files[0], state.current_candidate_path())
    return candidates


def generate_rmsnorm_triton_candidates(state: RunState, spec: RMSNormOpSpec) -> list[CandidateInfo]:
    if state.config is None:
        max_candidates = 4
    else:
        max_candidates = state.config.max_candidates
    output_dir = state.path("candidates")
    output_dir.mkdir(parents=True, exist_ok=True)
    template = Template(_TEMPLATE_PATH.read_text())

    candidates: list[CandidateInfo] = []
    for params in _candidate_grid(spec.hidden_size, max_candidates):
        candidate_id = state.allocate_candidate_id()
        source_path = state.candidate_path(candidate_id)
        rendered = template.safe_substitute(
            candidate_id=candidate_id,
            hidden_size=spec.hidden_size,
            block_size=params["block_size"],
            num_warps=params["num_warps"],
            dtype_label=spec.dtype,
        )
        source_path.write_text(rendered)
        candidate = CandidateInfo(
            candidate_id=candidate_id,
            backend="triton",
            entrypoint="kernel_fn",
            source_files=[source_path],
            build_required=False,
            params=params,
            metadata={
                "op_type": spec.op_type,
                "module_path": spec.module_path,
                "template": str(_TEMPLATE_PATH),
            },
        )
        candidates.append(candidate)
    return candidates


def _candidate_grid(hidden_size: int, max_candidates: int) -> list[dict[str, Any]]:
    base_block = _next_power_of_2(hidden_size)
    block_sizes = [base_block]
    for candidate in (512, 1024, 2048, 4096):
        if candidate >= hidden_size and candidate not in block_sizes:
            block_sizes.append(candidate)
    block_sizes = sorted(block_sizes)

    grid = []
    for block_size in block_sizes:
        for num_warps in (4, 8):
            grid.append({"block_size": block_size, "num_warps": num_warps, "accum_dtype": "fp32"})
            if len(grid) >= max_candidates:
                return grid
    return grid


def _next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError("hidden size must be positive")
    return 1 << (value - 1).bit_length()
