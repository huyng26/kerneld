from __future__ import annotations

import shutil
from pathlib import Path

from kerneld.ops.registry import validate_op_spec
from kerneld.run_state import RunState
from kerneld.schemas import CandidateInfo


def generate_run(run_dir: Path) -> list[CandidateInfo]:
    state = RunState.load(run_dir)
    op_spec_payload = state.read_json("op_spec.json")
    handler, spec = validate_op_spec(op_spec_payload)
    if handler.generate_candidates is None:
        raise ValueError(f"op type {handler.op_type!r} does not support candidate generation")
    candidates = handler.generate_candidates(state, spec)
    state.write_json("candidates.json", {"candidates": [c.model_dump(mode="json") for c in candidates]})
    if candidates:
        shutil.copyfile(candidates[0].source_files[0], state.current_candidate_path())
    return candidates
