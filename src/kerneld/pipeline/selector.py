from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from kerneld.pipeline.candidates import list_candidate_ids, load_candidate_info
from kerneld.run_state import RunState
from kerneld.schemas import MicrobenchResult, ModelbenchResult, SelectionResult, VerificationResult


def select_run(run_dir: Path) -> SelectionResult:
    state = RunState.load(run_dir)
    candidate_ids = list_candidate_ids(state)
    considered = []
    eligible = []
    threshold = state.config.min_model_speedup_pct if state.config is not None else 0.0

    for candidate_id in candidate_ids:
        considered.append(candidate_id)
        verification = _read_model(state, f"verification/{candidate_id}.json", VerificationResult)
        microbench = _read_model(state, f"microbench/{candidate_id}.json", MicrobenchResult)
        modelbench = _read_model(state, f"modelbench/{candidate_id}.json", ModelbenchResult)
        if verification is None or not verification.passed:
            continue
        if microbench is None or not microbench.passed:
            continue
        if microbench.baseline_ms is None or microbench.candidate_ms is None:
            continue
        if microbench.candidate_ms >= microbench.baseline_ms:
            continue
        if modelbench is None or not modelbench.passed:
            continue
        if modelbench.speedup_pct is None or modelbench.speedup_pct < threshold:
            continue
        eligible.append((candidate_id, modelbench, microbench))

    if not eligible:
        result = SelectionResult(
            accepted=False,
            reason="no candidate satisfied verification, microbench, and modelbench acceptance rules",
            considered_candidates=considered,
        )
        state.write_json("selection.json", result)
        return result

    eligible.sort(key=lambda item: (item[1].speedup_pct or float("-inf"), item[2].speedup_pct or float("-inf")), reverse=True)
    winner_id, winner_modelbench, winner_microbench = eligible[0]
    candidate = load_candidate_info(state, winner_id)
    if candidate.source_files:
        shutil.copyfile(candidate.source_files[0], state.artifact_path("final_kernel"))
    result = SelectionResult(
        accepted=True,
        candidate_id=winner_id,
        reason="selected highest modelbench speedup among accepted candidates",
        considered_candidates=considered,
        model_speedup_pct=winner_modelbench.speedup_pct,
        microbench_speedup_pct=winner_microbench.speedup_pct,
    )
    state.write_json("selection.json", result)
    return result


def _read_model(state: RunState, name: str, model_type: Any):
    path = state.path(name)
    if not path.exists():
        return None
    return model_type.model_validate(state.read_json(name))
