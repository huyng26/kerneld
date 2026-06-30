from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kerneld.pipeline.candidates import list_candidate_ids, load_candidate_info
from kerneld.run_state import RunState
from kerneld.schemas import MicrobenchResult, ModelbenchResult, SelectionResult, VerificationResult


@dataclass(frozen=True)
class _EligibleCandidate:
    candidate_id: str
    modelbench: ModelbenchResult
    microbench: MicrobenchResult
    source_path: Path
    microbench_speedup_pct: float


def select_run(run_dir: Path) -> SelectionResult:
    state = RunState.load(run_dir)
    candidate_ids = list_candidate_ids(state)
    considered: list[str] = []
    eligible: list[_EligibleCandidate] = []
    rejections: dict[str, str] = {}
    threshold = state.config.min_model_speedup_pct if state.config is not None else 0.0

    for candidate_id in candidate_ids:
        considered.append(candidate_id)
        verification = _read_model(state, f"verification/{candidate_id}.json", VerificationResult)
        microbench = _read_model(state, f"microbench/{candidate_id}.json", MicrobenchResult)
        modelbench = _read_model(state, f"modelbench/{candidate_id}.json", ModelbenchResult)
        rejection = _rejection_reason(verification, microbench, modelbench, threshold)
        if rejection is not None:
            rejections[candidate_id] = rejection
            continue
        assert microbench is not None
        assert modelbench is not None
        source_path = _candidate_source_path(state, candidate_id)
        if source_path is None:
            rejections[candidate_id] = "candidate source file is missing"
            continue
        eligible.append(
            _EligibleCandidate(
                candidate_id=candidate_id,
                modelbench=modelbench,
                microbench=microbench,
                source_path=source_path,
                microbench_speedup_pct=_microbench_speedup_pct(microbench),
            )
        )

    if not eligible:
        result = SelectionResult(
            accepted=False,
            reason="no candidate satisfied verification, microbench, and modelbench acceptance rules",
            considered_candidates=considered,
            metadata={"rejections": rejections, "min_model_speedup_pct": threshold},
        )
        state.write_json("selection.json", result)
        return result

    eligible.sort(
        key=lambda item: (item.modelbench.speedup_pct or float("-inf"), item.microbench_speedup_pct),
        reverse=True,
    )
    winner = eligible[0]
    try:
        shutil.copyfile(winner.source_path, state.artifact_path("final_kernel"))
    except OSError as exc:
        result = SelectionResult(
            accepted=False,
            reason=f"failed to copy selected candidate to final_kernel.py: {exc}",
            considered_candidates=considered,
            metadata={
                "eligible_candidates": [item.candidate_id for item in eligible],
                "rejections": rejections,
                "min_model_speedup_pct": threshold,
            },
            error=str(exc),
        )
        state.write_json("selection.json", result)
        return result

    result = SelectionResult(
        accepted=True,
        candidate_id=winner.candidate_id,
        reason="selected highest modelbench speedup among accepted candidates",
        considered_candidates=considered,
        model_speedup_pct=winner.modelbench.speedup_pct,
        microbench_speedup_pct=winner.microbench_speedup_pct,
        metadata={
            "eligible_candidates": [item.candidate_id for item in eligible],
            "rejections": rejections,
            "min_model_speedup_pct": threshold,
            "final_kernel": str(state.artifact_path("final_kernel")),
        },
    )
    state.write_json("selection.json", result)
    return result


def _read_model(state: RunState, name: str, model_type: Any):
    path = state.path(name)
    if not path.exists():
        return None
    return model_type.model_validate(state.read_json(name))


def _rejection_reason(
    verification: VerificationResult | None,
    microbench: MicrobenchResult | None,
    modelbench: ModelbenchResult | None,
    threshold: float,
) -> str | None:
    if verification is None:
        return "verification result is missing"
    if verification.error:
        return f"verification recorded an error: {verification.error}"
    if not verification.passed:
        return "verification did not pass"
    if microbench is None:
        return "microbench result is missing"
    if microbench.error:
        return f"microbench recorded an error: {microbench.error}"
    if not microbench.passed:
        return "microbench did not pass"
    if microbench.baseline_ms is None or microbench.candidate_ms is None:
        return "microbench latency is incomplete"
    if microbench.candidate_ms >= microbench.baseline_ms:
        return "microbench candidate latency is not lower than baseline latency"
    if modelbench is None:
        return "modelbench result is missing"
    if modelbench.error:
        return f"modelbench recorded an error: {modelbench.error}"
    if not modelbench.passed:
        return "modelbench did not pass"
    if modelbench.speedup_pct is None:
        return "modelbench speedup is missing"
    if modelbench.speedup_pct < threshold:
        return "modelbench speedup is below the configured threshold"
    return None


def _candidate_source_path(state: RunState, candidate_id: str) -> Path | None:
    candidate = load_candidate_info(state, candidate_id)
    if not candidate.source_files:
        return None
    source_path = candidate.source_files[0]
    if not source_path.is_absolute():
        source_path = state.run_dir / source_path
    source_path = source_path.resolve()
    if not source_path.exists() or not source_path.is_file():
        return None
    return source_path


def _microbench_speedup_pct(microbench: MicrobenchResult) -> float:
    if microbench.speedup_pct is not None:
        return microbench.speedup_pct
    if microbench.baseline_ms is None or microbench.candidate_ms is None or microbench.baseline_ms <= 0:
        return float("-inf")
    return ((microbench.baseline_ms - microbench.candidate_ms) / microbench.baseline_ms) * 100.0
