from __future__ import annotations

from pathlib import Path
from typing import Any

from kerneld.pipeline.candidates import list_candidate_ids
from kerneld.run_state import RunState


def write_report(run_dir: Path) -> Path:
    state = RunState.load(run_dir)
    lines = ["# Kerneld Run Report", ""]
    if state.config is not None:
        lines.extend(
            [
                "## Run",
                "",
                f"- Run ID: `{state.config.run_id}`",
                f"- Model: `{state.config.model_id}`",
                f"- Op: `{state.config.op}`",
                f"- Input shape: `{state.config.input_shape}`",
                f"- Dtype/device: `{state.config.dtype}` / `{state.config.device}`",
                "",
            ]
        )

    _append_json_summary(lines, "Op Spec", state, "op_spec.json")
    _append_candidates(lines, state)
    _append_json_summary(lines, "Selection", state, "selection.json")

    report_path = state.path("report.md")
    report_path.write_text("\n".join(lines).rstrip() + "\n")
    return report_path


def _append_candidates(lines: list[str], state: RunState) -> None:
    candidate_ids = list_candidate_ids(state)
    lines.extend(["## Candidates", ""])
    if not candidate_ids:
        lines.extend(["No candidates recorded.", ""])
        return
    lines.extend([
        "| Candidate | Verify | Microbench speedup | Model speedup |",
        "| --- | --- | ---: | ---: |",
    ])
    for candidate_id in candidate_ids:
        verification = _read_optional(state, f"verification/{candidate_id}.json")
        microbench = _read_optional(state, f"microbench/{candidate_id}.json")
        modelbench = _read_optional(state, f"modelbench/{candidate_id}.json")
        verify_status = _status(verification)
        micro_speed = _fmt_pct(microbench.get("speedup_pct") if microbench else None)
        model_speed = _fmt_pct(modelbench.get("speedup_pct") if modelbench else None)
        lines.append(f"| `{candidate_id}` | {verify_status} | {micro_speed} | {model_speed} |")
    lines.append("")


def _append_json_summary(lines: list[str], title: str, state: RunState, artifact: str) -> None:
    payload = _read_optional(state, artifact)
    lines.extend([f"## {title}", ""])
    if payload is None:
        lines.extend([f"`{artifact}` has not been written.", ""])
        return
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"- {key}: `{value}`")
    lines.append("")


def _read_optional(state: RunState, artifact: str) -> dict[str, Any] | None:
    path = state.path(artifact)
    if not path.exists():
        return None
    return state.read_json(artifact)


def _status(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "missing"
    if payload.get("passed") is True:
        return "passed"
    if payload.get("accepted") is True:
        return "accepted"
    return "failed"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "missing"
    return f"{float(value):.2f}%"
