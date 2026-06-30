from __future__ import annotations

from pathlib import Path
from typing import Any

from kerneld.pipeline.candidates import list_candidate_ids
from kerneld.run_state import RunState


def write_report(run_dir: Path) -> Path:
    state = RunState.load(run_dir)
    lines = ["# Kerneld Run Report", ""]

    _append_run_summary(lines, state)
    _append_target_op_spec(lines, state)
    _append_candidate_table(lines, state)
    _append_result_summary(lines, "Verification Summary", state, "verification")
    _append_result_summary(lines, "Microbench Summary", state, "microbench")
    _append_result_summary(lines, "Modelbench Summary", state, "modelbench")
    _append_selected_kernel(lines, state)
    _append_reproduce_commands(lines, state)
    _append_known_limitations(lines)

    report_path = state.path("report.md")
    report_path.write_text("\n".join(lines).rstrip() + "\n")
    return report_path


def _append_run_summary(lines: list[str], state: RunState) -> None:
    lines.extend(["## Run Summary", ""])
    if state.config is None:
        lines.extend(["`config.json` has not been written.", ""])
        return
    lines.extend(
        [
            f"- Run ID: `{state.config.run_id}`",
            f"- Run directory: `{state.run_dir}`",
            f"- Model: `{state.config.model_id}`",
            f"- Op: `{state.config.op}`",
            f"- Input shape: `{state.config.input_shape}`",
            f"- Dtype/device: `{state.config.dtype}` / `{state.config.device}`",
            f"- Minimum model speedup: `{state.config.min_model_speedup_pct:.2f}%`",
            "",
        ]
    )


def _append_target_op_spec(lines: list[str], state: RunState) -> None:
    spec = _read_optional(state, "op_spec.json")
    lines.extend(["## Target Op Spec", ""])
    if spec is None:
        lines.extend(["`op_spec.json` has not been written.", ""])
        return
    keys = [
        "op_type",
        "model_id",
        "module_path",
        "module_class",
        "input_shape",
        "input_stride",
        "hidden_size",
        "weight_shape",
        "dtype",
        "device",
        "eps",
        "baseline_label",
    ]
    for key in keys:
        if key in spec:
            lines.append(f"- {key}: `{spec[key]}`")
    lines.append("")


def _append_candidate_table(lines: list[str], state: RunState) -> None:
    candidate_ids = list_candidate_ids(state)
    lines.extend(["## Candidate Table", ""])
    if not candidate_ids:
        lines.extend(["No candidates recorded.", ""])
        return
    lines.extend(
        [
            "| Candidate | Verify | Microbench speedup | Model speedup | Selected |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    selection = _read_optional(state, "selection.json") or {}
    selected_id = selection.get("candidate_id") if selection.get("accepted") else None
    for candidate_id in candidate_ids:
        verification = _read_optional(state, f"verification/{candidate_id}.json")
        microbench = _read_optional(state, f"microbench/{candidate_id}.json")
        modelbench = _read_optional(state, f"modelbench/{candidate_id}.json")
        lines.append(
            "| "
            f"`{candidate_id}` | "
            f"{_status(verification)} | "
            f"{_fmt_pct(_value(microbench, 'speedup_pct'))} | "
            f"{_fmt_pct(_value(modelbench, 'speedup_pct'))} | "
            f"{_yes_no(candidate_id == selected_id)} |"
        )
    lines.append("")


def _append_result_summary(lines: list[str], title: str, state: RunState, kind: str) -> None:
    candidate_ids = list_candidate_ids(state)
    lines.extend([f"## {title}", ""])
    if not candidate_ids:
        lines.extend(["No candidates recorded.", ""])
        return

    if kind == "verification":
        lines.extend(
            [
                "| Candidate | Status | Max abs error | Max rel error | Error |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
    elif kind == "microbench":
        lines.extend(
            [
                "| Candidate | Status | Baseline ms | Candidate ms | Speedup | Error |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
    else:
        lines.extend(
            [
                "| Candidate | Status | Baseline ms | Patched ms | Speedup | Error |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )

    for candidate_id in candidate_ids:
        payload = _read_optional(state, f"{kind}/{candidate_id}.json")
        if kind == "verification":
            lines.append(
                f"| `{candidate_id}` | {_status(payload)} | "
                f"{_fmt_number(_value(payload, 'max_abs_error'))} | "
                f"{_fmt_number(_value(payload, 'max_rel_error'))} | "
                f"{_fmt_error(_value(payload, 'error'))} |"
            )
        elif kind == "microbench":
            lines.append(
                f"| `{candidate_id}` | {_status(payload)} | "
                f"{_fmt_ms(_value(payload, 'baseline_ms'))} | "
                f"{_fmt_ms(_value(payload, 'candidate_ms'))} | "
                f"{_fmt_pct(_value(payload, 'speedup_pct'))} | "
                f"{_fmt_error(_value(payload, 'error'))} |"
            )
        else:
            lines.append(
                f"| `{candidate_id}` | {_status(payload)} | "
                f"{_fmt_ms(_value(payload, 'baseline_ms'))} | "
                f"{_fmt_ms(_value(payload, 'patched_ms'))} | "
                f"{_fmt_pct(_value(payload, 'speedup_pct'))} | "
                f"{_fmt_error(_value(payload, 'error'))} |"
            )
    lines.append("")


def _append_selected_kernel(lines: list[str], state: RunState) -> None:
    selection = _read_optional(state, "selection.json")
    final_kernel = state.artifact_path("final_kernel")
    lines.extend(["## Selected Kernel", ""])
    if selection is None:
        lines.extend(["`selection.json` has not been written.", ""])
        return
    lines.extend(
        [
            f"- Accepted: `{selection.get('accepted')}`",
            f"- Candidate: `{selection.get('candidate_id') or 'none'}`",
            f"- Reason: `{selection.get('reason')}`",
            f"- Model speedup: `{_fmt_pct(selection.get('model_speedup_pct'))}`",
            f"- Microbench speedup: `{_fmt_pct(selection.get('microbench_speedup_pct'))}`",
            f"- Final kernel: `{final_kernel}` ({'present' if final_kernel.exists() else 'missing'})",
            "",
        ]
    )


def _append_reproduce_commands(lines: list[str], state: RunState) -> None:
    candidate_ids = list_candidate_ids(state)
    lines.extend(["## Reproduce Commands", ""])
    lines.append(f"- Generate report: `kerneld report --run {state.run_dir}`")
    if candidate_ids:
        for candidate_id in candidate_ids:
            lines.append(
                f"- Verify `{candidate_id}`: `kerneld verify --run {state.run_dir} --candidate {candidate_id}`"
            )
            lines.append(
                f"- Microbench `{candidate_id}`: "
                f"`kerneld microbench --run {state.run_dir} --candidate {candidate_id}`"
            )
            lines.append(
                f"- Modelbench `{candidate_id}`: "
                f"`kerneld modelbench --run {state.run_dir} --candidate {candidate_id}`"
            )
    lines.append(f"- Select winner: `kerneld select --run {state.run_dir}`")
    lines.append("")


def _append_known_limitations(lines: list[str]) -> None:
    lines.extend(
        [
            "## Known Limitations",
            "",
            "- V1 selection is scoped to RMSNorm candidates and the Triton backend.",
            "- Full-model benchmark speedup is the primary acceptance signal.",
            "- CUTLASS, CuTe DSL, fused residual add plus RMSNorm, and Hub packaging are reserved for future milestones.",
            "",
        ]
    )


def _read_optional(state: RunState, artifact: str) -> dict[str, Any] | None:
    path = state.path(artifact)
    if not path.exists():
        return None
    return state.read_json(artifact)


def _value(payload: dict[str, Any] | None, key: str) -> Any:
    if payload is None:
        return None
    return payload.get(key)


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


def _fmt_ms(value: Any) -> str:
    if value is None:
        return "missing"
    return f"{float(value):.4f}"


def _fmt_number(value: Any) -> str:
    if value is None:
        return "missing"
    return f"{float(value):.6g}"


def _fmt_error(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
