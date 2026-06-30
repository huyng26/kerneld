from __future__ import annotations

import json
import shutil
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Literal, Sequence

from kerneld.pipeline.candidates import list_candidate_ids, load_candidate_info
from kerneld.pipeline.microbench import microbench_run
from kerneld.pipeline.modelbench import modelbench_run
from kerneld.pipeline.report import write_report
from kerneld.pipeline.selector import select_run
from kerneld.pipeline.verifier import verify_run
from kerneld.run_state import RunState
from kerneld.runners.commands import CommandResult, run_command
from kerneld.schemas import CandidateInfo, MicrobenchResult

AgentProvider = Literal["codex"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT_PATH = _REPO_ROOT / "AGENT_CONTRACT.md"
_TEMPLATE_PATH = _REPO_ROOT / "templates" / "program.md.j2"

_DEFAULT_MICROBENCH_WARMUP = 20
_DEFAULT_MICROBENCH_MEASURED = 100
_DEFAULT_MODELBENCH_WARMUP = 5
_DEFAULT_MODELBENCH_MEASURED = 20
_DEFAULT_REPEATED_CORRECTNESS_FAILURES = 2
_AGENT_SCOPE_DIR = ".agent_scope"
_AGENT_SCOPE_FILES_DIR = "files"
_AGENT_SCOPE_MANIFEST = "manifest.json"


@dataclass(frozen=True)
class AgentPreparation:
    run_dir: Path
    contract_path: Path
    program_path: Path
    task_path: Path
    editable_candidate_path: Path
    selected_starting_candidate: str | None
    commands: dict[str, str]


@dataclass(frozen=True)
class AgentLoopResult:
    run_dir: Path
    program_path: Path
    task_path: Path
    attempts: list[dict[str, Any]]
    stop_reason: str
    selected_candidate_id: str | None = None
    agent_provider: str | None = None
    agent_session_id: str | None = None


@dataclass(frozen=True)
class AgentStepResult:
    run_dir: Path
    candidate_id: str | None
    feedback_path: Path
    verification_passed: bool
    microbench_passed: bool | None = None
    modelbench_passed: bool | None = None
    selection_accepted: bool | None = None
    selected_candidate_id: str | None = None
    skipped_after: str | None = None
    scope_violations: list[str] = field(default_factory=list)
    current_changed: bool | None = None


@dataclass(frozen=True)
class CodexInvocation:
    command_result: CommandResult
    command_payload: dict[str, Any]
    session_id: str | None
    prompt_path: Path
    events_path: Path
    final_message_path: Path
    scope_violations: list[str]
    current_changed: bool


@dataclass(frozen=True)
class ScopeGuard:
    protected_files: dict[Path, bytes]
    allowed_files: set[Path]
    root: Path


def prepare_agent_run(
    run_dir: Path,
    *,
    candidate_id: str | None = None,
    max_attempts: int = 50,
    microbench_warmup_iters: int = _DEFAULT_MICROBENCH_WARMUP,
    microbench_measured_iters: int = _DEFAULT_MICROBENCH_MEASURED,
    modelbench_warmup_iters: int = _DEFAULT_MODELBENCH_WARMUP,
    modelbench_measured_iters: int = _DEFAULT_MODELBENCH_MEASURED,
    repeated_correctness_failures: int = _DEFAULT_REPEATED_CORRECTNESS_FAILURES,
) -> AgentPreparation:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if repeated_correctness_failures <= 0:
        raise ValueError("repeated_correctness_failures must be positive")

    _validate_contract(_CONTRACT_PATH)
    state = RunState.load(run_dir)
    state.ensure_layout()

    selected_starting_candidate = _choose_starting_candidate(state, candidate_id)
    _ensure_current_candidate(state, selected_starting_candidate)

    commands = _command_templates(
        state.run_dir,
        microbench_warmup_iters=microbench_warmup_iters,
        microbench_measured_iters=microbench_measured_iters,
        modelbench_warmup_iters=modelbench_warmup_iters,
        modelbench_measured_iters=modelbench_measured_iters,
    )
    program_path = state.path("program.md")
    task_path = state.path("agent_task.json")
    program_path.write_text(
        _render_program(
            state=state,
            selected_starting_candidate=selected_starting_candidate,
            commands=commands,
            max_attempts=max_attempts,
            repeated_correctness_failures=repeated_correctness_failures,
        )
    )
    state.write_json(
        "agent_task.json",
        {
            "run_dir": str(state.run_dir),
            "run_id": state.config.run_id if state.config is not None else None,
            "contract_path": str(_CONTRACT_PATH),
            "program_path": str(program_path),
            "editable_candidate_path": str(state.current_candidate_path()),
            "selected_starting_candidate": selected_starting_candidate,
            "commands": commands,
            "agent_feedback_path": commands["agent_feedback"],
            "benchmark_defaults": {
                "microbench": {
                    "warmup_iters": microbench_warmup_iters,
                    "measured_iters": microbench_measured_iters,
                },
                "modelbench": {
                    "warmup_iters": modelbench_warmup_iters,
                    "measured_iters": modelbench_measured_iters,
                },
            },
            "max_attempts": max_attempts,
            "stop_conditions": {
                "repeated_correctness_failures": repeated_correctness_failures,
                "missing_editable_candidate": True,
                "manual_mode_evaluations": max_attempts,
            },
        },
    )
    _refresh_agent_scope_baseline(state)
    return AgentPreparation(
        run_dir=state.run_dir,
        contract_path=_CONTRACT_PATH,
        program_path=program_path,
        task_path=task_path,
        editable_candidate_path=state.current_candidate_path(),
        selected_starting_candidate=selected_starting_candidate,
        commands=commands,
    )


def agent_step(
    run_dir: Path,
    *,
    microbench_warmup_iters: int = _DEFAULT_MICROBENCH_WARMUP,
    microbench_measured_iters: int = _DEFAULT_MICROBENCH_MEASURED,
    modelbench_warmup_iters: int = _DEFAULT_MODELBENCH_WARMUP,
    modelbench_measured_iters: int = _DEFAULT_MODELBENCH_MEASURED,
) -> AgentStepResult:
    state = RunState.load(run_dir)
    state.ensure_layout()
    current_path = state.current_candidate_path()
    if not current_path.exists():
        raise FileNotFoundError(f"editable candidate is missing: {current_path}")

    scope_violations = _restore_agent_scope_baseline(state)
    if scope_violations:
        details: dict[str, Any] = {"scope_violations": scope_violations}
        feedback_path = _write_agent_feedback(
            state,
            candidate_id=None,
            details=details,
            skipped_after="scope",
        )
        result = AgentStepResult(
            run_dir=state.run_dir,
            candidate_id=None,
            feedback_path=feedback_path,
            verification_passed=False,
            skipped_after="scope",
            scope_violations=scope_violations,
            current_changed=None,
        )
        _write_agent_step_result(state, result)
        _refresh_agent_scope_baseline(state)
        return result

    current_changed = _current_candidate_changed(state)
    if current_changed is False:
        details = {"current_changed": False}
        feedback_path = _write_agent_feedback(
            state,
            candidate_id=None,
            details=details,
            skipped_after="no_edit",
        )
        result = AgentStepResult(
            run_dir=state.run_dir,
            candidate_id=None,
            feedback_path=feedback_path,
            verification_passed=False,
            skipped_after="no_edit",
            current_changed=False,
        )
        _write_agent_step_result(state, result)
        _refresh_agent_scope_baseline(state)
        return result

    candidate = _snapshot_current_candidate(
        state,
        attempt_index=_next_agent_step_index(state),
        starting_candidate_id=None,
        command_result=None,
        agent_provider=None,
        source="agent_step",
    )
    details = {"candidate_id": candidate.candidate_id, "current_changed": current_changed}

    verification = verify_run(state.run_dir, candidate_id=candidate.candidate_id)
    details["verification"] = verification.model_dump(mode="json")
    microbench = None
    modelbench = None
    selection = None
    skipped_after = None

    if not verification.passed:
        skipped_after = "verification"
    else:
        microbench = microbench_run(
            state.run_dir,
            candidate_id=candidate.candidate_id,
            warmup_iters=microbench_warmup_iters,
            measured_iters=microbench_measured_iters,
        )
        details["microbench"] = microbench.model_dump(mode="json")
        if not _microbench_is_promising(microbench):
            skipped_after = "microbench"
        else:
            modelbench = modelbench_run(
                state.run_dir,
                candidate_id=candidate.candidate_id,
                warmup_iters=modelbench_warmup_iters,
                measured_iters=modelbench_measured_iters,
            )
            details["modelbench"] = modelbench.model_dump(mode="json")
            selection = select_run(state.run_dir)
            details["selection"] = selection.model_dump(mode="json")
            write_report(state.run_dir)

    feedback_path = _write_agent_feedback(
        state,
        candidate_id=candidate.candidate_id,
        details=details,
        skipped_after=skipped_after,
    )
    result = AgentStepResult(
        run_dir=state.run_dir,
        candidate_id=candidate.candidate_id,
        feedback_path=feedback_path,
        verification_passed=verification.passed,
        microbench_passed=microbench.passed if microbench is not None else None,
        modelbench_passed=modelbench.passed if modelbench is not None else None,
        selection_accepted=selection.accepted if selection is not None else None,
        selected_candidate_id=selection.candidate_id if selection is not None else None,
        skipped_after=skipped_after,
        current_changed=current_changed if current_changed is not None else True,
    )
    _write_agent_step_result(state, result)
    _refresh_agent_scope_baseline(state)
    return result


def run_agent_loop(
    run_dir: Path,
    *,
    candidate_id: str | None = None,
    max_attempts: int = 1,
    agent_command: Sequence[str] | None = None,
    agent_provider: AgentProvider | None = None,
    agent_model: str | None = None,
    timeout_s: float = 3600.0,
    microbench_warmup_iters: int = _DEFAULT_MICROBENCH_WARMUP,
    microbench_measured_iters: int = _DEFAULT_MICROBENCH_MEASURED,
    modelbench_warmup_iters: int = _DEFAULT_MODELBENCH_WARMUP,
    modelbench_measured_iters: int = _DEFAULT_MODELBENCH_MEASURED,
    repeated_correctness_failures: int = _DEFAULT_REPEATED_CORRECTNESS_FAILURES,
) -> AgentLoopResult:
    if agent_command and agent_provider is not None:
        raise ValueError("agent_command and agent_provider are mutually exclusive")
    if agent_model and agent_provider != "codex":
        raise ValueError("agent_model is only supported with agent_provider='codex'")

    preparation = prepare_agent_run(
        run_dir,
        candidate_id=candidate_id,
        max_attempts=max_attempts,
        microbench_warmup_iters=microbench_warmup_iters,
        microbench_measured_iters=microbench_measured_iters,
        modelbench_warmup_iters=modelbench_warmup_iters,
        modelbench_measured_iters=modelbench_measured_iters,
        repeated_correctness_failures=repeated_correctness_failures,
    )
    state = RunState.load(preparation.run_dir)
    attempts: list[dict[str, Any]] = []
    stop_reason: str | None = None
    selected_candidate_id: str | None = None
    correctness_failures = 0
    codex_session_id = _load_codex_session_id(state) if agent_provider == "codex" else None

    for attempt_index in range(max_attempts):
        command_payload = None
        if agent_provider == "codex":
            codex_invocation = _invoke_codex_agent(
                state,
                attempt_index=attempt_index,
                timeout_s=timeout_s,
                agent_model=agent_model,
                session_id=codex_session_id,
                attempts=attempts,
            )
            command_payload = codex_invocation.command_payload
            if codex_invocation.session_id:
                codex_session_id = codex_invocation.session_id
                _write_codex_session(state, codex_session_id, attempt_index, agent_model)
            if codex_invocation.scope_violations:
                stop_reason = "agent scope violation"
                attempts.append(
                    {
                        "attempt_index": attempt_index,
                        "candidate_id": None,
                        "agent_provider": "codex",
                        "agent_command": command_payload,
                        "codex_session_id": codex_session_id,
                        "prompt_path": str(codex_invocation.prompt_path),
                        "events_path": str(codex_invocation.events_path),
                        "final_message_path": str(codex_invocation.final_message_path),
                        "scope_violations": codex_invocation.scope_violations,
                        "stop_reason": stop_reason,
                    }
                )
                break
            if not codex_invocation.command_result.succeeded:
                stop_reason = (
                    "codex agent timed out" if codex_invocation.command_result.timed_out else "codex agent failed"
                )
                attempts.append(
                    {
                        "attempt_index": attempt_index,
                        "candidate_id": None,
                        "agent_provider": "codex",
                        "agent_command": command_payload,
                        "codex_session_id": codex_session_id,
                        "prompt_path": str(codex_invocation.prompt_path),
                        "events_path": str(codex_invocation.events_path),
                        "final_message_path": str(codex_invocation.final_message_path),
                        "stop_reason": stop_reason,
                    }
                )
                break
            if codex_session_id is None:
                stop_reason = "codex session id missing"
                attempts.append(
                    {
                        "attempt_index": attempt_index,
                        "candidate_id": None,
                        "agent_provider": "codex",
                        "agent_command": command_payload,
                        "prompt_path": str(codex_invocation.prompt_path),
                        "events_path": str(codex_invocation.events_path),
                        "final_message_path": str(codex_invocation.final_message_path),
                        "stop_reason": stop_reason,
                    }
                )
                break
            if not codex_invocation.current_changed:
                stop_reason = "codex agent produced no edit"
                attempts.append(
                    {
                        "attempt_index": attempt_index,
                        "candidate_id": None,
                        "agent_provider": "codex",
                        "agent_command": command_payload,
                        "codex_session_id": codex_session_id,
                        "prompt_path": str(codex_invocation.prompt_path),
                        "events_path": str(codex_invocation.events_path),
                        "final_message_path": str(codex_invocation.final_message_path),
                        "stop_reason": stop_reason,
                    }
                )
                break
        elif agent_command:
            command_result = run_command(
                agent_command,
                cwd=state.run_dir,
                timeout_s=timeout_s,
                log_path=state.path("logs", f"agent_attempt_{attempt_index:03d}.log"),
            )
            command_payload = _command_result_payload(command_result)
            if not command_result.succeeded:
                stop_reason = "agent command timed out" if command_result.timed_out else "agent command failed"
                attempts.append(
                    {
                        "attempt_index": attempt_index,
                        "candidate_id": None,
                        "agent_command": command_payload,
                        "stop_reason": stop_reason,
                    }
                )
                break
        elif attempt_index > 0:
            stop_reason = "manual mode evaluated one snapshot"
            break

        if not state.current_candidate_path().exists():
            stop_reason = "missing editable candidate"
            attempts.append(
                {
                    "attempt_index": attempt_index,
                    "candidate_id": None,
                    "agent_provider": agent_provider,
                    "agent_command": command_payload,
                    "codex_session_id": codex_session_id,
                    "stop_reason": stop_reason,
                }
            )
            break

        candidate = _snapshot_current_candidate(
            state,
            attempt_index=attempt_index,
            starting_candidate_id=preparation.selected_starting_candidate,
            command_result=command_payload,
            agent_provider=agent_provider,
        )
        attempt = {
            "attempt_index": attempt_index,
            "candidate_id": candidate.candidate_id,
            "agent_provider": agent_provider,
            "agent_command": command_payload,
            "codex_session_id": codex_session_id,
        }
        attempts.append(attempt)

        verification = verify_run(state.run_dir, candidate_id=candidate.candidate_id)
        attempt["verification_passed"] = verification.passed
        if verification.error:
            attempt["verification_error"] = verification.error
        if not verification.passed:
            correctness_failures += 1
            attempt["skipped_after"] = "verification"
            if correctness_failures >= repeated_correctness_failures:
                stop_reason = "repeated correctness failures"
                break
            if not agent_command and agent_provider is None:
                stop_reason = "manual mode evaluated one snapshot"
                break
            continue
        correctness_failures = 0

        microbench = microbench_run(
            state.run_dir,
            candidate_id=candidate.candidate_id,
            warmup_iters=microbench_warmup_iters,
            measured_iters=microbench_measured_iters,
        )
        attempt["microbench_passed"] = microbench.passed
        attempt["microbench_speedup_pct"] = microbench.speedup_pct
        if microbench.error:
            attempt["microbench_error"] = microbench.error
        if not _microbench_is_promising(microbench):
            attempt["skipped_after"] = "microbench"
            if not agent_command and agent_provider is None:
                stop_reason = "manual mode evaluated one snapshot"
                break
            continue

        modelbench = modelbench_run(
            state.run_dir,
            candidate_id=candidate.candidate_id,
            warmup_iters=modelbench_warmup_iters,
            measured_iters=modelbench_measured_iters,
        )
        attempt["modelbench_passed"] = modelbench.passed
        attempt["modelbench_speedup_pct"] = modelbench.speedup_pct
        if modelbench.error:
            attempt["modelbench_error"] = modelbench.error

        selection = select_run(state.run_dir)
        write_report(state.run_dir)
        attempt["selection_accepted"] = selection.accepted
        attempt["selected_candidate_id"] = selection.candidate_id
        selected_candidate_id = selection.candidate_id if selection.accepted else selected_candidate_id

        if not agent_command and agent_provider is None:
            stop_reason = "manual mode evaluated one snapshot"
            break

    if stop_reason is None:
        stop_reason = "max attempts reached" if agent_command or agent_provider else "manual mode evaluated one snapshot"

    summary = AgentLoopResult(
        run_dir=state.run_dir,
        program_path=preparation.program_path,
        task_path=preparation.task_path,
        attempts=attempts,
        stop_reason=stop_reason,
        selected_candidate_id=selected_candidate_id,
        agent_provider=agent_provider,
        agent_session_id=codex_session_id,
    )
    state.write_json("agent_loop.json", _agent_loop_result_payload(summary))
    return summary


def _invoke_codex_agent(
    state: RunState,
    *,
    attempt_index: int,
    timeout_s: float,
    agent_model: str | None,
    session_id: str | None,
    attempts: list[dict[str, Any]],
) -> CodexInvocation:
    logs_dir = state.path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = logs_dir / f"codex_attempt_{attempt_index:03d}_prompt.md"
    events_path = logs_dir / f"codex_attempt_{attempt_index:03d}_events.jsonl"
    final_message_path = logs_dir / f"codex_attempt_{attempt_index:03d}_final.md"
    prompt = _codex_prompt(state, attempt_index=attempt_index, attempts=attempts)
    prompt_path.write_text(prompt)

    current_path = state.current_candidate_path()
    current_before = current_path.read_bytes() if current_path.exists() else None
    guard = _create_scope_guard(state, allowed_files={current_path, final_message_path})
    cmd = _codex_command(
        state.run_dir,
        session_id=session_id,
        final_message_path=final_message_path,
        agent_model=agent_model,
    )
    result = run_command(cmd, cwd=state.run_dir, timeout_s=timeout_s, input_text=prompt)
    violations = _restore_scope_violations(guard)
    current_after = current_path.read_bytes() if current_path.exists() else None
    events_path.write_text(result.stdout)
    parsed_session_id = _parse_codex_thread_id(result.stdout) or session_id
    return CodexInvocation(
        command_result=result,
        command_payload={
            **_command_result_payload(result),
            "provider": "codex",
            "prompt_path": str(prompt_path),
            "events_path": str(events_path),
            "final_message_path": str(final_message_path),
            "session_id": parsed_session_id,
            "scope_violations": violations,
        },
        session_id=parsed_session_id,
        prompt_path=prompt_path,
        events_path=events_path,
        final_message_path=final_message_path,
        scope_violations=violations,
        current_changed=current_before != current_after,
    )


def _codex_command(
    run_dir: Path,
    *,
    session_id: str | None,
    final_message_path: Path,
    agent_model: str | None,
) -> list[str]:
    cmd = ["codex", "--sandbox", "workspace-write", "-C", str(run_dir)]
    if agent_model:
        cmd.extend(["--model", agent_model])
    if session_id is None:
        cmd.extend(["exec", "--json", "--skip-git-repo-check", "-o", str(final_message_path), "-"])
    else:
        cmd.extend(["exec", "resume", "--json", "--skip-git-repo-check", "-o", str(final_message_path), session_id, "-"])
    return cmd


def _codex_prompt(state: RunState, *, attempt_index: int, attempts: list[dict[str, Any]]) -> str:
    feedback = _attempt_feedback_text(attempts)
    return f"""You are the kerneld kernel optimization coding agent for attempt {attempt_index}.

Read ./program.md and ./agent_task.json for the full run context and permanent rules. Your only editable file is:

./candidates/current.py

Hard constraints:
- Edit only ./candidates/current.py.
- Do not edit program.md, agent_task.json, AGENT_CONTRACT.md, candidates.json, result JSON, reports, logs, verifier code, benchmark code, selector code, or any infrastructure.
- Do not run kerneld verify, microbench, modelbench, select, or report. kerneld will run the authoritative evaluation after you return.
- Preserve the candidate public function signature and RMSNorm semantics.
- Make one focused optimization attempt, then stop and summarize what changed.

Latest kerneld feedback:
{feedback}
"""


def _attempt_feedback_text(attempts: list[dict[str, Any]]) -> str:
    if not attempts:
        return "No prior agent-generated attempt has been evaluated in this loop."
    lines = []
    for attempt in attempts[-3:]:
        lines.append(f"- Attempt {attempt.get('attempt_index')} candidate `{attempt.get('candidate_id')}`:")
        if "verification_passed" in attempt:
            lines.append(f"  verification_passed={attempt.get('verification_passed')}")
        if attempt.get("verification_error"):
            lines.append(f"  verification_error={attempt.get('verification_error')}")
        if "microbench_passed" in attempt:
            lines.append(
                "  microbench_passed="
                f"{attempt.get('microbench_passed')}, speedup_pct={attempt.get('microbench_speedup_pct')}"
            )
        if attempt.get("microbench_error"):
            lines.append(f"  microbench_error={attempt.get('microbench_error')}")
        if "modelbench_passed" in attempt:
            lines.append(
                "  modelbench_passed="
                f"{attempt.get('modelbench_passed')}, speedup_pct={attempt.get('modelbench_speedup_pct')}"
            )
        if attempt.get("modelbench_error"):
            lines.append(f"  modelbench_error={attempt.get('modelbench_error')}")
        if attempt.get("skipped_after"):
            lines.append(f"  skipped_after={attempt.get('skipped_after')}")
        if "selection_accepted" in attempt:
            lines.append(
                "  selection_accepted="
                f"{attempt.get('selection_accepted')}, selected_candidate_id={attempt.get('selected_candidate_id')}"
            )
    return "\n".join(lines)


def _parse_codex_thread_id(stdout: str) -> str | None:
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started" and event.get("thread_id"):
            return str(event["thread_id"])
    return None


def _load_codex_session_id(state: RunState) -> str | None:
    path = state.path("codex_session.json")
    if not path.exists():
        return None
    payload = state.read_json("codex_session.json")
    session_id = payload.get("session_id")
    return str(session_id) if session_id else None


def _write_codex_session(state: RunState, session_id: str, attempt_index: int, agent_model: str | None) -> None:
    state.write_json(
        "codex_session.json",
        {
            "provider": "codex",
            "session_id": session_id,
            "last_attempt_index": attempt_index,
            "model": agent_model,
        },
    )


def _create_scope_guard(state: RunState, *, allowed_files: set[Path]) -> ScopeGuard:
    root = state.run_dir.resolve()
    allowed = {path.resolve() for path in allowed_files}
    protected: dict[Path, bytes] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in allowed:
            continue
        protected[resolved] = resolved.read_bytes()
    return ScopeGuard(protected_files=protected, allowed_files=allowed, root=root)


def _restore_scope_violations(guard: ScopeGuard) -> list[str]:
    violations: list[str] = []
    existing_files = {path.resolve() for path in guard.root.rglob("*") if path.is_file()}
    for path, original in guard.protected_files.items():
        if not path.exists():
            violations.append(str(path.relative_to(guard.root)))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(original)
            continue
        if path.read_bytes() != original:
            violations.append(str(path.relative_to(guard.root)))
            path.write_bytes(original)
    for path in sorted(existing_files - set(guard.protected_files) - guard.allowed_files):
        violations.append(str(path.relative_to(guard.root)))
        path.unlink()
    return sorted(set(violations))


def _agent_scope_path(state: RunState, *parts: str) -> Path:
    return state.path(_AGENT_SCOPE_DIR, *parts)


def _relative_run_path(state: RunState, path: Path) -> str:
    return path.resolve().relative_to(state.run_dir.resolve()).as_posix()


def _scope_copy_path(state: RunState, relative_path: str) -> Path:
    return _agent_scope_path(state, _AGENT_SCOPE_FILES_DIR, *relative_path.split("/"))


def _is_agent_scope_excluded(state: RunState, path: Path) -> bool:
    relative_path = _relative_run_path(state, path)
    return relative_path == "candidates/current.py" or relative_path.startswith(f"{_AGENT_SCOPE_DIR}/")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _refresh_agent_scope_baseline(state: RunState) -> None:
    scope_dir = _agent_scope_path(state)
    if scope_dir.exists():
        shutil.rmtree(scope_dir)
    files_dir = _agent_scope_path(state, _AGENT_SCOPE_FILES_DIR)
    files_dir.mkdir(parents=True, exist_ok=True)

    files: list[dict[str, str]] = []
    for path in sorted(state.run_dir.rglob("*")):
        if not path.is_file() or _is_agent_scope_excluded(state, path):
            continue
        relative_path = _relative_run_path(state, path)
        data = path.read_bytes()
        copy_path = _scope_copy_path(state, relative_path)
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        copy_path.write_bytes(data)
        files.append({"path": relative_path, "sha256": _sha256_bytes(data)})

    manifest = {
        "version": 1,
        "editable_candidate_path": "candidates/current.py",
        "files": files,
    }
    _agent_scope_path(state, _AGENT_SCOPE_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _restore_agent_scope_baseline(state: RunState) -> list[str]:
    manifest_path = _agent_scope_path(state, _AGENT_SCOPE_MANIFEST)
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    protected = {
        str(item["path"]): str(item.get("sha256", ""))
        for item in manifest.get("files", [])
        if "path" in item
    }
    violations: list[str] = []

    for relative_path, expected_hash in protected.items():
        path = state.path(*relative_path.split("/"))
        copy_path = _scope_copy_path(state, relative_path)
        if not path.exists():
            violations.append(relative_path)
            if copy_path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(copy_path, path)
            continue
        data = path.read_bytes()
        if expected_hash and _sha256_bytes(data) == expected_hash:
            continue
        violations.append(relative_path)
        if copy_path.exists():
            shutil.copyfile(copy_path, path)

    protected_paths = set(protected)
    for path in sorted(state.run_dir.rglob("*"), reverse=True):
        if not path.is_file() or _is_agent_scope_excluded(state, path):
            continue
        relative_path = _relative_run_path(state, path)
        if relative_path in protected_paths:
            continue
        violations.append(relative_path)
        path.unlink()
        _remove_empty_run_dirs(state, path.parent)

    return sorted(set(violations))


def _remove_empty_run_dirs(state: RunState, start: Path) -> None:
    root = state.run_dir.resolve()
    path = start.resolve()
    while path != root and path.exists():
        try:
            path.rmdir()
        except OSError:
            break
        path = path.parent


def _latest_numbered_candidate_path(state: RunState) -> Path | None:
    candidate_ids = sorted(candidate_id for candidate_id in list_candidate_ids(state) if candidate_id.startswith("candidate_"))
    for candidate_id in reversed(candidate_ids):
        source_path = _candidate_source_path(state, candidate_id)
        if source_path is not None and source_path.resolve() != state.current_candidate_path().resolve():
            return source_path
    return None


def _current_candidate_changed(state: RunState) -> bool | None:
    latest_path = _latest_numbered_candidate_path(state)
    if latest_path is None:
        return None
    return state.current_candidate_path().read_bytes() != latest_path.read_bytes()


def _next_agent_step_index(state: RunState) -> int:
    payload_path = state.path("candidates.json")
    if not payload_path.exists():
        return 0
    payload = state.read_json("candidates.json")
    indexes = [
        int(item.get("metadata", {}).get("attempt_index", -1))
        for item in payload.get("candidates", [])
        if item.get("metadata", {}).get("source") == "agent_step"
    ]
    return max(indexes, default=-1) + 1


def _write_agent_feedback(
    state: RunState,
    *,
    candidate_id: str | None,
    details: dict[str, Any],
    skipped_after: str | None,
) -> Path:
    candidate_label = candidate_id or "none"
    lines = [
        "# Agent Feedback",
        "",
        f"- Candidate: `{candidate_label}`",
        f"- Skipped after: `{skipped_after or 'none'}`",
        "",
    ]

    scope_violations = details.get("scope_violations") or []
    if scope_violations:
        lines.extend(["## Scope Violations", ""])
        lines.extend(f"- `{path}`" for path in scope_violations)
        lines.extend(["", "Protected files were restored and unexpected files were removed.", ""])

    if details.get("current_changed") is False:
        lines.extend(["## Edit Check", "", "`candidates/current.py` is unchanged from the latest candidate snapshot.", ""])

    verification = details.get("verification")
    lines.extend(["## Verification", ""])
    if verification is None:
        lines.extend(["Not run.", ""])
    else:
        artifact = f"verification/{candidate_id}.json" if candidate_id else ""
        lines.extend([
            f"- Passed: `{verification.get('passed')}`",
            f"- Error: `{verification.get('error') or ''}`",
            f"- Artifact: `{artifact}`",
            "",
        ])

    microbench = details.get("microbench")
    lines.extend(["## Microbench", ""])
    if microbench is None:
        lines.extend(["Not run.", ""])
    else:
        lines.extend([
            f"- Passed: `{microbench.get('passed')}`",
            f"- Baseline ms: `{microbench.get('baseline_ms')}`",
            f"- Candidate ms: `{microbench.get('candidate_ms')}`",
            f"- Speedup pct: `{microbench.get('speedup_pct')}`",
            f"- Error: `{microbench.get('error') or ''}`",
            f"- Artifact: `microbench/{candidate_id}.json`",
            "",
        ])
    modelbench = details.get("modelbench")
    lines.extend(["## Modelbench", ""])
    if modelbench is None:
        lines.extend(["Not run.", ""])
    else:
        lines.extend([
            f"- Passed: `{modelbench.get('passed')}`",
            f"- Baseline ms: `{modelbench.get('baseline_ms')}`",
            f"- Patched ms: `{modelbench.get('patched_ms')}`",
            f"- Speedup pct: `{modelbench.get('speedup_pct')}`",
            f"- Error: `{modelbench.get('error') or ''}`",
            f"- Artifact: `modelbench/{candidate_id}.json`",
            "",
        ])
    selection = details.get("selection")
    lines.extend(["## Selection", ""])
    if selection is None:
        lines.extend(["Not run.", ""])
    else:
        lines.extend([
            f"- Accepted: `{selection.get('accepted')}`",
            f"- Candidate: `{selection.get('candidate_id') or ''}`",
            f"- Reason: `{selection.get('reason')}`",
            f"- Model speedup pct: `{selection.get('model_speedup_pct')}`",
            f"- Microbench speedup pct: `{selection.get('microbench_speedup_pct')}`",
            "- Artifact: `selection.json`",
            "",
        ])
    lines.extend([
        "## Next Guidance",
        "",
        _agent_feedback_guidance(details, skipped_after),
        "",
    ])
    path = state.path("agent_feedback.md")
    path.write_text("\n".join(lines).rstrip() + "\n")
    return path


def _write_agent_step_result(state: RunState, result: AgentStepResult) -> None:
    state.write_json(
        "agent_step.json",
        {
            "candidate_id": result.candidate_id,
            "feedback_path": str(result.feedback_path),
            "verification_passed": result.verification_passed,
            "microbench_passed": result.microbench_passed,
            "modelbench_passed": result.modelbench_passed,
            "selection_accepted": result.selection_accepted,
            "selected_candidate_id": result.selected_candidate_id,
            "skipped_after": result.skipped_after,
            "scope_violations": result.scope_violations,
            "current_changed": result.current_changed,
        },
    )


def _agent_feedback_guidance(details: dict[str, Any], skipped_after: str | None) -> str:
    if skipped_after == "scope":
        return "Only edit candidates/current.py. Protected files were restored; remove any workflow changes and make one candidate edit before running agent-step again."
    if skipped_after == "no_edit":
        return "Edit candidates/current.py before running agent-step. Unchanged candidates are not evaluated by agent-step."
    if skipped_after == "verification":
        return "Fix correctness before attempting performance changes. Re-check masks, dtype preservation, shape handling, and RMSNorm math."
    if skipped_after == "microbench":
        return "Correctness passed, but standalone latency was not promising. Try one focused performance change or revert the last slowdown."
    selection = details.get("selection")
    if selection and selection.get("accepted"):
        return "This candidate is accepted. Continue only if you have a specific, low-risk idea to improve full-model speedup."
    if details.get("modelbench"):
        return "Modelbench ran. Use full-model speedup as the primary signal; do not optimize only for microbench if modelbench regresses."
    return "Use the latest failure or skipped stage to choose one focused next edit."


def _validate_contract(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing agent contract: {path}")
    text = path.read_text()
    required = ["Editable Scope", "Correctness Rules", "Benchmark Rules", "Stop Conditions"]
    missing = [item for item in required if item not in text]
    if missing:
        raise ValueError(f"agent contract is missing required sections: {', '.join(missing)}")


def _choose_starting_candidate(state: RunState, requested_candidate_id: str | None) -> str | None:
    if requested_candidate_id is not None:
        return requested_candidate_id
    selection_path = state.path("selection.json")
    if selection_path.exists():
        selection = state.read_json("selection.json")
        if selection.get("accepted") and selection.get("candidate_id"):
            return str(selection["candidate_id"])
    candidate_ids = list_candidate_ids(state)
    if candidate_ids:
        return candidate_ids[0]
    return None


def _ensure_current_candidate(state: RunState, selected_candidate_id: str | None) -> None:
    current_path = state.current_candidate_path()
    if selected_candidate_id is None:
        if current_path.exists():
            return
        raise FileNotFoundError("no candidate source is available for candidates/current.py")

    source_path = _candidate_source_path(state, selected_candidate_id)
    if source_path is None:
        raise FileNotFoundError(f"candidate source is missing for {selected_candidate_id}")
    if source_path.resolve() != current_path.resolve():
        current_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, current_path)


def _candidate_source_path(state: RunState, candidate_id: str) -> Path | None:
    candidate = load_candidate_info(state, candidate_id)
    source_paths = candidate.source_files or [state.candidate_path(candidate_id)]
    source_path = source_paths[0]
    if not source_path.is_absolute():
        source_path = state.run_dir / source_path
    source_path = source_path.resolve()
    if not source_path.exists() or not source_path.is_file():
        return None
    return source_path


def _command_templates(
    run_dir: Path,
    *,
    microbench_warmup_iters: int,
    microbench_measured_iters: int,
    modelbench_warmup_iters: int,
    modelbench_measured_iters: int,
) -> dict[str, str]:
    run_arg = str(run_dir)
    return {
        "agent_step": f"kerneld agent-step --run {run_arg}",
        "agent_feedback": str(run_dir / "agent_feedback.md"),
        "verify": f"kerneld verify --run {run_arg} --candidate <candidate_id>",
        "microbench": (
            f"kerneld microbench --run {run_arg} --candidate <candidate_id> "
            f"--warmup-iters {microbench_warmup_iters} --measured-iters {microbench_measured_iters}"
        ),
        "modelbench": (
            f"kerneld modelbench --run {run_arg} --candidate <candidate_id> "
            f"--warmup-iters {modelbench_warmup_iters} --measured-iters {modelbench_measured_iters} "
            "--input-seed 0"
        ),
        "select": f"kerneld select --run {run_arg}",
        "report": f"kerneld report --run {run_arg}",
    }


def _render_program(
    *,
    state: RunState,
    selected_starting_candidate: str | None,
    commands: dict[str, str],
    max_attempts: int,
    repeated_correctness_failures: int,
) -> str:
    template = Template(_TEMPLATE_PATH.read_text())
    return template.safe_substitute(
        run_dir=state.run_dir,
        run_id=state.config.run_id if state.config is not None else "unknown",
        model_id=state.config.model_id if state.config is not None else "unknown",
        op=state.config.op if state.config is not None else _op_summary_value(state, "op_type"),
        input_shape=state.config.input_shape if state.config is not None else "unknown",
        dtype=state.config.dtype if state.config is not None else _op_summary_value(state, "dtype"),
        device=state.config.device if state.config is not None else _op_summary_value(state, "device"),
        min_model_speedup_pct=state.config.min_model_speedup_pct if state.config is not None else 0.0,
        op_spec_path=state.artifact_path("op_spec"),
        contract_path=_CONTRACT_PATH,
        editable_candidate_path=state.current_candidate_path(),
        selected_starting_candidate=selected_starting_candidate or "candidates/current.py",
        current_best=_current_best_summary(state),
        agent_step_command=commands["agent_step"],
        agent_feedback_path=commands["agent_feedback"],
        verify_command=commands["verify"],
        microbench_command=commands["microbench"],
        modelbench_command=commands["modelbench"],
        select_command=commands["select"],
        report_command=commands["report"],
        max_attempts=max_attempts,
        repeated_correctness_failures=repeated_correctness_failures,
        op_summary=_op_summary(state),
    )


def _op_summary(state: RunState) -> str:
    path = state.artifact_path("op_spec")
    if not path.exists():
        return "op_spec.json is missing."
    spec = state.read_json("op_spec.json")
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
    ]
    return "\n".join(f"- {key}: `{spec[key]}`" for key in keys if key in spec)


def _op_summary_value(state: RunState, key: str) -> str:
    path = state.artifact_path("op_spec")
    if not path.exists():
        return "unknown"
    return str(state.read_json("op_spec.json").get(key, "unknown"))


def _current_best_summary(state: RunState) -> str:
    selection_path = state.path("selection.json")
    if not selection_path.exists():
        return "No selection has been recorded yet."
    selection = state.read_json("selection.json")
    if not selection.get("accepted"):
        return f"Selection rejected all candidates: {selection.get('reason', 'unknown reason')}"
    return (
        f"Selected `{selection.get('candidate_id')}` with model speedup "
        f"`{selection.get('model_speedup_pct')}` and microbench speedup "
        f"`{selection.get('microbench_speedup_pct')}`."
    )


def _snapshot_current_candidate(
    state: RunState,
    *,
    attempt_index: int,
    starting_candidate_id: str | None,
    command_result: dict[str, Any] | None,
    agent_provider: str | None = None,
    source: str = "agent_loop",
) -> CandidateInfo:
    candidate_id = state.allocate_candidate_id()
    source_path = state.candidate_path(candidate_id)
    shutil.copyfile(state.current_candidate_path(), source_path)
    metadata = {
        "source": source,
        "attempt_index": attempt_index,
        "starting_candidate_id": starting_candidate_id,
        "agent_command": command_result,
    }
    if agent_provider is not None:
        metadata["agent_provider"] = agent_provider
    candidate = CandidateInfo(
        candidate_id=candidate_id,
        backend="triton",
        entrypoint="kernel_fn",
        source_files=[source_path],
        build_required=False,
        metadata=metadata,
    )
    _append_candidate_info(state, candidate)
    return candidate


def _append_candidate_info(state: RunState, candidate: CandidateInfo) -> None:
    path = state.path("candidates.json")
    if path.exists():
        payload = state.read_json("candidates.json")
    else:
        payload = {"candidates": []}
    candidates = [item for item in payload.get("candidates", []) if item.get("candidate_id") != candidate.candidate_id]
    candidates.append(candidate.model_dump(mode="json"))
    payload["candidates"] = candidates
    state.write_json("candidates.json", payload)


def _microbench_is_promising(result: MicrobenchResult) -> bool:
    if not result.passed or result.error:
        return False
    if result.baseline_ms is not None and result.candidate_ms is not None:
        return result.candidate_ms < result.baseline_ms
    if result.speedup_pct is not None:
        return result.speedup_pct > 0.0
    return False


def _command_result_payload(result: CommandResult) -> dict[str, Any]:
    return {
        "cmd": result.cmd,
        "cwd": str(result.cwd),
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "duration_s": result.duration_s,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": result.error,
    }


def _agent_loop_result_payload(result: AgentLoopResult) -> dict[str, Any]:
    return {
        "run_dir": str(result.run_dir),
        "program_path": str(result.program_path),
        "task_path": str(result.task_path),
        "attempts": result.attempts,
        "stop_reason": result.stop_reason,
        "selected_candidate_id": result.selected_candidate_id,
        "agent_provider": result.agent_provider,
        "agent_session_id": result.agent_session_id,
    }
