from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CommandResult:
    cmd: list[str]
    cwd: Path
    returncode: int | None
    timed_out: bool
    duration_s: float
    stdout: str
    stderr: str
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.timed_out and self.error is None


def run_command(
    cmd: Sequence[str],
    cwd: Path,
    timeout_s: float,
    log_path: Path | None = None,
    input_text: str | None = None,
) -> CommandResult:
    """Run an external command without a shell and capture its output."""
    command = [str(part) for part in cmd]
    cwd = cwd.resolve()
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            timeout=timeout_s,
            input=input_text,
            capture_output=True,
            text=True,
            shell=False,
            check=False,
        )
        result = CommandResult(
            cmd=command,
            cwd=cwd,
            returncode=completed.returncode,
            timed_out=False,
            duration_s=time.perf_counter() - start,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        result = CommandResult(
            cmd=command,
            cwd=cwd,
            returncode=None,
            timed_out=True,
            duration_s=time.perf_counter() - start,
            stdout=_to_text(exc.stdout),
            stderr=_to_text(exc.stderr),
            error=f"command timed out after {timeout_s:.2f}s",
        )
    except OSError as exc:
        result = CommandResult(
            cmd=command,
            cwd=cwd,
            returncode=127,
            timed_out=False,
            duration_s=time.perf_counter() - start,
            stdout="",
            stderr=str(exc),
            error=str(exc),
        )

    if log_path is not None:
        _write_log(log_path, result)
    return result


def run_interactive_command(
    cmd: Sequence[str],
    cwd: Path,
    timeout_s: float,
    log_path: Path | None = None,
) -> CommandResult:
    """Run an external command attached to the current terminal."""
    command = [str(part) for part in cmd]
    cwd = cwd.resolve()
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            timeout=timeout_s,
            shell=False,
            check=False,
        )
        result = CommandResult(
            cmd=command,
            cwd=cwd,
            returncode=completed.returncode,
            timed_out=False,
            duration_s=time.perf_counter() - start,
            stdout="",
            stderr="",
        )
    except subprocess.TimeoutExpired as exc:
        result = CommandResult(
            cmd=command,
            cwd=cwd,
            returncode=None,
            timed_out=True,
            duration_s=time.perf_counter() - start,
            stdout=_to_text(exc.stdout),
            stderr=_to_text(exc.stderr),
            error=f"command timed out after {timeout_s:.2f}s",
        )
    except OSError as exc:
        result = CommandResult(
            cmd=command,
            cwd=cwd,
            returncode=127,
            timed_out=False,
            duration_s=time.perf_counter() - start,
            stdout="",
            stderr=str(exc),
            error=str(exc),
        )

    if log_path is not None:
        _write_log(log_path, result)
    return result


def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _write_log(path: Path, result: CommandResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"cmd: {_format_command(result.cmd)}",
        f"cwd: {result.cwd}",
        f"returncode: {result.returncode}",
        f"timed_out: {result.timed_out}",
        f"duration_s: {result.duration_s:.6f}",
    ]
    if result.error:
        lines.extend(["", "error:", result.error])
    lines.extend(["", "stdout:", result.stdout, "", "stderr:", result.stderr])
    path.write_text("\n".join(lines).rstrip() + "\n")


def _format_command(cmd: Sequence[str]) -> str:
    return " ".join(cmd)
