from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .schemas import RunConfig

_CANDIDATE_RE = re.compile(r"candidate_(\d{3,})$")


class RunState:
    """Filesystem view for one artifact-driven kerneld run."""

    def __init__(self, run_dir: Path, config: RunConfig | None = None) -> None:
        self.run_dir = run_dir.resolve()
        self.config = config

    @classmethod
    def create(cls, workspace: Path, config: RunConfig) -> "RunState":
        run_dir = config.run_dir
        if not run_dir.is_absolute():
            run_dir = workspace / run_dir
        run_dir = run_dir.resolve()
        normalized = config.model_copy(update={"run_dir": run_dir})
        state = cls(run_dir=run_dir, config=normalized)
        state.ensure_layout()
        state.write_json("config.json", normalized)
        return state

    @classmethod
    def load(cls, run_dir: Path) -> "RunState":
        run_dir = run_dir.resolve()
        config_path = run_dir / "config.json"
        config = None
        if config_path.exists():
            config = RunConfig.model_validate_json(config_path.read_text())
        return cls(run_dir=run_dir, config=config)

    def ensure_layout(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("candidates", "verification", "microbench", "modelbench"):
            (self.run_dir / name).mkdir(exist_ok=True)

    def path(self, *parts: str) -> Path:
        return self.run_dir.joinpath(*parts)

    def write_json(self, name: str, model: BaseModel | dict[str, Any]) -> Path:
        output_path = self.path(name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(model, BaseModel):
            payload = model.model_dump(mode="json")
        else:
            payload = model
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return output_path

    def read_json(self, name: str) -> dict[str, Any]:
        with self.path(name).open() as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError(f"JSON artifact {name!r} must contain an object")
        return payload

    def candidate_path(self, candidate_id: str) -> Path:
        return self.path("candidates", f"{candidate_id}.py")

    def current_candidate_path(self) -> Path:
        return self.path("candidates", "current.py")

    def allocate_candidate_id(self) -> str:
        existing = []
        candidates_dir = self.path("candidates")
        if candidates_dir.exists():
            for child in candidates_dir.iterdir():
                if child.suffix != ".py":
                    continue
                match = _CANDIDATE_RE.match(child.stem)
                if match:
                    existing.append(int(match.group(1)))
        next_index = max(existing, default=-1) + 1
        return f"candidate_{next_index:03d}"

    def artifact_path(self, kind: str, candidate_id: str | None = None) -> Path:
        if kind == "plan":
            return self.path("plan.json")
        if kind == "op_spec":
            return self.path("op_spec.json")
        if kind == "selection":
            return self.path("selection.json")
        if kind == "final_kernel":
            return self.path("final_kernel.py")
        if kind == "report":
            return self.path("report.md")
        if kind in {"verification", "microbench", "modelbench"}:
            if candidate_id is None:
                raise ValueError(f"candidate_id is required for {kind} artifacts")
            return self.path(kind, f"{candidate_id}.json")
        raise ValueError(f"unknown artifact kind: {kind}")
