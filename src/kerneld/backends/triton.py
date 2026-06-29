from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from kerneld.backends.base import BackendError
from kerneld.schemas import CandidateInfo, OpSpec


class TritonBackend:
    name = "triton"
    build_required = False

    def render_candidate(
        self,
        spec: OpSpec,
        params: dict[str, Any],
        output_dir: Path,
    ) -> CandidateInfo:
        raise NotImplementedError("Triton template rendering is implemented in the generator milestone")

    def build_candidate(self, candidate: CandidateInfo) -> CandidateInfo:
        if candidate.backend != self.name:
            raise BackendError(f"candidate {candidate.candidate_id} is not a Triton candidate")
        return candidate.model_copy(update={"build_required": False, "build_artifact": None})

    def load_entrypoint(self, candidate: CandidateInfo) -> Callable[..., Any]:
        if candidate.backend != self.name:
            raise BackendError(f"candidate {candidate.candidate_id} is not a Triton candidate")
        if not candidate.source_files:
            raise BackendError(f"candidate {candidate.candidate_id} has no source files")
        module_path = candidate.source_files[0]
        module = _load_module(module_path, f"kerneld_candidate_{candidate.candidate_id}")
        try:
            entrypoint = getattr(module, candidate.entrypoint)
        except AttributeError as exc:
            raise BackendError(
                f"candidate {candidate.candidate_id} does not define {candidate.entrypoint!r}"
            ) from exc
        if not callable(entrypoint):
            raise BackendError(f"candidate entrypoint {candidate.entrypoint!r} is not callable")
        return entrypoint


def _load_module(path: Path, module_name: str) -> ModuleType:
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise BackendError(f"could not load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module
