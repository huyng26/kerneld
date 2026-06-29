from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from kerneld.schemas import BackendName, CandidateInfo, OpSpec


@runtime_checkable
class Backend(Protocol):
    name: BackendName
    build_required: bool

    def render_candidate(
        self,
        spec: OpSpec,
        params: dict[str, Any],
        output_dir: Path,
    ) -> CandidateInfo:
        ...

    def build_candidate(self, candidate: CandidateInfo) -> CandidateInfo:
        ...

    def load_entrypoint(self, candidate: CandidateInfo) -> Callable[..., Any]:
        ...


class BackendError(RuntimeError):
    """Raised when a backend cannot prepare or load a candidate."""
