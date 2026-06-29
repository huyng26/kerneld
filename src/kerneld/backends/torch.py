from __future__ import annotations

from kerneld.backends.base import BackendError
from kerneld.schemas import CandidateInfo


class TorchBackend:
    name = "torch"
    build_required = False

    def render_candidate(self, *args, **kwargs):
        raise NotImplementedError("Torch baseline rendering is not part of MILESTONE_0")

    def build_candidate(self, candidate: CandidateInfo) -> CandidateInfo:
        if candidate.backend != self.name:
            raise BackendError(f"candidate {candidate.candidate_id} is not a Torch candidate")
        return candidate

    def load_entrypoint(self, candidate: CandidateInfo):
        raise NotImplementedError("Torch baseline entrypoint loading is not part of MILESTONE_0")
