from __future__ import annotations

from kerneld.run_state import RunState
from kerneld.schemas import CandidateInfo


def load_candidate_info(state: RunState, candidate_id: str) -> CandidateInfo:
    candidates_path = state.path("candidates.json")
    if candidates_path.exists():
        payload = state.read_json("candidates.json")
        for item in payload.get("candidates", []):
            if item.get("candidate_id") == candidate_id:
                return CandidateInfo.model_validate(item)
    source_path = state.candidate_path(candidate_id)
    return CandidateInfo(
        candidate_id=candidate_id,
        backend="triton",
        entrypoint="kernel_fn",
        source_files=[source_path],
        build_required=False,
    )


def list_candidate_ids(state: RunState) -> list[str]:
    candidates_path = state.path("candidates.json")
    if candidates_path.exists():
        payload = state.read_json("candidates.json")
        return [item["candidate_id"] for item in payload.get("candidates", []) if "candidate_id" in item]
    candidates_dir = state.path("candidates")
    if not candidates_dir.exists():
        return []
    return sorted(path.stem for path in candidates_dir.glob("candidate_*.py"))
