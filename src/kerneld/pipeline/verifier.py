from __future__ import annotations

from pathlib import Path
from kerneld.backends.triton import TritonBackend
from kerneld.ops.rmsnorm.verify import verify_candidate_fn
from kerneld.pipeline.candidates import load_candidate_info
from kerneld.run_state import RunState
from kerneld.schemas import CandidateInfo, RMSNormOpSpec, VerificationResult


def verify_run(run_dir: Path, *, candidate_id: str) -> VerificationResult:
    state = RunState.load(run_dir)
    spec_payload = state.read_json("op_spec.json")
    if spec_payload.get("op_type") != "rmsnorm":
        result = VerificationResult(
            candidate_id=candidate_id,
            passed=False,
            error=f"unsupported op spec type: {spec_payload.get('op_type')!r}",
        )
        state.write_json(f"verification/{candidate_id}.json", result)
        return result
    spec = RMSNormOpSpec.model_validate(spec_payload)
    candidate = load_candidate_info(state, candidate_id)
    result = verify_candidate(candidate, spec)
    state.write_json(f"verification/{candidate_id}.json", result)
    return result


def verify_candidate(candidate: CandidateInfo, spec: RMSNormOpSpec) -> VerificationResult:
    try:
        if candidate.backend != "triton":
            return VerificationResult(
                candidate_id=candidate.candidate_id,
                passed=False,
                error=f"unsupported backend for v1 verifier: {candidate.backend!r}",
            )
        kernel_fn = TritonBackend().load_entrypoint(candidate)
        return verify_candidate_fn(candidate_id=candidate.candidate_id, kernel_fn=kernel_fn, spec=spec)
    except Exception as exc:
        return VerificationResult(candidate_id=candidate.candidate_id, passed=False, error=str(exc))
