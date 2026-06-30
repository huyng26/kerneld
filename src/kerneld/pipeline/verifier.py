from __future__ import annotations

from pathlib import Path

from kerneld.backends.triton import TritonBackend
from kerneld.ops.registry import OpHandler, validate_op_spec
from kerneld.pipeline.candidates import load_candidate_info
from kerneld.run_state import RunState
from kerneld.schemas import CandidateInfo, OpSpec, VerificationResult


def verify_run(run_dir: Path, *, candidate_id: str) -> VerificationResult:
    state = RunState.load(run_dir)
    spec_payload = state.read_json("op_spec.json")
    try:
        handler, spec = validate_op_spec(spec_payload)
    except Exception as exc:
        result = VerificationResult(
            candidate_id=candidate_id,
            passed=False,
            error=str(exc),
        )
        state.write_json(f"verification/{candidate_id}.json", result)
        return result
    candidate = load_candidate_info(state, candidate_id)
    result = verify_candidate(candidate, spec, handler=handler)
    state.write_json(f"verification/{candidate_id}.json", result)
    return result


def verify_candidate(candidate: CandidateInfo, spec: OpSpec, *, handler: OpHandler | None = None) -> VerificationResult:
    try:
        if handler is None:
            handler, spec = validate_op_spec(spec.model_dump(mode="json"))
        if handler.verify_kernel is None:
            raise ValueError(f"op type {handler.op_type!r} does not support verification")
        if candidate.backend != "triton":
            return VerificationResult(
                candidate_id=candidate.candidate_id,
                passed=False,
                error=f"unsupported backend for v1 verifier: {candidate.backend!r}",
            )
        kernel_fn = TritonBackend().load_entrypoint(candidate)
        return handler.verify_kernel(candidate_id=candidate.candidate_id, kernel_fn=kernel_fn, spec=spec)
    except Exception as exc:
        return VerificationResult(candidate_id=candidate.candidate_id, passed=False, error=str(exc))
