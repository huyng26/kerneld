from __future__ import annotations

from pathlib import Path

from kerneld.backends.triton import TritonBackend
from kerneld.ops.registry import validate_op_spec
from kerneld.pipeline.candidates import load_candidate_info
from kerneld.run_state import RunState
from kerneld.schemas import MicrobenchResult


def microbench_run(
    run_dir: Path,
    *,
    candidate_id: str,
    warmup_iters: int = 20,
    measured_iters: int = 100,
) -> MicrobenchResult:
    state = RunState.load(run_dir)
    spec_payload = state.read_json("op_spec.json")
    try:
        handler, spec = validate_op_spec(spec_payload)
    except Exception as exc:
        result = MicrobenchResult(
            candidate_id=candidate_id,
            passed=False,
            error=str(exc),
        )
        state.write_json(f"microbench/{candidate_id}.json", result)
        return result
    candidate = load_candidate_info(state, candidate_id)
    try:
        if handler.benchmark_kernel is None:
            raise ValueError(f"op type {handler.op_type!r} does not support microbenchmarking")
        if candidate.backend != "triton":
            raise ValueError(f"unsupported backend for v1 microbench: {candidate.backend!r}")
        kernel_fn = TritonBackend().load_entrypoint(candidate)
        result = handler.benchmark_kernel(
            candidate_id=candidate_id,
            kernel_fn=kernel_fn,
            spec=spec,
            warmup_iters=warmup_iters,
            measured_iters=measured_iters,
        )
    except Exception as exc:
        result = MicrobenchResult(candidate_id=candidate_id, passed=False, error=str(exc))
    state.write_json(f"microbench/{candidate_id}.json", result)
    return result
