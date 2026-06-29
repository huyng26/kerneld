from __future__ import annotations

from pathlib import Path

from kerneld.backends.triton import TritonBackend
from kerneld.ops.rmsnorm.bench import benchmark_candidate_fn
from kerneld.pipeline.candidates import load_candidate_info
from kerneld.run_state import RunState
from kerneld.schemas import MicrobenchResult, RMSNormOpSpec


def microbench_run(
    run_dir: Path,
    *,
    candidate_id: str,
    warmup_iters: int = 20,
    measured_iters: int = 100,
) -> MicrobenchResult:
    state = RunState.load(run_dir)
    spec_payload = state.read_json("op_spec.json")
    if spec_payload.get("op_type") != "rmsnorm":
        result = MicrobenchResult(
            candidate_id=candidate_id,
            passed=False,
            error=f"unsupported op spec type: {spec_payload.get('op_type')!r}",
        )
        state.write_json(f"microbench/{candidate_id}.json", result)
        return result
    spec = RMSNormOpSpec.model_validate(spec_payload)
    candidate = load_candidate_info(state, candidate_id)
    try:
        if candidate.backend != "triton":
            raise ValueError(f"unsupported backend for v1 microbench: {candidate.backend!r}")
        kernel_fn = TritonBackend().load_entrypoint(candidate)
        result = benchmark_candidate_fn(
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
