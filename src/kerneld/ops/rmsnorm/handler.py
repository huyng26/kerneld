from __future__ import annotations

from typing import Any

from kerneld.ops.registry import OpHandler
from kerneld.ops.rmsnorm.bench import benchmark_candidate_fn
from kerneld.ops.rmsnorm.generator import generate_rmsnorm_triton_candidates
from kerneld.ops.rmsnorm.patchers import patch_model
from kerneld.ops.rmsnorm.verify import verify_candidate_fn
from kerneld.schemas import RMSNormOpSpec


def rmsnorm_patch_metadata(spec: RMSNormOpSpec, patch_result: Any) -> dict[str, Any]:
    return {
        "module_path": spec.module_path,
        "patch_scope": patch_result.patch_scope,
        "patched_module_paths": patch_result.patched_module_paths,
        "num_patched_modules": len(patch_result.patched_module_paths),
        "skipped_modules": [
            {"module_path": decision.module_path, "reason": decision.reason}
            for decision in patch_result.skipped_modules
        ],
    }


RMSNORM_HANDLER = OpHandler(
    op_type="rmsnorm",
    spec_model=RMSNormOpSpec,
    generate_candidates=generate_rmsnorm_triton_candidates,
    verify_kernel=verify_candidate_fn,
    benchmark_kernel=benchmark_candidate_fn,
    patch_model=patch_model,
    patch_metadata=rmsnorm_patch_metadata,
)
