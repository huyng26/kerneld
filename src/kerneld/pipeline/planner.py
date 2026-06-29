from __future__ import annotations

from kerneld.schemas import Plan, RunConfig

SUPPORTED_OPS = {"rmsnorm"}


def create_plan(config: RunConfig, *, agent_enabled: bool = False) -> Plan:
    if config.op not in SUPPORTED_OPS:
        supported = ", ".join(sorted(SUPPORTED_OPS))
        raise ValueError(f"unsupported op {config.op!r}; supported ops: {supported}")
    return Plan(
        run_id=config.run_id,
        model_id=config.model_id,
        op=config.op,
        input_shape=config.input_shape,
        dtype=config.dtype,
        device=config.device,
        max_candidates=config.max_candidates,
        min_model_speedup_pct=config.min_model_speedup_pct,
        agent_enabled=agent_enabled,
    )
