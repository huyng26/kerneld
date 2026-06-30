from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from kerneld.pipeline.planner import SUPPORTED_OPS, create_plan
from kerneld.run_state import RunState
from kerneld.schemas import RunConfig


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kerneld", description="Kernel optimization run tools")
    parser.add_argument("--version", action="version", version="kerneld 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_run = subparsers.add_parser("init-run", help="create a new artifact run directory")
    init_run.add_argument("--workspace", type=Path, default=Path("workspace/runs"))
    init_run.add_argument("--run-id", default=None, help="run id; defaults to a deterministic model/op label")
    init_run.add_argument("--model", "--model-id", dest="model_id", required=True)
    init_run.add_argument("--op", required=True, choices=sorted(SUPPORTED_OPS))
    init_run.add_argument("--input-shape", required=True, help="comma-separated shape, for example 1,1024")
    init_run.add_argument("--dtype", default="float16")
    init_run.add_argument("--device", default="cuda")
    init_run.add_argument("--max-candidates", type=int, default=4)
    init_run.add_argument("--min-model-speedup-pct", type=float, default=0.0)
    init_run.add_argument("--agent-enabled", action="store_true")
    init_run.set_defaults(func=cmd_init_run)

    extract = subparsers.add_parser("extract", help="extract the target op spec for a run")
    extract.add_argument("--run", type=Path, required=True, help="run directory")
    extract.add_argument("--module-path", default=None, help="optional exact RMSNorm module path")
    extract.set_defaults(func=cmd_extract)

    generate = subparsers.add_parser("generate", help="generate candidate kernels for a run")
    generate.add_argument("--run", type=Path, required=True, help="run directory")
    generate.set_defaults(func=cmd_generate)

    verify = subparsers.add_parser("verify", help="verify a generated candidate")
    verify.add_argument("--run", type=Path, required=True, help="run directory")
    verify.add_argument("--candidate", required=True, help="candidate id, for example candidate_000")
    verify.set_defaults(func=cmd_verify)

    microbench = subparsers.add_parser("microbench", help="benchmark a candidate RMSNorm op in isolation")
    microbench.add_argument("--run", type=Path, required=True, help="run directory")
    microbench.add_argument("--candidate", required=True, help="candidate id, for example candidate_000")
    microbench.add_argument("--warmup-iters", type=int, default=20)
    microbench.add_argument("--measured-iters", type=int, default=100)
    microbench.set_defaults(func=cmd_microbench)

    modelbench = subparsers.add_parser("modelbench", help="benchmark a candidate inside the real model path")
    modelbench.add_argument("--run", type=Path, required=True, help="run directory")
    modelbench.add_argument("--candidate", required=True, help="candidate id, for example candidate_000")
    modelbench.add_argument("--warmup-iters", type=int, default=5)
    modelbench.add_argument("--measured-iters", type=int, default=20)
    modelbench.add_argument("--input-seed", type=int, default=0, help="deterministic modelbench input token seed")
    modelbench.set_defaults(func=cmd_modelbench)

    select = subparsers.add_parser("select", help="select the best accepted candidate")
    select.add_argument("--run", type=Path, required=True, help="run directory")
    select.set_defaults(func=cmd_select)

    report = subparsers.add_parser("report", help="write a run report")
    report.add_argument("--run", type=Path, required=True, help="run directory")
    report.set_defaults(func=cmd_report)

    agent_prepare = subparsers.add_parser("agent-prepare", help="prepare agent-loop artifacts for a run")
    agent_prepare.add_argument("--run", type=Path, required=True, help="run directory")
    agent_prepare.add_argument("--candidate", default=None, help="candidate id to seed candidates/current.py")
    agent_prepare.add_argument("--max-attempts", type=int, default=50, help="manual agent trial cap")
    agent_prepare.set_defaults(func=cmd_agent_prepare)

    agent_step = subparsers.add_parser("agent-step", help="snapshot and evaluate candidates/current.py once")
    agent_step.add_argument("--run", type=Path, required=True, help="run directory")
    agent_step.add_argument("--microbench-warmup-iters", type=int, default=20)
    agent_step.add_argument("--microbench-measured-iters", type=int, default=100)
    agent_step.add_argument("--modelbench-warmup-iters", type=int, default=5)
    agent_step.add_argument("--modelbench-measured-iters", type=int, default=20)
    agent_step.set_defaults(func=cmd_agent_step)

    agent_loop = subparsers.add_parser("agent-loop", help="run the optional coding-agent evaluation loop")
    agent_loop.add_argument("--run", type=Path, required=True, help="run directory")
    agent_loop.add_argument("--candidate", default=None, help="candidate id to seed candidates/current.py")
    agent_loop.add_argument("--max-attempts", type=int, default=1)
    agent_loop.add_argument("--timeout-s", type=float, default=3600.0)
    agent_loop.add_argument("--agent", choices=["codex"], default=None, help="built-in agent provider")
    agent_loop.add_argument("--agent-model", default=None, help="model to pass to the built-in agent provider")
    agent_loop.add_argument(
        "--agent-command",
        nargs=argparse.REMAINDER,
        default=None,
        help="external agent command argv; must be the final option when used",
    )
    agent_loop.set_defaults(func=cmd_agent_loop)

    return parser


def cmd_init_run(args: argparse.Namespace) -> int:
    run_id = args.run_id or _default_run_id(args.model_id, args.op)
    workspace = args.workspace
    config = RunConfig(
        run_id=run_id,
        run_dir=Path(run_id),
        model_id=args.model_id,
        op=args.op,
        input_shape=args.input_shape,
        dtype=args.dtype,
        device=args.device,
        max_candidates=args.max_candidates,
        min_model_speedup_pct=args.min_model_speedup_pct,
    )
    state = RunState.create(workspace=workspace, config=config)
    plan = create_plan(config, agent_enabled=args.agent_enabled)
    state.write_json("plan.json", plan)
    print(state.run_dir)
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    from kerneld.pipeline.extractor import extract_run

    spec = extract_run(args.run, module_path=args.module_path)
    print(f"wrote {Path(args.run).resolve() / 'op_spec.json'} for {spec.module_path}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from kerneld.pipeline.generator import generate_run

    candidates = generate_run(args.run)
    print(f"generated {len(candidates)} candidate(s) in {Path(args.run).resolve() / 'candidates'}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from kerneld.pipeline.verifier import verify_run

    result = verify_run(args.run, candidate_id=args.candidate)
    print(f"{args.candidate}: {'passed' if result.passed else 'failed'}")
    if result.error:
        print(result.error)
    return 0 if result.passed else 1


def cmd_microbench(args: argparse.Namespace) -> int:
    from kerneld.pipeline.microbench import microbench_run

    result = microbench_run(
        args.run,
        candidate_id=args.candidate,
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
    )
    print(f"{args.candidate}: {'passed' if result.passed else 'failed'}")
    if result.speedup_pct is not None:
        print(f"microbench speedup: {result.speedup_pct:.2f}%")
    if result.error:
        print(result.error)
    return 0 if result.passed else 1


def cmd_modelbench(args: argparse.Namespace) -> int:
    from kerneld.pipeline.modelbench import modelbench_run

    result = modelbench_run(
        args.run,
        candidate_id=args.candidate,
        warmup_iters=args.warmup_iters,
        measured_iters=args.measured_iters,
        input_seed=args.input_seed,
    )
    print(f"{args.candidate}: {'passed' if result.passed else 'failed'}")
    if result.speedup_pct is not None:
        print(f"modelbench speedup: {result.speedup_pct:.2f}%")
    if result.error:
        print(result.error)
    return 0 if result.passed else 1


def cmd_select(args: argparse.Namespace) -> int:
    from kerneld.pipeline.selector import select_run

    result = select_run(args.run)
    print(f"selection: {'accepted' if result.accepted else 'rejected'}")
    if result.candidate_id:
        print(f"candidate: {result.candidate_id}")
    print(result.reason)
    return 0 if result.accepted else 1


def cmd_report(args: argparse.Namespace) -> int:
    from kerneld.pipeline.report import write_report

    report_path = write_report(args.run)
    print(report_path)
    return 0


def cmd_agent_prepare(args: argparse.Namespace) -> int:
    from kerneld.runners.agent_loop import prepare_agent_run

    result = prepare_agent_run(args.run, candidate_id=args.candidate, max_attempts=args.max_attempts)
    print(f"program: {result.program_path}")
    print(f"task: {result.task_path}")
    print(f"editable: {result.editable_candidate_path}")
    return 0


def cmd_agent_step(args: argparse.Namespace) -> int:
    from kerneld.runners.agent_loop import agent_step

    result = agent_step(
        args.run,
        microbench_warmup_iters=args.microbench_warmup_iters,
        microbench_measured_iters=args.microbench_measured_iters,
        modelbench_warmup_iters=args.modelbench_warmup_iters,
        modelbench_measured_iters=args.modelbench_measured_iters,
    )
    print(f"candidate: {result.candidate_id or 'none'}")
    print(f"feedback: {result.feedback_path}")
    if result.skipped_after in {"scope", "no_edit"}:
        print("verification: not run")
    else:
        print(f"verification: {'passed' if result.verification_passed else 'failed'}")
    if result.microbench_passed is not None:
        print(f"microbench: {'passed' if result.microbench_passed else 'failed'}")
    if result.modelbench_passed is not None:
        print(f"modelbench: {'passed' if result.modelbench_passed else 'failed'}")
    if result.selected_candidate_id:
        print(f"selected: {result.selected_candidate_id}")
    if result.skipped_after:
        print(f"skipped after: {result.skipped_after}")
    if result.scope_violations:
        print("scope violations:")
        for violation in result.scope_violations:
            print(f"- {violation}")
    if result.skipped_after in {"scope", "no_edit"}:
        return 1
    return 0 if result.verification_passed else 1


def cmd_agent_loop(args: argparse.Namespace) -> int:
    from kerneld.runners.agent_loop import run_agent_loop

    agent_command = args.agent_command or None
    if args.agent and agent_command:
        print("error: --agent and --agent-command are mutually exclusive", file=sys.stderr)
        return 2
    if args.agent_model and args.agent != "codex":
        print("error: --agent-model requires --agent codex", file=sys.stderr)
        return 2
    result = run_agent_loop(
        args.run,
        candidate_id=args.candidate,
        max_attempts=args.max_attempts,
        agent_command=agent_command,
        agent_provider=args.agent,
        agent_model=args.agent_model,
        timeout_s=args.timeout_s,
    )
    print(f"stop: {result.stop_reason}")
    for attempt in result.attempts:
        candidate_id = attempt.get("candidate_id") or "none"
        print(f"attempt {attempt['attempt_index']}: {candidate_id}")
    if result.agent_provider:
        print(f"agent: {result.agent_provider}")
    if result.agent_session_id:
        print(f"agent session: {result.agent_session_id}")
    if result.selected_candidate_id:
        print(f"selected: {result.selected_candidate_id}")
    return 0


def _default_run_id(model_id: str, op: str) -> str:
    label = model_id.lower().replace("/", "-").replace("_", "-")
    allowed = []
    previous_dash = False
    for char in label:
        if char.isalnum():
            allowed.append(char)
            previous_dash = False
        elif not previous_dash:
            allowed.append("-")
            previous_dash = True
    normalized = "".join(allowed).strip("-") or "model"
    return f"{op}-{normalized}"


if __name__ == "__main__":
    raise SystemExit(main())
