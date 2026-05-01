from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import __version__
from .policy import make_exploration_policy, write_policy
from .vllm import (
    PATCH_PATH,
    apply_patch,
    check_patch,
    env_for_policy,
    exec_vllm,
    require_vllm,
    shell_exports,
)


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def cmd_doctor(args: argparse.Namespace) -> int:
    ok_vllm, vllm_detail = require_vllm()
    payload = {
        "graphadmit": __version__,
        "python": sys.version.split()[0],
        "vllm": {"available": ok_vllm, "detail": vllm_detail},
        "patch": str(PATCH_PATH),
        "torch": {"available": False, "detail": "not checked"},
    }
    try:
        import torch  # type: ignore

        payload["torch"] = {
            "available": True,
            "detail": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except Exception as exc:  # pragma: no cover - depends on optional deps
        payload["torch"] = {"available": False, "detail": str(exc)}
    _print_json(payload)
    return 0


def cmd_make_policy(args: argparse.Namespace) -> int:
    policy = make_exploration_policy(
        bucket_preset=args.bucket_preset,
        max_tokens=args.max_tokens,
        base_capture_size=args.base_capture_size,
        min_tokens=args.min_tokens,
        default_action=args.default_action,
        graph_action=args.graph_action,
        max_extra_templates=args.max_extra_templates,
        live_admission=not args.no_live_admission,
        live_min_samples=args.live_min_samples,
        live_min_useful_rate=args.live_min_useful_rate,
        live_min_saving_ms=args.live_min_saving_ms,
        live_max_p95_regression_ms=args.live_max_p95_regression_ms,
        live_capture=args.live_capture,
    )
    output = write_policy(policy, args.output)
    runtime_policy = policy["runtime_policy"]
    _print_json(
        {
            "output": str(output),
            "kind": runtime_policy["kind"],
            "extra_capture_sizes": runtime_policy["residual_capture"]["extra_capture_sizes"],
            "rules": len(runtime_policy["rules"]),
            "live_admission": runtime_policy["live_admission"],
            "live_capture": runtime_policy["live_capture"],
        }
    )
    return 0


def cmd_vllm_patch(args: argparse.Namespace) -> int:
    status = apply_patch(args.target, args.patch) if args.apply else check_patch(args.target, args.patch)
    _print_json(status.to_json())
    return 0 if status.status in {"clean", "applied"} else 2


def cmd_vllm_env(args: argparse.Namespace) -> int:
    env = env_for_policy(
        args.policy,
        observations=args.observations,
        active=not args.inactive,
        live_admission=args.live_admission,
        shadow_baseline=not args.no_shadow_baseline,
        fixed_metadata_arena=not args.no_fixed_metadata_arena,
        template_scheduler=args.template_scheduler,
        live_capture=args.live_capture,
    )
    if args.json:
        _print_json(env)
    else:
        print(shell_exports(env))
    return 0


def cmd_vllm_serve(args: argparse.Namespace) -> int:
    env = env_for_policy(
        args.policy,
        observations=args.observations,
        template_scheduler=args.template_scheduler,
        live_capture=args.live_capture,
    )
    vllm_args = list(args.vllm_args)
    if vllm_args and vllm_args[0] == "--":
        vllm_args = vllm_args[1:]
    return int(exec_vllm(["serve", *vllm_args], env))


def cmd_bench_torch(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.compositional_cuda_graph_microbench",
        "--output",
        args.output,
        "--repeat",
        str(args.repeat),
        "--hidden",
        str(args.hidden),
        "--inter",
        str(args.inter),
        "--layers",
        str(args.layers),
        "--dtype",
        args.dtype,
    ]
    return subprocess.call(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphadmit",
        description="Online CUDA Graph admission tools for dynamic LLM serving.",
    )
    parser.add_argument("--version", action="version", version=f"graphadmit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check optional runtime dependencies")
    doctor.set_defaults(func=cmd_doctor)

    policy = sub.add_parser("make-policy", help="emit a fail-closed online exploration policy")
    policy.add_argument("-o", "--output", required=True)
    policy.add_argument("--bucket-preset", default="sglang-pcg", choices=["default", "sglang-pcg", "default+sglang-pcg"])
    policy.add_argument("--max-tokens", type=int, default=4096)
    policy.add_argument("--base-capture-size", type=int, default=512)
    policy.add_argument("--min-tokens", type=int, default=0)
    policy.add_argument("--default-action", default="cp")
    policy.add_argument("--graph-action", default="ours_cp")
    policy.add_argument("--max-extra-templates", type=int, default=0)
    policy.add_argument("--no-live-admission", action="store_true")
    policy.add_argument("--live-min-samples", type=int, default=2)
    policy.add_argument("--live-min-useful-rate", type=float, default=0.67)
    policy.add_argument("--live-min-saving-ms", type=float, default=0.5)
    policy.add_argument("--live-max-p95-regression-ms", type=float, default=5.0)
    policy.add_argument(
        "--live-capture",
        action="store_true",
        help=(
            "enable same-engine capture machinery; unvalidated templates still "
            "fallback unless trusted live graph-replay observations admit them"
        ),
    )
    policy.set_defaults(func=cmd_make_policy)

    vllm_patch = sub.add_parser("vllm-patch", help="check or apply the vLLM/FlowPrefill patch")
    vllm_patch.add_argument("--target", required=True, help="path to a vLLM or FlowPrefill checkout")
    vllm_patch.add_argument("--patch", default=None, help="override patch path")
    vllm_patch.add_argument("--apply", action="store_true", help="apply the patch if it is clean")
    vllm_patch.set_defaults(func=cmd_vllm_patch)

    vllm_env = sub.add_parser("vllm-env", help="print environment variables for patched vLLM")
    vllm_env.add_argument("--policy", required=True)
    vllm_env.add_argument("--observations", default=None)
    vllm_env.add_argument("--inactive", action="store_true")
    vllm_env.add_argument("--live-admission", action=argparse.BooleanOptionalAction, default=None)
    vllm_env.add_argument("--no-shadow-baseline", action="store_true")
    vllm_env.add_argument("--no-fixed-metadata-arena", action="store_true")
    vllm_env.add_argument("--template-scheduler", action="store_true")
    vllm_env.add_argument("--live-capture", action=argparse.BooleanOptionalAction, default=None)
    vllm_env.add_argument("--json", action="store_true")
    vllm_env.set_defaults(func=cmd_vllm_env)

    vllm_serve = sub.add_parser("vllm-serve", help="run `vllm serve` with GraphAdmit env vars")
    vllm_serve.add_argument("--policy", required=True)
    vllm_serve.add_argument("--observations", default=None)
    vllm_serve.add_argument("--template-scheduler", action="store_true")
    vllm_serve.add_argument("--live-capture", action=argparse.BooleanOptionalAction, default=None)
    vllm_serve.add_argument("vllm_args", nargs=argparse.REMAINDER)
    vllm_serve.set_defaults(func=cmd_vllm_serve)

    bench = sub.add_parser("bench", help="run packaged microbenchmarks")
    bench_sub = bench.add_subparsers(dest="bench_command", required=True)
    torch_bench = bench_sub.add_parser("torch", help="run compositional torch CUDA Graph microbench")
    torch_bench.add_argument("-o", "--output", default="results/compositional_cuda_graph_microbench.json")
    torch_bench.add_argument("--repeat", type=int, default=30)
    torch_bench.add_argument("--hidden", type=int, default=2048)
    torch_bench.add_argument("--inter", type=int, default=8192)
    torch_bench.add_argument("--layers", type=int, default=4)
    torch_bench.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    torch_bench.set_defaults(func=cmd_bench_torch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
