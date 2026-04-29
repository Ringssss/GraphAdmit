#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prefill_graph.runtime import (  # noqa: E402
    DEFAULT_RESIDUAL_BUCKETS,
    ResidualCaptureObservation,
    ResidualCapturePlanner,
    policy_graph_covers,
    residual_buckets_for_preset,
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def workload_token_lengths(path: str | Path) -> list[int]:
    data = load_json(path)
    lengths: list[int] = []
    for idx, req in enumerate(data.get("requests", [])):
        if not isinstance(req, dict):
            continue
        raw = req.get("actual_input_length", req.get("tokens", req.get("tok")))
        if raw is None:
            continue
        try:
            lengths.append(int(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid token length in request {idx}: {raw!r}") from exc
    return lengths


def get_result(data: dict[str, Any], contains: str) -> dict[str, Any]:
    for result in data.get("results", []):
        if contains.lower() in str(result.get("config", "")).lower():
            return result
    raise KeyError(f"no result config contains {contains!r}")


def correctness_flags(candidate: dict[str, Any], n: int) -> list[bool]:
    values = candidate.get("same_outputs_vs_reference")
    if isinstance(values, list) and len(values) >= n:
        return [bool(v) for v in values[:n]]
    if candidate.get("all_same_outputs_vs_reference") is not None:
        return [bool(candidate["all_same_outputs_vs_reference"])] * n
    values = candidate.get("same_outputs_vs_first")
    if isinstance(values, list) and len(values) >= n:
        return [bool(v) for v in values[:n]]
    if candidate.get("all_same_outputs_vs_first") is not None:
        return [bool(candidate["all_same_outputs_vs_first"])] * n
    raise ValueError(
        "candidate result has no explicit correctness flags; refusing to build "
        "a residual-capture policy from unvalidated outputs"
    )


def parse_int_list(raw: str | None, default: list[int] | None = None) -> list[int]:
    if raw is None or raw.strip() == "":
        return list(default or [])
    return sorted({int(item.strip()) for item in raw.split(",") if item.strip()})


def resolve_buckets(
    *,
    raw: str | None,
    preset: str,
    max_tokens: int,
) -> list[int]:
    if raw is not None and raw.strip():
        return parse_int_list(raw, DEFAULT_RESIDUAL_BUCKETS)
    return residual_buckets_for_preset(preset, max_tokens=max_tokens)


def rows_from_e2e(
    data: dict[str, Any],
    *,
    baseline_contains: str,
    candidate_contains: str,
    candidate_policy: dict[str, Any] | None = None,
    base_capture_size: int = 512,
) -> list[ResidualCaptureObservation]:
    baseline = get_result(data, baseline_contains)
    candidate = get_result(data, candidate_contains)
    base_rows = baseline.get("per_req", [])
    cand_rows = candidate.get("per_req", [])
    n = min(len(base_rows), len(cand_rows))
    correct = correctness_flags(candidate, n)
    rows: list[ResidualCaptureObservation] = []
    for idx in range(n):
        tokens = int(cand_rows[idx].get("tok", base_rows[idx].get("tok", 0)))
        graph_allowed = None
        matched_rule = None
        if candidate_policy is not None:
            graph_allowed, matched_rule = policy_graph_covers(
                candidate_policy,
                tokens,
                base_capture_size=base_capture_size,
            )
        candidate_template_tokens = None
        if graph_allowed and isinstance(matched_rule, dict):
            raw_template = matched_rule.get("template_tokens")
            if raw_template is not None:
                candidate_template_tokens = int(raw_template)
        rows.append(
            ResidualCaptureObservation(
                idx=idx,
                tokens=tokens,
                fallback_ms=float(base_rows[idx]["ms"]),
                graph_ms=float(cand_rows[idx]["ms"]),
                correct=bool(correct[idx]),
                metadata={
                    "baseline_config": baseline.get("config"),
                    "candidate_config": candidate.get("config"),
                    "candidate_graph_allowed": graph_allowed,
                    "candidate_template_tokens": candidate_template_tokens,
                },
            )
        )
    return rows


def make_exploration_policy(
    *,
    buckets: list[int],
    default_action: str,
    graph_action: str,
    base_capture_size: int,
    max_tokens: int,
    min_tokens: int = 0,
    max_extra_templates: int = 0,
    active_buckets: set[int] | None = None,
    live_enabled: bool = False,
    live_min_samples: int = 0,
    live_min_useful_rate: float = 0.0,
    live_min_saving_ms: float = 0.0,
    live_max_p95_regression_ms: float | None = None,
) -> dict[str, Any]:
    rules = []
    arena_ranges = []
    left = 0
    for bucket in sorted(buckets):
        if bucket <= base_capture_size:
            left = max(left, bucket)
            continue
        hi = min(int(bucket), int(max_tokens))
        if hi <= left:
            continue
        action = graph_action
        if min_tokens and hi <= min_tokens:
            action = default_action
        if active_buckets is not None and int(bucket) not in active_buckets:
            action = default_action
        rule = {
            "lo": int(left),
            "hi": int(hi),
            "action": action,
            "n": 0,
            "template_tokens": int(bucket),
            "reason": "residual exploration template; admit only for measurement",
        }
        rules.append(rule)
        if action == graph_action:
            arena_ranges.append({
                "lo": int(left),
                "hi": int(hi),
                "template_tokens": int(bucket),
                "action": action,
                "n": 0,
            })
        left = hi
        if left >= max_tokens:
            break
    if left < max_tokens:
        rules.append({
            "lo": int(left),
            "hi": int(max_tokens),
            "action": default_action,
            "n": 0,
            "reason": "outside residual exploration capture budget",
        })
    return {
        "runtime_policy": {
            "kind": "residual_capture_exploration_policy",
            "default_action": default_action,
            "baseline_action": default_action,
            "correctness_required": True,
            "rules": rules,
            "fixed_metadata_arena_ranges": arena_ranges,
            "single_engine_graph_actions": ["default", "ours", "cp", graph_action],
            "single_engine_fallback_actions": [
                "eager",
                "compile",
                "compiled",
                "fallback",
                "none",
            ],
            "single_engine_allow_multi_req_extra": True,
            "single_engine_requires_fixed_metadata_arena": True,
            "single_engine_max_extra_templates": max_extra_templates or max(1, len(arena_ranges)),
            "single_engine_min_rule_n": 0,
            "single_engine_base_capture_size": int(base_capture_size),
            "live_admission": {
                "enabled": bool(live_enabled),
                "explore_until_min_samples": True,
                "min_samples": int(live_min_samples),
                "min_useful_rate": float(live_min_useful_rate),
                "min_saving_ms": float(live_min_saving_ms),
                "max_p95_regression_ms": (
                    float(live_max_p95_regression_ms)
                    if live_max_p95_regression_ms is not None
                    else None
                ),
            },
            "residual_capture": {
                "mode": "broad_exploration",
                "template_buckets": buckets,
                "extra_capture_sizes": [
                    int(row["template_tokens"]) for row in arena_ranges
                ],
                "active_buckets": (
                    sorted(active_buckets) if active_buckets is not None else None
                ),
            },
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="E2E JSON containing baseline and exploration/current runtime results")
    parser.add_argument("--baseline-contains", default="vLLM graph max512 CP")
    parser.add_argument("--candidate-contains", default="Single-engine runtime")
    parser.add_argument("--seed-policy", default=None)
    parser.add_argument("--candidate-policy", default=None,
                        help="policy used by the measured candidate; when set, only actually graph-admitted rows can seed residual capture")
    parser.add_argument("--preserve-seed-rules", action="store_true",
                        help="preserve seed graph rules in learn_all mode; default is to rebuild graph windows from measured evidence")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["residual_only", "learn_all"], default="residual_only")
    parser.add_argument("--bucket-preset", default="default",
                        choices=["default", "sglang-pcg", "default+sglang-pcg"],
                        help="named template bucket preset used when --template-buckets is omitted")
    parser.add_argument("--template-buckets", default=None,
                        help="comma-separated explicit bucket list; overrides --bucket-preset")
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--min-useful-rate", type=float, default=0.75)
    parser.add_argument("--min-saving-ms", type=float, default=0.5)
    parser.add_argument("--min-p95-saving-ms", type=float, default=None,
                        help="optional bucket-local P95 saving required before admission")
    parser.add_argument("--max-p95-regression-ms", type=float, default=2.0)
    parser.add_argument("--max-regression-ms", type=float, default=5.0)
    parser.add_argument("--min-template-tokens", type=int, default=0,
                        help="ignore candidate templates below this token size")
    parser.add_argument("--allow-template-extrapolation", action="store_true",
                        help="allow samples from a different measured template to seed a candidate; off by default")
    parser.add_argument("--tail-token-threshold", type=int, default=0,
                        help="template/range token threshold where tail-specific admission thresholds apply")
    parser.add_argument("--tail-min-samples", type=int, default=None)
    parser.add_argument("--tail-min-useful-rate", type=float, default=None)
    parser.add_argument("--tail-min-saving-ms", type=float, default=None)
    parser.add_argument("--tail-min-p95-saving-ms", type=float, default=None)
    parser.add_argument("--tail-max-p95-regression-ms", type=float, default=None)
    parser.add_argument("--tail-max-regression-ms", type=float, default=None)
    parser.add_argument("--max-segments", type=int, default=4)
    parser.add_argument("--capture-ms-per-template", type=float, default=0.0)
    parser.add_argument("--warmup-ms-per-template", type=float, default=0.0)
    parser.add_argument("--amortization-replays", type=int, default=32)
    parser.add_argument("--default-action", default="cp")
    parser.add_argument("--graph-action", default="ours_cp")
    parser.add_argument("--base-capture-size", type=int, default=512)
    parser.add_argument("--allow-fallback-candidate-evidence", action="store_true",
                        help="allow rows where the candidate policy itself would have fallen back to seed residual capture")
    parser.add_argument("--make-exploration-policy", action="store_true")
    parser.add_argument("--workload", default=None,
                        help="optional workload JSON used with --demand-filter-policy")
    parser.add_argument("--demand-filter-policy", action="store_true",
                        help="when making an exploration policy, activate only buckets hit by --workload")
    parser.add_argument("--exploration-max-tokens", type=int, default=4096)
    parser.add_argument("--exploration-min-tokens", type=int, default=0)
    parser.add_argument("--exploration-max-extra-templates", type=int, default=0)
    parser.add_argument("--exploration-live-admission", action="store_true",
                        help="enable live admission directly in the emitted exploration policy")
    parser.add_argument("--exploration-live-min-samples", type=int, default=0)
    parser.add_argument("--exploration-live-min-useful-rate", type=float, default=0.0)
    parser.add_argument("--exploration-live-min-saving-ms", type=float, default=0.0)
    parser.add_argument("--exploration-live-max-p95-regression-ms", type=float, default=None)
    args = parser.parse_args()

    buckets = resolve_buckets(
        raw=args.template_buckets,
        preset=args.bucket_preset,
        max_tokens=args.exploration_max_tokens,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.make_exploration_policy:
        active_buckets = None
        if args.demand_filter_policy:
            if not args.workload:
                raise ValueError("--demand-filter-policy requires --workload")
            active_buckets = {
                int(template)
                for tokens in workload_token_lengths(args.workload)
                if int(tokens) > int(args.exploration_min_tokens)
                for template in [next((b for b in buckets if int(tokens) <= b), None)]
                if template is not None
            }
        payload = make_exploration_policy(
            buckets=buckets,
            default_action=args.default_action,
            graph_action=args.graph_action,
            base_capture_size=args.base_capture_size,
            max_tokens=args.exploration_max_tokens,
            min_tokens=args.exploration_min_tokens,
            max_extra_templates=args.exploration_max_extra_templates,
            active_buckets=active_buckets,
            live_enabled=args.exploration_live_admission,
            live_min_samples=args.exploration_live_min_samples,
            live_min_useful_rate=args.exploration_live_min_useful_rate,
            live_min_saving_ms=args.exploration_live_min_saving_ms,
            live_max_p95_regression_ms=args.exploration_live_max_p95_regression_ms,
        )
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({
            "output": str(out),
            "mode": "exploration_policy",
            "extra_capture_sizes": payload["runtime_policy"]["residual_capture"]["extra_capture_sizes"],
            "rules": payload["runtime_policy"]["rules"],
        }, indent=2, ensure_ascii=False))
        return

    if not args.input:
        raise ValueError("--input is required unless --make-exploration-policy is used")
    data = load_json(args.input)
    candidate_policy = load_json(args.candidate_policy) if args.candidate_policy else None
    rows = rows_from_e2e(
        data,
        baseline_contains=args.baseline_contains,
        candidate_contains=args.candidate_contains,
        candidate_policy=candidate_policy,
        base_capture_size=args.base_capture_size,
    )
    seed_policy = load_json(args.seed_policy) if args.seed_policy else None
    planner = ResidualCapturePlanner(
        template_buckets=buckets,
        min_samples=args.min_samples,
        min_useful_rate=args.min_useful_rate,
        min_saving_ms=args.min_saving_ms,
        min_p95_saving_ms=args.min_p95_saving_ms,
        max_p95_regression_ms=args.max_p95_regression_ms,
        max_regression_ms=args.max_regression_ms,
        min_template_tokens=args.min_template_tokens,
        require_exact_template_evidence=not args.allow_template_extrapolation,
        tail_token_threshold=args.tail_token_threshold,
        tail_min_samples=args.tail_min_samples,
        tail_min_useful_rate=args.tail_min_useful_rate,
        tail_min_saving_ms=args.tail_min_saving_ms,
        tail_min_p95_saving_ms=args.tail_min_p95_saving_ms,
        tail_max_p95_regression_ms=args.tail_max_p95_regression_ms,
        tail_max_regression_ms=args.tail_max_regression_ms,
        max_segments=args.max_segments,
        capture_ms_per_template=args.capture_ms_per_template,
        warmup_ms_per_template=args.warmup_ms_per_template,
        amortization_replays=args.amortization_replays,
        graph_action=args.graph_action,
        default_action=args.default_action,
        base_capture_size=args.base_capture_size,
        residual_only=args.mode == "residual_only",
        require_candidate_graph=not args.allow_fallback_candidate_evidence,
    )
    seed_for_plan = seed_policy if (args.mode == "residual_only" or args.preserve_seed_rules) else None
    plan = planner.plan(rows, seed_policy=seed_for_plan)
    payload = plan.to_json()
    payload["analysis"].update({
        "input": args.input,
        "baseline_contains": args.baseline_contains,
        "candidate_contains": args.candidate_contains,
        "seed_policy": args.seed_policy,
        "preserve_seed_rules": args.preserve_seed_rules,
        "candidate_policy": args.candidate_policy,
        "mode": args.mode,
        "allow_fallback_candidate_evidence": args.allow_fallback_candidate_evidence,
        "allow_template_extrapolation": args.allow_template_extrapolation,
        "min_template_tokens": args.min_template_tokens,
        "bucket_preset": args.bucket_preset,
        "tail_token_threshold": args.tail_token_threshold,
    })
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "mode": args.mode,
        "extra_capture_sizes": plan.extra_capture_sizes,
        "global_stats": payload["analysis"]["global_stats"],
        "residual_stats": payload["analysis"]["residual_stats"],
        "admitted": [
            {
                "lo": item.lo,
                "hi": item.hi,
                "template_tokens": item.template_tokens,
                "n": item.n,
                "useful_rate": item.useful_rate,
                "avg_saving_ms": item.avg_saving_ms,
                "effective_saving_ms": item.effective_saving_ms,
                "p95_saving_ms": item.p95_saving_ms,
                "tokens": [obs.tokens for obs in item.observations],
            }
            for item in plan.admitted
        ],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
