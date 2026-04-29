#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prefill_graph.runtime import DynamicityProfiler, RequestContext, RuntimePlanner, RuntimePolicy


ACTION_PREFIXES = {
    "eager": "1.",
    "default": "2.",
    "ours": "3.",
    "cp": "4.",
    "ours_cp": "5.",
}


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def stats_ms(values: list[float]) -> dict[str, Any]:
    return {
        "avg_ms": sum(values) / len(values) if values else None,
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "total_ms": sum(values),
    }


def stats_s(values: list[float]) -> dict[str, Any]:
    return {
        "avg_s": sum(values) / len(values) if values else None,
        "p50_s": percentile(values, 50),
        "p95_s": percentile(values, 95),
        "p99_s": percentile(values, 99),
        "total_s": sum(values),
    }


def action_name(config: str) -> str:
    for action, prefix in ACTION_PREFIXES.items():
        if config.startswith(prefix):
            return action
    return config


def available_vllm_actions(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    actions = {}
    for result in data["results"]:
        action = action_name(result["config"])
        if action not in actions:
            actions[action] = result
    return actions


def replay_vllm(input_path: Path, policy_path: Path, output_path: Path) -> dict[str, Any]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    policy = RuntimePolicy.from_json_file(policy_path)
    planner = RuntimePlanner(policy)
    profiler = DynamicityProfiler()
    actions = available_vllm_actions(data)
    reference = actions.get("default") or next(iter(actions.values()))
    chosen_ms: list[float] = []
    rows = []
    for idx, ref_row in enumerate(reference["per_req"]):
        tok = int(ref_row["tok"])
        profiler.observe("num_tokens", tok, in_graph_key=True, semantic=True, component="vllm_runtime_replay")
        profiler.observe("feature_flags", sorted(actions), in_graph_key=True, semantic=True, component="vllm_runtime_replay")
        decision = planner.choose(RequestContext(idx=idx, tokens=tok, mode="prefill"))
        if decision.action not in actions:
            raise ValueError(f"policy selected {decision.action!r}, but input has only {sorted(actions)}")
        source = actions[decision.action]["per_req"][idx]
        same = source.get("output_token_ids") == ref_row.get("output_token_ids")
        if policy.correctness_required and not same:
            fallback = policy.baseline_action if policy.baseline_action in actions else "default"
            source = actions[fallback]["per_req"][idx]
            decision_action = fallback
            admitted = False
            reason = "runtime_correctness_fallback"
            same = source.get("output_token_ids") == ref_row.get("output_token_ids")
        else:
            decision_action = decision.action
            admitted = True
            reason = decision.reason
        ms = float(source["ms"])
        chosen_ms.append(ms)
        rows.append(
            {
                "idx": idx,
                "tok": tok,
                "planned_action": decision.action,
                "action": decision_action,
                "admitted": admitted,
                "reason": reason,
                "ms": ms,
                "same_output_vs_default": same,
            }
        )
    baseline_stats = {
        action: stats_ms([float(row["ms"]) for row in result["per_req"]])
        for action, result in actions.items()
    }
    runtime_stats = stats_ms(chosen_ms)
    output = {
        "kind": "vllm_online_runtime_replay",
        "source": str(input_path),
        "policy_source": str(policy_path),
        "model": data.get("model"),
        "workload": data.get("workload"),
        "runtime_stats": runtime_stats,
        "baseline_stats": baseline_stats,
        "speedup_vs_default_avg": (
            baseline_stats["default"]["avg_ms"] / runtime_stats["avg_ms"]
            if "default" in baseline_stats and runtime_stats["avg_ms"]
            else None
        ),
        "speedup_vs_cp_avg": (
            baseline_stats["cp"]["avg_ms"] / runtime_stats["avg_ms"]
            if "cp" in baseline_stats and runtime_stats["avg_ms"]
            else None
        ),
        "all_same_outputs_vs_default": all(row["same_output_vs_default"] for row in rows),
        "planner": planner.summary(),
        "dynamicity": profiler.summary(),
        "rows": rows,
        "runtime_boundary": (
            "This is an online request-by-request replay over calibrated action measurements. "
            "It validates planner/admission behavior without pretending that stock vLLM can "
            "switch CP/capture-size engine configuration per request inside one engine."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def replay_dinfer(input_path: Path, output_path: Path, min_prompt_len: int) -> dict[str, Any]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    profiler = DynamicityProfiler()
    rows = []
    chosen_s: list[float] = []
    eager_s: list[float] = []
    graph_s: list[float] = []
    for row in data["rows"]:
        idx = int(row["idx"])
        prompt_len = int(row["prompt_len"])
        stats = row.get("graph_stats") or {}
        profiler.observe("num_tokens", prompt_len, in_graph_key=True, semantic=True, component="dinfer_runtime_replay")
        profiler.observe("mask_positions", stats.get("metadata_updates", 0), in_graph_key=True, semantic=True, component="dinfer_runtime_replay")
        profiler.observe("kv_cache", stats.get("template_count", 0), in_graph_key=True, semantic=True, component="dinfer_runtime_replay")
        profiler.observe("expert_ids", stats.get("template_misses", 0), in_graph_key=True, semantic=True, component="dinfer_runtime_replay")
        wants_graph = prompt_len >= min_prompt_len
        graph_correct = bool(row.get("same_tokens"))
        graph_available = bool(row.get("used_graph", True))
        use_graph = wants_graph and graph_available and graph_correct
        seconds = float(row["graph_s"] if use_graph else row["eager_s"])
        chosen_s.append(seconds)
        eager_s.append(float(row["eager_s"]))
        graph_s.append(float(row["graph_s"]))
        rows.append(
            {
                "idx": idx,
                "prompt_len": prompt_len,
                "planned_action": "graph" if wants_graph else "eager",
                "action": "graph" if use_graph else "eager",
                "admitted": use_graph,
                "reason": (
                    "decoded_validation_passed"
                    if use_graph
                    else "prompt_len_guard"
                    if not wants_graph
                    else "correctness_or_availability_fallback"
                ),
                "seconds": seconds,
                "same_tokens": True if not use_graph else graph_correct,
                "graph_stats": stats,
            }
        )
    runtime_stats = stats_s(chosen_s)
    eager_stats = stats_s(eager_s)
    graph_stats = stats_s(graph_s)
    cleanup_total = float(data.get("cleanup_total_s", 0.0) or 0.0)
    output = {
        "kind": "dinfer_online_runtime_replay",
        "source": str(input_path),
        "model": data.get("model"),
        "workload": data.get("workload"),
        "min_prompt_len": min_prompt_len,
        "runtime_stats": runtime_stats,
        "eager_stats": eager_stats,
        "graph_stats": graph_stats,
        "cleanup_total_s": cleanup_total,
        "runtime_total_with_cleanup_s": runtime_stats["total_s"] + cleanup_total,
        "speedup_vs_eager_total": (
            eager_stats["total_s"] / runtime_stats["total_s"]
            if runtime_stats["total_s"]
            else None
        ),
        "speedup_vs_eager_total_with_cleanup": (
            eager_stats["total_s"] / (runtime_stats["total_s"] + cleanup_total)
            if runtime_stats["total_s"] is not None and runtime_stats["total_s"] + cleanup_total > 0
            else None
        ),
        "all_same_tokens": all(row["same_tokens"] for row in rows),
        "dynamicity": profiler.summary(),
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["vllm", "dinfer"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", help="vLLM runtime policy JSON")
    parser.add_argument("--min-prompt-len", type=int, default=0)
    args = parser.parse_args()

    if args.kind == "vllm":
        if not args.policy:
            raise ValueError("--policy is required for --kind vllm")
        result = replay_vllm(Path(args.input), Path(args.policy), Path(args.output))
    else:
        result = replay_dinfer(Path(args.input), Path(args.output), args.min_prompt_len)
    printable = {key: value for key, value in result.items() if key not in {"rows", "dynamicity"}}
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
