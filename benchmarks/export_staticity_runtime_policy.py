#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="policy produced by search_vllm_staticity_policy.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", choices=["vllm", "dinfer"], default="vllm")
    parser.add_argument(
        "--single-engine-graph-actions",
        default="default,ours,cp,ours_cp",
        help="actions that are graph-capable inside the current vLLM engine",
    )
    parser.add_argument(
        "--single-engine-fallback-actions",
        default="eager,compile,compiled,fallback,none",
        help="actions that should force CUDAGraphMode.NONE inside the current vLLM engine",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    policy = {
        "kind": args.kind,
        "rules": data.get("rules", []),
        "default_action": data.get("default_action") or (data.get("rules", [{}])[-1].get("action", "default") if data.get("rules") else "default"),
        "baseline_action": data.get("baseline_action", "default"),
        "correctness_required": bool(data.get("require_same_output", data.get("require_correct", True))),
        "latency_margin_pct": float(data.get("prefer_fallback_margin_pct", 0.0) or 0.0),
        "baseline_stats": data.get("baseline_stats"),
        "policy_stats": data.get("policy_stats") or data.get("best_policy_stats"),
        "single_engine_graph_actions": [
            item.strip()
            for item in args.single_engine_graph_actions.split(",")
            if item.strip()
        ],
        "single_engine_fallback_actions": [
            item.strip()
            for item in args.single_engine_fallback_actions.split(",")
            if item.strip()
        ],
        "engine_level_actions": ["cp", "default", "ours", "ours_cp"],
        "runtime_boundary": (
            "This policy can be consumed online.  In stock vLLM one engine can only "
            "force CUDA graph on/off for its current configuration; it cannot switch "
            "chunked-prefill or capture-size configuration per request.  True per-request "
            "selection among default/CP/ours requires a broker over multiple engines or "
            "deeper scheduler/engine patching."
        ),
        "source_policy": args.input,
        "source_summary": {
            "policy_stats": data.get("policy_stats") or data.get("best_policy_stats"),
            "baseline_stats": data.get("baseline_stats"),
            "speedup_vs_default_avg": data.get("speedup_vs_default_avg"),
            "speedup_vs_cp_avg": data.get("speedup_vs_cp_avg"),
            "all_same_outputs_vs_reference": data.get("all_same_outputs_vs_reference"),
            "best_all_same_tokens": data.get("best_all_same_tokens"),
            "best_speedup_vs_eager_total": data.get("best_speedup_vs_eager_total"),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runtime_policy": policy}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(policy, indent=2, ensure_ascii=False))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
