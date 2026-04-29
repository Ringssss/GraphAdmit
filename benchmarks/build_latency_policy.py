#!/usr/bin/env python3
"""Build a measured latency policy from vLLM workload result JSON files.

The policy is intentionally measurement-driven: a graph/staticization choice is useful
only if its observed latency beats the best fallback for the same token range.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RANGES = [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 32768)]


def range_name(tok):
    for lo, hi in RANGES:
        if lo < tok <= hi:
            return f"({lo},{hi}]"
    return f">{RANGES[-1][1]}"


def short_action(config):
    if config.startswith("1."):
        return "eager"
    if config.startswith("2."):
        return "vllm_default_graph_or_compiled_fallback"
    if config.startswith("3."):
        return "ours_candidate_graph_policy"
    if config.startswith("4."):
        return "chunked_prefill_graph"
    if config.startswith("5."):
        return "ours_candidate_graph_plus_chunked_prefill"
    return config


def collect(paths):
    rows = defaultdict(lambda: defaultdict(list))
    meta = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        meta.append({"path": str(path), "planner": data.get("planner")})
        for result in data["results"]:
            action = short_action(result["config"])
            for item in result["per_req"]:
                rows[range_name(int(item["tok"]))][action].append(float(item["ms"]))
    return rows, meta


def build_policy(rows, min_samples, margin_pct):
    ranges = {}
    for rng, actions in rows.items():
        candidates = {}
        for action, vals in actions.items():
            if len(vals) < min_samples:
                continue
            arr = np.array(vals, dtype=np.float64)
            candidates[action] = {
                "n": int(len(arr)),
                "avg_ms": float(arr.mean()),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
            }
        if not candidates:
            continue
        best_action = min(candidates, key=lambda key: candidates[key]["avg_ms"])
        best = candidates[best_action]
        ours = candidates.get("ours_candidate_graph_policy")
        useful_ours = False
        if ours is not None:
            non_ours = {k: v for k, v in candidates.items() if k != "ours_candidate_graph_policy"}
            if non_ours:
                best_fallback = min(v["avg_ms"] for v in non_ours.values())
                useful_ours = ours["avg_ms"] <= best_fallback * (1.0 - margin_pct / 100.0)
        ranges[rng] = {
            "chosen_action": best_action,
            "chosen_avg_ms": best["avg_ms"],
            "ours_is_useful": useful_ours,
            "actions": candidates,
        }
    return ranges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--margin-pct", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=Path("results/vllm_latency_policy.json"))
    args = parser.parse_args()
    rows, meta = collect(args.inputs)
    ranges = build_policy(rows, args.min_samples, args.margin_pct)
    output = {
        "inputs": meta,
        "min_samples": args.min_samples,
        "margin_pct": args.margin_pct,
        "ranges": ranges,
        "interpretation": "Choose the measured fastest action per token range; mark our graph policy useful only if it beats the best non-ours fallback by the requested margin.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
