#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def percentile(values, pct):
    return float(np.percentile(np.array(values, dtype=np.float64), pct)) if values else None


def stats(values):
    return {
        "avg_s": float(np.mean(values)) if values else None,
        "p50_s": percentile(values, 50),
        "p95_s": percentile(values, 95),
        "p99_s": percentile(values, 99),
        "total_s": float(sum(values)),
    }


def evaluate_threshold(rows, min_prompt_len, require_correct):
    chosen = []
    policy_rows = []
    correct = True
    unevaluable_graph_requests = 0
    for row in rows:
        wants_graph = int(row["prompt_len"]) >= min_prompt_len
        has_measured_graph = bool(row.get("used_graph", True))
        use_graph = wants_graph and has_measured_graph
        if wants_graph and not has_measured_graph:
            unevaluable_graph_requests += 1
        if require_correct and use_graph and not row.get("same_tokens", False):
            use_graph = False
        seconds = float(row["graph_s"] if use_graph else row["eager_s"])
        chosen.append(seconds)
        same = bool(row.get("same_tokens", False)) if use_graph else True
        correct = correct and same
        policy_rows.append(
            {
                "idx": int(row["idx"]),
                "prompt_len": int(row["prompt_len"]),
                "action": "graph" if use_graph else "eager",
                "wanted_graph": wants_graph,
                "has_measured_graph": has_measured_graph,
                "seconds": seconds,
                "same_tokens": same,
                "observed_graph_speedup": row.get("speedup"),
            }
        )
    eager_total = sum(float(row["eager_s"]) for row in rows)
    return {
        "min_prompt_len": int(min_prompt_len),
        "policy_stats": stats(chosen),
        "speedup_vs_eager_total": eager_total / sum(chosen) if chosen else None,
        "all_same_tokens": correct,
        "unevaluable_graph_requests": unevaluable_graph_requests,
        "rows": policy_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--candidate-thresholds",
        default=None,
        help="comma-separated prompt_len thresholds; default uses observed lengths plus 0",
    )
    parser.add_argument("--require-correct", action="store_true")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = data["rows"]
    if args.candidate_thresholds:
        thresholds = sorted({int(item) for item in args.candidate_thresholds.split(",") if item.strip()})
    else:
        thresholds = sorted({0, *[int(row["prompt_len"]) for row in rows], max(int(row["prompt_len"]) for row in rows) + 1})

    candidates = [
        evaluate_threshold(rows, threshold, args.require_correct)
        for threshold in thresholds
    ]
    best = min(
        candidates,
        key=lambda item: (
            not item["all_same_tokens"],
            item["unevaluable_graph_requests"],
            item["policy_stats"]["total_s"],
            item["min_prompt_len"],
        ),
    )
    output = {
        "source": args.input,
        "require_correct": args.require_correct,
        "best_min_prompt_len": best["min_prompt_len"],
        "best_policy_stats": best["policy_stats"],
        "best_speedup_vs_eager_total": best["speedup_vs_eager_total"],
        "best_all_same_tokens": best["all_same_tokens"],
        "candidates": [
            {key: value for key, value in candidate.items() if key != "rows"}
            for candidate in candidates
        ],
        "rows": best["rows"],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
