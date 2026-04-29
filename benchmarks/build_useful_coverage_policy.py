#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_BUCKETS = [512, 640, 768, 832, 896, 1024, 1280, 1536, 2048, 3072, 4096]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
        "a useful-coverage policy from unvalidated outputs"
    )


def template_for_tokens(tokens: int, buckets: list[int]) -> int | None:
    for bucket in sorted(buckets):
        if tokens <= bucket:
            return bucket
    return None


def candidate_rows(data: dict[str, Any], baseline_contains: str, candidate_contains: str) -> list[dict[str, Any]]:
    baseline = get_result(data, baseline_contains)
    candidate = get_result(data, candidate_contains)
    base_rows = baseline.get("per_req", [])
    cand_rows = candidate.get("per_req", [])
    n = min(len(base_rows), len(cand_rows))
    correct = correctness_flags(candidate, n)
    rows = []
    for idx in range(n):
        tok = int(cand_rows[idx].get("tok", base_rows[idx].get("tok", 0)))
        base_ms = float(base_rows[idx]["ms"])
        cand_ms = float(cand_rows[idx]["ms"])
        rows.append({
            "idx": idx,
            "tokens": tok,
            "baseline_ms": base_ms,
            "candidate_ms": cand_ms,
            "delta_ms": cand_ms - base_ms,
            "correct": bool(correct[idx]),
        })
    return rows


def segment_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "wins": 0,
            "losses": 0,
            "mismatches": 0,
            "avg_delta_ms": None,
            "avg_baseline_ms": None,
            "avg_candidate_ms": None,
            "useful_rate": 0.0,
            "max_regression_ms": None,
        }
    useful = [row for row in rows if row["correct"] and row["candidate_ms"] < row["baseline_ms"]]
    negative = [row for row in rows if (not row["correct"]) or row["candidate_ms"] >= row["baseline_ms"]]
    return {
        "n": len(rows),
        "wins": len(useful),
        "losses": len(negative),
        "mismatches": sum(not row["correct"] for row in rows),
        "avg_delta_ms": mean(row["delta_ms"] for row in rows),
        "avg_baseline_ms": mean(row["baseline_ms"] for row in rows),
        "avg_candidate_ms": mean(row["candidate_ms"] for row in rows),
        "useful_rate": len(useful) / len(rows),
        "max_regression_ms": max(row["delta_ms"] for row in rows),
        "max_win_ms": min(row["delta_ms"] for row in rows),
    }


def dynamic_program_segments(
    rows: list[dict[str, Any]],
    *,
    template_buckets: list[int],
    min_admit_tokens: int,
    max_admit_tokens: int,
    min_samples: int,
    min_useful_rate: float,
    min_avg_saving_ms: float,
    max_regression_ms: float,
    max_segments: int,
) -> list[dict[str, Any]]:
    ordered = [
        row for row in sorted(rows, key=lambda item: (item["tokens"], item["idx"]))
        if (not min_admit_tokens or int(row["tokens"]) >= min_admit_tokens)
        and (not max_admit_tokens or int(row["tokens"]) <= max_admit_tokens)
    ]
    n = len(ordered)
    if n == 0:
        return []

    # Candidate intervals are token-contiguous slices over the sorted trace.
    intervals = []
    for left in range(n):
        for right in range(left, n):
            slice_rows = ordered[left:right + 1]
            lo = int(ordered[left]["tokens"]) - 1
            hi = int(ordered[right]["tokens"])
            if min_admit_tokens and hi < min_admit_tokens:
                continue
            if max_admit_tokens and lo >= max_admit_tokens:
                continue
            if min_admit_tokens:
                lo = max(lo, min_admit_tokens - 1)
            if max_admit_tokens:
                hi = min(hi, max_admit_tokens)
            template = template_for_tokens(hi, template_buckets)
            if template is None:
                continue
            stats = segment_stats(slice_rows)
            if stats["n"] < min_samples:
                continue
            avg_saving = -float(stats["avg_delta_ms"])
            if (
                stats["mismatches"] == 0
                and stats["useful_rate"] >= min_useful_rate
                and avg_saving >= min_avg_saving_ms
                and float(stats["max_regression_ms"]) <= max_regression_ms
            ):
                intervals.append({
                    "left": left,
                    "right": right,
                    "lo": lo,
                    "hi": hi,
                    "template_tokens": template,
                    "stats": stats,
                    "score": avg_saving * stats["n"],
                    "tokens": [int(row["tokens"]) for row in slice_rows],
                })

    # Weighted interval scheduling with a small segment cap.
    intervals.sort(key=lambda item: (item["right"], item["left"]))
    prev = []
    for item in intervals:
        p = -1
        for j in range(len(intervals) - 1, -1, -1):
            if intervals[j]["right"] < item["left"]:
                p = j
                break
        prev.append(p)

    dp = [[0.0] * (len(intervals) + 1) for _ in range(max_segments + 1)]
    take = [[False] * (len(intervals) + 1) for _ in range(max_segments + 1)]
    for k in range(1, max_segments + 1):
        for i in range(1, len(intervals) + 1):
            skip = dp[k][i - 1]
            item = intervals[i - 1]
            with_item = item["score"] + dp[k - 1][prev[i - 1] + 1]
            if with_item > skip:
                dp[k][i] = with_item
                take[k][i] = True
            else:
                dp[k][i] = skip

    selected = []
    k = max_segments
    i = len(intervals)
    while k > 0 and i > 0:
        if take[k][i]:
            item = intervals[i - 1]
            selected.append(item)
            i = prev[i - 1] + 1
            k -= 1
        else:
            i -= 1
    return sorted(selected, key=lambda item: item["lo"])


def fill_policy_rules(
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    template_buckets: list[int],
    default_action: str,
    graph_action: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    max_tok = max([row["tokens"] for row in rows], default=4096)
    boundaries = {0, max_tok}
    for segment in selected:
        boundaries.add(segment["lo"])
        boundaries.add(segment["hi"])
    sorted_bounds = sorted(boundaries)

    rules = []
    arena_ranges = []
    for left, right in zip(sorted_bounds, sorted_bounds[1:]):
        if left == right:
            continue
        admitted = None
        for segment in selected:
            if segment["lo"] == left and segment["hi"] == right:
                admitted = segment
                break
        if admitted:
            template = admitted.get("template_tokens") or template_for_tokens(right, template_buckets)
            if template is None:
                raise ValueError(f"admitted range ({left},{right}] has no template bucket")
            rule = {
                "lo": left,
                "hi": right,
                "action": graph_action,
                "n": admitted["stats"]["n"],
                "template_tokens": template,
                "reason": (
                    "auto-admitted by useful-coverage policy: "
                    f"useful_rate={admitted['stats']['useful_rate']:.2f}, "
                    f"avg_saving_ms={-admitted['stats']['avg_delta_ms']:.2f}"
                ),
            }
            rules.append(rule)
            arena_ranges.append({
                "lo": left,
                "hi": right,
                "template_tokens": template,
                "action": graph_action,
                "n": admitted["stats"]["n"],
            })
        else:
            in_range = [row for row in rows if left < row["tokens"] <= right]
            rules.append({
                "lo": left,
                "hi": right,
                "action": default_action,
                "n": len(in_range),
                "reason": "auto-rejected: insufficient useful coverage or regression guard",
            })
    return rules, arena_ranges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--baseline-contains", required=True)
    parser.add_argument("--candidate-contains", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--default-action", default="cp")
    parser.add_argument("--graph-action", default="ours_cp")
    parser.add_argument("--template-buckets", default="512,640,768,832,896,1024,1280,1536,2048,3072,4096")
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--min-admit-tokens", type=int, default=0)
    parser.add_argument("--max-admit-tokens", type=int, default=0)
    parser.add_argument("--min-useful-rate", type=float, default=0.75)
    parser.add_argument("--min-avg-saving-ms", type=float, default=1.0)
    parser.add_argument("--max-regression-ms", type=float, default=5.0)
    parser.add_argument("--max-segments", type=int, default=4)
    args = parser.parse_args()

    data = load_json(args.input)
    rows = candidate_rows(data, args.baseline_contains, args.candidate_contains)
    template_buckets = [
        int(item) for item in args.template_buckets.split(",")
        if item.strip()
    ] or DEFAULT_BUCKETS
    selected = dynamic_program_segments(
        rows,
        template_buckets=template_buckets,
        min_admit_tokens=args.min_admit_tokens,
        max_admit_tokens=args.max_admit_tokens,
        min_samples=args.min_samples,
        min_useful_rate=args.min_useful_rate,
        min_avg_saving_ms=args.min_avg_saving_ms,
        max_regression_ms=args.max_regression_ms,
        max_segments=args.max_segments,
    )
    rules, arena_ranges = fill_policy_rules(
        rows,
        selected,
        template_buckets=template_buckets,
        default_action=args.default_action,
        graph_action=args.graph_action,
    )
    policy = {
        "runtime_policy": {
            "kind": "vllm",
            "source_e2e": args.input,
            "default_action": args.default_action,
            "baseline_action": args.default_action,
            "correctness_required": True,
            "rules": rules,
            "fixed_metadata_arena_ranges": arena_ranges,
            "single_engine_graph_actions": ["default", "ours", "cp", "ours_cp"],
            "single_engine_fallback_actions": [
                "eager",
                "compile",
                "compiled",
                "fallback",
                "none",
            ],
            "single_engine_allow_multi_req_extra": True,
            "single_engine_requires_fixed_metadata_arena": True,
            "single_engine_max_extra_templates": max(1, len(arena_ranges)),
            "single_engine_min_rule_n": args.min_samples,
            "single_engine_base_capture_size": 512,
            "online_admission": {
                "mode": "useful_coverage_learned_policy",
                "min_useful_rate": args.min_useful_rate,
                "min_avg_saving_ms": args.min_avg_saving_ms,
                "max_regression_ms": args.max_regression_ms,
                "max_segments": args.max_segments,
                "min_admit_tokens": args.min_admit_tokens,
                "max_admit_tokens": args.max_admit_tokens,
                "selected_segments": selected,
            },
        },
        "analysis": {
            "input": args.input,
            "rows": rows,
            "global_stats": segment_stats(rows),
            "selected_segments": selected,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "rules": len(rules),
        "arena_ranges": arena_ranges,
        "selected": [
            {
                "lo": item["lo"],
                "hi": item["hi"],
                "n": item["stats"]["n"],
                "useful_rate": item["stats"]["useful_rate"],
                "avg_saving_ms": -item["stats"]["avg_delta_ms"],
                "max_regression_ms": item["stats"]["max_regression_ms"],
                "tokens": item["tokens"],
            }
            for item in selected
        ],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
