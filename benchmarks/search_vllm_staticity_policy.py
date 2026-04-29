#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ACTION_PREFIXES = {
    "eager": "1.",
    "default": "2.",
    "ours": "3.",
    "cp": "4.",
    "ours_cp": "5.",
}


@dataclass(frozen=True)
class SegmentChoice:
    start: int
    end: int
    action: str
    cost: float


def percentile(values, pct):
    return float(np.percentile(np.array(values, dtype=np.float64), pct)) if values else None


def stats(values):
    return {
        "avg_ms": float(np.mean(values)) if values else None,
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
    }


def action_name(config):
    for action, prefix in ACTION_PREFIXES.items():
        if config.startswith(prefix):
            return action
    return config


def available_actions(results):
    actions = {}
    for result in results:
        action = action_name(result["config"])
        if action in ACTION_PREFIXES and action not in actions:
            actions[action] = result
    return actions


def choose_segment_action(
    sorted_rows,
    action_results,
    start,
    end,
    allowed_actions,
    prefer_fallback_margin_pct,
    static_actions,
    reference_outputs,
    require_same_output,
):
    best_action = None
    best_cost = float("inf")
    best_values = None
    for action in allowed_actions:
        if require_same_output:
            same = True
            for pos in range(start, end):
                idx = sorted_rows[pos]["idx"]
                same = (
                    action_results[action]["per_req"][idx].get("output_token_ids")
                    == reference_outputs[idx]
                )
                if not same:
                    break
            if not same:
                continue
        values = [
            float(action_results[action]["per_req"][sorted_rows[pos]["idx"]]["ms"])
            for pos in range(start, end)
        ]
        cost = float(sum(values))
        if cost < best_cost:
            best_action = action
            best_cost = cost
            best_values = values

    if (
        best_action in static_actions
        and prefer_fallback_margin_pct > 0
        and len(allowed_actions) > 1
    ):
        fallback_candidates = [
            action
            for action in allowed_actions
            if action not in static_actions
        ]
        if fallback_candidates:
            fallback_action = min(
                fallback_candidates,
                key=lambda action: sum(
                    float(action_results[action]["per_req"][sorted_rows[pos]["idx"]]["ms"])
                    for pos in range(start, end)
                ),
            )
            fallback_values = [
                float(action_results[fallback_action]["per_req"][sorted_rows[pos]["idx"]]["ms"])
                for pos in range(start, end)
            ]
            if require_same_output:
                fallback_same = all(
                    action_results[fallback_action]["per_req"][sorted_rows[pos]["idx"]].get("output_token_ids")
                    == reference_outputs[sorted_rows[pos]["idx"]]
                    for pos in range(start, end)
                )
                if not fallback_same:
                    return SegmentChoice(start, end, best_action, best_cost), best_values
            fallback_cost = float(sum(fallback_values))
            required = fallback_cost * (1.0 - prefer_fallback_margin_pct / 100.0)
            if best_cost > required:
                best_action = fallback_action
                best_cost = fallback_cost
                best_values = fallback_values

    return SegmentChoice(start, end, best_action, best_cost), best_values


def collapse_segments(segments, sorted_rows):
    collapsed = []
    for seg in segments:
        lo = 0 if seg.start == 0 else int(sorted_rows[seg.start - 1]["tok"])
        hi = int(sorted_rows[seg.end - 1]["tok"])
        item = {
            "lo": lo,
            "hi": hi,
            "action": seg.action,
            "n": int(seg.end - seg.start),
        }
        if collapsed and collapsed[-1]["action"] == item["action"] and collapsed[-1]["hi"] == item["lo"]:
            collapsed[-1]["hi"] = item["hi"]
            collapsed[-1]["n"] += item["n"]
        else:
            collapsed.append(item)
    return collapsed


def build_policy(
    data,
    allowed_actions,
    max_segments,
    min_segment_samples,
    segment_penalty_ms,
    prefer_fallback_margin_pct,
    static_actions,
    require_same_output,
):
    action_results = available_actions(data["results"])
    missing = [action for action in allowed_actions if action not in action_results]
    if missing:
        raise ValueError(f"requested actions not found in input: {missing}")

    reference = next(iter(action_results.values()))
    reference_outputs = [row.get("output_token_ids") for row in reference["per_req"]]
    sorted_rows = sorted(
        (
            {"idx": idx, "tok": int(row["tok"])}
            for idx, row in enumerate(reference["per_req"])
        ),
        key=lambda item: (item["tok"], item["idx"]),
    )
    n = len(sorted_rows)
    if n == 0:
        raise ValueError("empty per_req")

    choices = {}
    for start in range(n):
        for end in range(start + min_segment_samples, n + 1):
            choice, _ = choose_segment_action(
                sorted_rows,
                action_results,
                start,
                end,
                allowed_actions,
                prefer_fallback_margin_pct,
                static_actions,
                reference_outputs,
                require_same_output,
            )
            if choice.action is None:
                continue
            choices[(start, end)] = choice

    dp = [[(float("inf"), []) for _ in range(n + 1)] for _ in range(max_segments + 1)]
    dp[0][0] = (0.0, [])
    for parts in range(1, max_segments + 1):
        for end in range(1, n + 1):
            best_cost = float("inf")
            best_path = None
            for start in range(0, end):
                if end - start < min_segment_samples and end != n:
                    continue
                prev_cost, prev_path = dp[parts - 1][start]
                if prev_cost == float("inf"):
                    continue
                choice = choices.get((start, end))
                if choice is None:
                    continue
                cost = prev_cost + choice.cost + segment_penalty_ms
                if cost < best_cost:
                    best_cost = cost
                    best_path = prev_path + [choice]
            if best_path is not None:
                dp[parts][end] = (best_cost, best_path)

    candidates = [
        dp[parts][n]
        for parts in range(1, max_segments + 1)
        if dp[parts][n][0] != float("inf")
    ]
    if not candidates:
        raise ValueError("no policy candidate found")
    _, segments = min(candidates, key=lambda item: (item[0], len(item[1])))
    rules = collapse_segments(segments, sorted_rows)

    rows = []
    chosen_ms = []
    for idx, row in enumerate(reference["per_req"]):
        tok = int(row["tok"])
        action = None
        for rule in rules:
            if rule["lo"] < tok <= rule["hi"]:
                action = rule["action"]
                break
        if action is None:
            action = rules[-1]["action"]
        source = action_results[action]["per_req"][idx]
        ms = float(source["ms"])
        chosen_ms.append(ms)
        rows.append(
            {
                "idx": idx,
                "tok": tok,
                "action": action,
                "ms": ms,
                "same_output_vs_reference": source.get("output_token_ids") == row.get("output_token_ids"),
            }
        )

    baseline_stats = {
        action: stats([float(row["ms"]) for row in result["per_req"]])
        for action, result in action_results.items()
        if action in allowed_actions
    }
    return {
        "source": None,
        "allowed_actions": allowed_actions,
        "max_segments": max_segments,
        "min_segment_samples": min_segment_samples,
        "segment_penalty_ms": segment_penalty_ms,
        "prefer_fallback_margin_pct": prefer_fallback_margin_pct,
        "static_actions": sorted(static_actions),
        "require_same_output": require_same_output,
        "rules": rules,
        "policy_stats": stats(chosen_ms),
        "baseline_stats": baseline_stats,
        "speedup_vs_default_avg": (
            baseline_stats["default"]["avg_ms"] / stats(chosen_ms)["avg_ms"]
            if "default" in baseline_stats
            else None
        ),
        "speedup_vs_cp_avg": (
            baseline_stats["cp"]["avg_ms"] / stats(chosen_ms)["avg_ms"]
            if "cp" in baseline_stats
            else None
        ),
        "all_same_outputs_vs_reference": all(row["same_output_vs_reference"] for row in rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allowed-actions",
        default="default,ours,cp",
        help="comma-separated subset from eager,default,ours,cp",
    )
    parser.add_argument("--max-segments", type=int, default=3)
    parser.add_argument("--min-segment-samples", type=int, default=1)
    parser.add_argument("--segment-penalty-ms", type=float, default=0.0)
    parser.add_argument(
        "--prefer-fallback-margin-pct",
        type=float,
        default=0.0,
        help="if a static action wins by less than this margin, choose the best non-static fallback",
    )
    parser.add_argument(
        "--static-actions",
        default="ours",
        help="comma-separated actions treated as staticization candidates for margin gating",
    )
    parser.add_argument(
        "--require-same-output",
        action="store_true",
        help="only choose an action for a request if its output token ids match the first action",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    allowed_actions = [item.strip() for item in args.allowed_actions.split(",") if item.strip()]
    unknown = sorted(set(allowed_actions) - set(ACTION_PREFIXES))
    if unknown:
        raise ValueError(f"unknown actions: {unknown}")
    static_actions = {item.strip() for item in args.static_actions.split(",") if item.strip()}
    policy = build_policy(
        data,
        allowed_actions,
        args.max_segments,
        args.min_segment_samples,
        args.segment_penalty_ms,
        args.prefer_fallback_margin_pct,
        static_actions,
        args.require_same_output,
    )
    policy["source"] = args.input
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in policy.items() if key != "rows"}, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
