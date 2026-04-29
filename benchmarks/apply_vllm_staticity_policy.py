#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


ACTION_PREFIXES = {
    "eager": "1.",
    "default": "2.",
    "ours": "3.",
    "cp": "4.",
    "ours_cp": "5.",
}


def percentile(values, pct):
    return float(np.percentile(np.array(values, dtype=np.float64), pct)) if values else None


def stats(values):
    return {
        "avg_ms": float(np.mean(values)),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
    }


def parse_rule(rule):
    try:
        lo, hi, action = rule.split(":")
    except ValueError as exc:
        raise ValueError(f"bad rule {rule!r}; expected lo:hi:action") from exc
    action = action.strip()
    if action not in ACTION_PREFIXES:
        raise ValueError(f"unknown action {action!r}; expected {sorted(ACTION_PREFIXES)}")
    return float(lo), float(hi), action


def find_action(results, action):
    prefix = ACTION_PREFIXES[action]
    for result in results:
        if result["config"].startswith(prefix):
            return result
    raise ValueError(f"missing action {action!r} with config prefix {prefix!r}")


def choose_action(tok, rules, default_action):
    for lo, hi, action in rules:
        if lo < tok <= hi:
            return action
    return default_action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--rule",
        action="append",
        default=[],
        help="range action as lo:hi:action, e.g. 0:832:ours",
    )
    parser.add_argument("--default-action", choices=sorted(ACTION_PREFIXES), default="cp")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rules = [parse_rule(rule) for rule in args.rule]
    action_results = {
        action: find_action(data["results"], action)
        for action in ACTION_PREFIXES
        if any(result["config"].startswith(ACTION_PREFIXES[action]) for result in data["results"])
    }

    if args.default_action not in action_results:
        raise ValueError(f"default action {args.default_action!r} is not present in input")
    for _, _, action in rules:
        if action not in action_results:
            raise ValueError(f"rule action {action!r} is not present in input")

    reference = data["results"][0]
    rows = []
    chosen_ms = []
    for idx, item in enumerate(reference["per_req"]):
        tok = int(item["tok"])
        action = choose_action(tok, rules, args.default_action)
        source = action_results[action]["per_req"][idx]
        ms = float(source["ms"])
        chosen_ms.append(ms)
        rows.append(
            {
                "idx": idx,
                "tok": tok,
                "action": action,
                "ms": ms,
                "same_output_vs_reference": source.get("output_token_ids") == item.get("output_token_ids"),
            }
        )

    baseline_stats = {
        action: stats([float(row["ms"]) for row in result["per_req"]])
        for action, result in action_results.items()
    }
    policy_stats = stats(chosen_ms)
    output = {
        "source": args.input,
        "rules": [{"lo": lo, "hi": hi, "action": action} for lo, hi, action in rules],
        "default_action": args.default_action,
        "policy_stats": policy_stats,
        "baseline_stats": baseline_stats,
        "speedup_vs_default_avg": (
            baseline_stats["default"]["avg_ms"] / policy_stats["avg_ms"]
            if "default" in baseline_stats
            else None
        ),
        "speedup_vs_cp_avg": (
            baseline_stats["cp"]["avg_ms"] / policy_stats["avg_ms"]
            if "cp" in baseline_stats
            else None
        ),
        "all_same_outputs_vs_reference": all(row["same_output_vs_reference"] for row in rows),
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
