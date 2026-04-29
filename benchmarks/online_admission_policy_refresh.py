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

from prefill_graph.runtime import OnlineSelfLearningAdmissionController


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
        "an admission policy from unvalidated outputs"
    )


def template_for_tokens(tokens: int, buckets: list[int]) -> int | None:
    for bucket in sorted(buckets):
        if tokens <= bucket:
            return bucket
    return None


def read_rows(
    e2e: dict[str, Any],
    *,
    baseline_contains: str,
    candidate_contains: str,
) -> list[dict[str, Any]]:
    baseline = get_result(e2e, baseline_contains)
    candidate = get_result(e2e, candidate_contains)
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
            "fallback_ms": base_ms,
            "graph_ms": cand_ms,
            "correct": bool(correct[idx]),
            "delta_ms": cand_ms - base_ms,
        })
    return rows


def template_id_for_row(row: dict[str, Any], buckets: list[int]) -> tuple[str, int | None]:
    template = template_for_tokens(int(row["tokens"]), buckets)
    if template is None:
        return "overflow", None
    return f"tokens={template}", template


def build_online_policy(
    rows: list[dict[str, Any]],
    *,
    template_buckets: list[int],
    min_admit_tokens: int,
    max_admit_tokens: int,
    graph_action: str,
    default_action: str,
    controller: OnlineSelfLearningAdmissionController,
) -> dict[str, Any]:
    observed_by_template: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (int(item["tokens"]), int(item["idx"]))):
        tokens = int(row["tokens"])
        if min_admit_tokens and tokens < min_admit_tokens:
            continue
        if max_admit_tokens and tokens > max_admit_tokens:
            continue
        template_id, template = template_id_for_row(row, template_buckets)
        if template is None:
            continue
        observed = observed_by_template.setdefault(
            template_id,
            {
                "min_tokens": tokens,
                "max_tokens": tokens,
                "template_tokens": template,
                "action": graph_action,
            },
        )
        observed["min_tokens"] = min(int(observed["min_tokens"]), tokens)
        observed["max_tokens"] = max(int(observed["max_tokens"]), tokens)
        controller.observe(
            template_id,
            graph_ms=float(row["graph_ms"]),
            fallback_ms=float(row["fallback_ms"]),
            correct=bool(row["correct"]),
            metadata={"template_tokens": template, "action": graph_action},
        )
    for template_id, observed in observed_by_template.items():
        state = controller.state(template_id)
        state.metadata.update({
            "lo": int(observed["min_tokens"]) - 1,
            "hi": int(observed["max_tokens"]),
            "template_tokens": int(observed["template_tokens"]),
            "action": observed["action"],
        })
    policy = controller.export_runtime_policy(default_action=default_action)
    policy["online_admission"]["source"] = "online_admission_policy_refresh"
    policy["single_engine_graph_actions"] = ["default", "ours", "cp", graph_action]
    fallback_actions = [
        "eager",
        "compile",
        "compiled",
        "fallback",
        "none",
    ]
    policy["single_engine_fallback_actions"] = [
        action for action in fallback_actions if action != default_action
    ]
    policy["single_engine_allow_multi_req_extra"] = True
    policy["single_engine_requires_fixed_metadata_arena"] = True
    policy["single_engine_base_capture_size"] = 512
    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e", required=True)
    parser.add_argument("--baseline-contains", required=True)
    parser.add_argument("--candidate-contains", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--template-buckets", default="832,1024")
    parser.add_argument("--min-admit-tokens", type=int, default=0)
    parser.add_argument("--max-admit-tokens", type=int, default=0)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--min-useful-rate", type=float, default=0.75)
    parser.add_argument("--min-saving-ms", type=float, default=0.5)
    parser.add_argument("--max-p95-regression-ms", type=float, default=2.0)
    parser.add_argument("--amortization-replays", type=int, default=32)
    parser.add_argument("--graph-action", default="ours_cp")
    parser.add_argument("--default-action", default="cp")
    args = parser.parse_args()

    buckets = [
        int(item.strip())
        for item in args.template_buckets.split(",")
        if item.strip()
    ]
    data = load_json(args.e2e)
    rows = read_rows(
        data,
        baseline_contains=args.baseline_contains,
        candidate_contains=args.candidate_contains,
    )
    controller = OnlineSelfLearningAdmissionController(
        min_samples=args.min_samples,
        min_useful_rate=args.min_useful_rate,
        min_saving_ms=args.min_saving_ms,
        max_p95_regression_ms=args.max_p95_regression_ms,
        amortization_replays=args.amortization_replays,
        fallback_action=args.default_action,
    )
    policy = build_online_policy(
        rows,
        template_buckets=buckets,
        min_admit_tokens=args.min_admit_tokens,
        max_admit_tokens=args.max_admit_tokens,
        graph_action=args.graph_action,
        default_action=args.default_action,
        controller=controller,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runtime_policy": policy}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "rules": policy["rules"],
        "fixed_metadata_arena_ranges": policy["fixed_metadata_arena_ranges"],
        "templates": policy["online_admission"]["templates"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
