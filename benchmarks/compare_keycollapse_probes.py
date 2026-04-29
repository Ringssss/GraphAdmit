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

from benchmarks.analyze_vllm_keycollapse_runtime import summarize


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def row_map(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["group_id"]): row for row in data.get("rows", [])}


def compare_outputs(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_rows = row_map(base)
    cand_rows = row_map(candidate)
    common = sorted(set(base_rows) & set(cand_rows))
    mismatches = []
    latency = []
    for group_id in common:
        base_row = base_rows[group_id]
        cand_row = cand_rows[group_id]
        same = base_row.get("output_token_ids") == cand_row.get("output_token_ids")
        if not same:
            mismatches.append({
                "group_id": group_id,
                "baseline": base_row.get("output_token_ids"),
                "candidate": cand_row.get("output_token_ids"),
            })
        latency.append({
            "group_id": group_id,
            "num_reqs": cand_row.get("num_reqs"),
            "target_total_tokens": cand_row.get("target_total_tokens"),
            "baseline_ms": base_row.get("latency_ms"),
            "candidate_ms": cand_row.get("latency_ms"),
            "speedup": (
                float(base_row["latency_ms"]) / float(cand_row["latency_ms"])
                if cand_row.get("latency_ms") else None
            ),
        })
    valid_speedups = [x["speedup"] for x in latency if x["speedup"] is not None]
    return {
        "common_groups": len(common),
        "output_mismatches": len(mismatches),
        "mismatch_examples": mismatches[:8],
        "all_outputs_match": len(mismatches) == 0 and len(common) > 0,
        "latency": latency,
        "avg_speedup": (
            sum(valid_speedups) / len(valid_speedups) if valid_speedups else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    baseline_summary = summarize(
        baseline.get("dispatcher_profile"),
        baseline.get("runner_profile"),
        baseline.get("attention_profile"),
    )
    candidate_summary = summarize(
        candidate.get("dispatcher_profile"),
        candidate.get("runner_profile"),
        candidate.get("attention_profile"),
    )
    report = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "output_correctness": compare_outputs(baseline, candidate),
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "claim_checks": {
            "same_total_multi_layout_seen": (
                candidate_summary.get("tokens_with_multiple_layouts", 0) > 0
            ),
            "candidate_outputs_match_baseline": None,
            "candidate_metadata_arena_active": False,
            "candidate_num_reqs_fixed": False,
        },
    }
    correctness = report["output_correctness"]
    report["claim_checks"]["candidate_outputs_match_baseline"] = (
        correctness["all_outputs_match"]
    )
    candidate_fields = candidate_summary.get("metadata_fields", {})
    query_info = candidate_fields.get("query_start_loc", {})
    block_info = candidate_fields.get("block_table_gid_0", {})
    seq_info = candidate_fields.get("seq_lens", {})
    is_prefilling_info = candidate_fields.get("is_prefilling", {})
    rank_count = max(
        int(query_info.get("unique_devices", 0) or 0),
        int(seq_info.get("unique_devices", 0) or 0),
        int(block_info.get("unique_devices", 0) or 0),
        1,
    )

    def ptr_stable_per_process(info: dict[str, Any]) -> bool:
        if info.get("max_unique_ptrs_per_device", 0) <= 1:
            return True
        # CPU pinned tensors are reported as device="cpu" for every TP worker.
        # A stable per-process arena therefore has up to one CPU ptr per rank.
        return (
            info.get("unique_devices", 0) == 1
            and info.get("unique_ptrs", 0) <= rank_count
        )

    report["claim_checks"]["candidate_metadata_arena_active"] = bool(
        candidate_summary.get("attention_events")
        and candidate_fields
        and ptr_stable_per_process(is_prefilling_info)
    )
    report["claim_checks"]["candidate_num_reqs_fixed"] = bool(
        query_info.get("unique_shapes", 0) <= 1
        and seq_info.get("unique_shapes", 0) <= 1
        and is_prefilling_info.get("unique_shapes", 0) <= 1
    )
    report["claim_checks"]["candidate_request_metadata_arena_active"] = bool(
        report["claim_checks"]["candidate_num_reqs_fixed"]
        and ptr_stable_per_process(query_info)
        and ptr_stable_per_process(seq_info)
        and ptr_stable_per_process(is_prefilling_info)
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "claim_checks": report["claim_checks"],
        "all_outputs_match": correctness["all_outputs_match"],
        "avg_speedup": correctness["avg_speedup"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
