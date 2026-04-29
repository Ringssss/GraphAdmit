#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def canonicalize(lengths: list[int], total_tokens: int, arena_reqs: int) -> dict:
    if len(lengths) > arena_reqs:
        raise ValueError(f"layout has {len(lengths)} reqs > arena_reqs={arena_reqs}")
    query_start_loc = [0]
    running = 0
    for length in lengths:
        running += int(length)
        query_start_loc.append(running)
    query_start_loc.extend([total_tokens] * (arena_reqs + 1 - len(query_start_loc)))
    seq_lens = [int(x) for x in lengths] + [0] * (arena_reqs - len(lengths))
    is_prefilling = [True] * len(lengths) + [False] * (arena_reqs - len(lengths))
    block_table_shape = [arena_reqs, 32]
    return {
        "template_key": {
            "bucketed_tokens": total_tokens,
            "mode_flags": "prefill",
        },
        "full_key_baseline": {
            "bucketed_tokens": total_tokens,
            "num_reqs": len(lengths),
            "layout": lengths,
        },
        "metadata_shapes": {
            "query_start_loc": [arena_reqs + 1],
            "seq_lens": [arena_reqs],
            "is_prefilling": [arena_reqs],
            "block_table": block_table_shape,
            "slot_mapping": [total_tokens],
        },
        "metadata_values": {
            "query_start_loc": query_start_loc,
            "seq_lens": seq_lens,
            "is_prefilling": is_prefilling,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default="results/keycollapse_controlled_small.json")
    parser.add_argument("--arena-reqs", type=int, default=8)
    parser.add_argument("--output", default="results/fixed_arena_keycollapse_poc.json")
    args = parser.parse_args()

    workload = json.loads(Path(args.workload).read_text(encoding="utf-8"))
    groups: dict[str, list[dict]] = {}
    for req in workload["requests"]:
        groups.setdefault(req["group_id"], []).append(req)

    rows = []
    baseline_keys = set()
    canonical_keys = set()
    shape_keys = set()
    by_total: dict[int, list[str]] = {}
    for group_id, reqs in groups.items():
        lengths = [int(req["actual_input_length"]) for req in reqs]
        total = sum(lengths)
        item = canonicalize(lengths, total, args.arena_reqs)
        baseline_keys.add(json.dumps(item["full_key_baseline"], sort_keys=True))
        canonical_keys.add(json.dumps(item["template_key"], sort_keys=True))
        shape_keys.add(json.dumps(item["metadata_shapes"], sort_keys=True))
        by_total.setdefault(total, []).append(group_id)
        rows.append({
            "group_id": group_id,
            "lengths": lengths,
            **item,
        })

    out_data = {
        "workload": args.workload,
        "arena_reqs": args.arena_reqs,
        "group_count": len(rows),
        "baseline_key_count": len(baseline_keys),
        "canonical_template_count": len(canonical_keys),
        "metadata_shape_count": len(shape_keys),
        "collapse_ratio": len(baseline_keys) / max(1, len(canonical_keys)),
        "totals_with_multiple_layouts": {
            str(total): ids
            for total, ids in by_total.items()
            if len(ids) > 1
        },
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "baseline_key_count": out_data["baseline_key_count"],
        "canonical_template_count": out_data["canonical_template_count"],
        "metadata_shape_count": out_data["metadata_shape_count"],
        "collapse_ratio": out_data["collapse_ratio"],
    }, indent=2))


if __name__ == "__main__":
    main()
