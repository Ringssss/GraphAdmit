#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQ_RE = re.compile(r"num_reqs=(None|\d+)")
TOK_RE = re.compile(r"num_tokens=(\d+)")


def read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_batch_descriptor(desc: Any) -> dict[str, Any]:
    if isinstance(desc, dict):
        return {
            "num_tokens": desc.get("num_tokens"),
            "num_reqs": desc.get("num_reqs"),
            "uniform": desc.get("uniform"),
        }
    text = str(desc)
    tok = TOK_RE.search(text)
    req = REQ_RE.search(text)
    return {
        "num_tokens": int(tok.group(1)) if tok else None,
        "num_reqs": None if not req or req.group(1) == "None" else int(req.group(1)),
        "uniform": "uniform=True" in text,
    }


def summarize_metadata_fields(attn_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_shapes: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    tensor_ptrs: dict[str, set[int]] = defaultdict(set)
    tensor_ptrs_by_device: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in attn_rows:
        for tensor in row.get("tensors", []):
            name = tensor.get("name")
            if not name:
                continue
            if tensor.get("shape") is not None:
                tensor_shapes[name].add(tuple(int(x) for x in tensor["shape"]))
            if tensor.get("data_ptr") is not None:
                tensor_ptrs[name].add(int(tensor["data_ptr"]))
                tensor_ptrs_by_device[name][str(tensor.get("device"))].add(
                    int(tensor["data_ptr"])
                )

    metadata_fields = {}
    for name in sorted(set(tensor_shapes) | set(tensor_ptrs)):
        ptrs_by_device = tensor_ptrs_by_device.get(name, {})
        metadata_fields[name] = {
            "unique_shapes": len(tensor_shapes[name]),
            "shape_examples": [list(x) for x in sorted(tensor_shapes[name])[:8]],
            "unique_ptrs": len(tensor_ptrs[name]),
            "unique_devices": len(ptrs_by_device),
            "max_unique_ptrs_per_device": (
                max((len(ptrs) for ptrs in ptrs_by_device.values()), default=0)
            ),
        }
    return metadata_fields


def diagnose_metadata(metadata_fields: dict[str, Any]) -> dict[str, Any]:
    request_metadata_names = {
        "query_start_loc",
        "query_start_loc_cpu",
        "seq_lens",
        "block_table_gid_0",
        "is_prefilling",
    }
    request_metadata_infos = [
        metadata_fields[name] for name in sorted(request_metadata_names)
        if name in metadata_fields
    ]
    request_metadata_shapes_fixed = bool(request_metadata_infos) and all(
        info["unique_shapes"] <= 1 for info in request_metadata_infos
    )
    request_metadata_ptrs_fixed = bool(request_metadata_infos) and all(
        info["max_unique_ptrs_per_device"] <= 1
        or (
            info["unique_devices"] == 1
            and info["unique_ptrs"] <= max(
                1,
                metadata_fields.get("query_start_loc", {}).get(
                    "unique_devices", 1
                ),
            )
        )
        for info in request_metadata_infos
    )
    slot_mapping_info = metadata_fields.get("slot_mapping_gid_0", {})
    token_shape_dynamic = slot_mapping_info.get("unique_shapes", 0) > 1
    return {
        "request_metadata_arena_active": (
            request_metadata_shapes_fixed and request_metadata_ptrs_fixed
        ),
        "request_metadata_shapes_fixed": request_metadata_shapes_fixed,
        "request_metadata_ptrs_fixed": request_metadata_ptrs_fixed,
        "token_shape_dynamic": token_shape_dynamic,
        "metadata_arena_needed": not (
            request_metadata_shapes_fixed and request_metadata_ptrs_fixed
        ),
    }


def select_policy_window_arena_rows(
    dispatch_rows: list[dict[str, Any]],
    attn_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    policy_ts = [
        float(row["ts"]) for row in dispatch_rows
        if row.get("staticity_runtime_policy")
        and row.get("staticity_runtime_action") in {"ours", "ours_cp"}
        and row.get("ts") is not None
    ]
    if not policy_ts:
        return []
    lo = min(policy_ts) - 0.1
    hi = max(policy_ts) + 0.1
    return [
        row for row in attn_rows
        if lo <= float(row.get("ts", 0.0)) <= hi
        and row.get("staticity_fixed_metadata_arena", False)
        and not row.get("for_cudagraph_capture", False)
    ]


def summarize(
    dispatcher: str | None,
    runner: str | None,
    attention: str | None,
    serving_only: bool = False,
) -> dict[str, Any]:
    dispatch_rows = [
        row for row in read_jsonl(dispatcher)
        if row.get("kind") == "dispatch"
    ]
    runner_rows = read_jsonl(runner)
    attn_rows_all = [
        row for row in read_jsonl(attention)
        if row.get("component") == "attention_metadata"
    ]
    attn_rows = attn_rows_all
    if serving_only:
        attn_rows = [
            row for row in attn_rows
            if not row.get("for_cudagraph_capture", False)
            and row.get("num_reqs") not in (None, 512)
        ]

    by_tokens: dict[int, set[tuple[Any, Any, Any]]] = defaultdict(set)
    reason_by_tokens: dict[int, Counter] = defaultdict(Counter)
    for row in dispatch_rows:
        desc = parse_batch_descriptor(row.get("batch_descriptor"))
        tokens = desc.get("num_tokens") or row.get("requested_num_tokens")
        if tokens is None:
            continue
        key = (
            row.get("mode"),
            desc.get("num_reqs"),
            desc.get("uniform"),
        )
        by_tokens[int(tokens)].add(key)
        reason_by_tokens[int(tokens)][row.get("reason")] += 1

    layout_by_total = defaultdict(set)
    for row in runner_rows:
        if row.get("component") != "gpu_model_runner_padding":
            continue
        if serving_only and (
            row.get("force_eager", False)
            or row.get("phase") != "initial_dispatch"
            or row.get("num_reqs") == 512
        ):
            continue
        total = row.get("num_tokens_after_sp_padding") or row.get("num_tokens_after_dp_padding")
        if total is None:
            continue
        layout = tuple(int(x) for x in row.get("num_scheduled_tokens", []))
        layout_by_total[int(total)].add(layout)

    metadata_fields = summarize_metadata_fields(attn_rows)
    arena_attn_rows = [
        row for row in attn_rows
        if row.get("staticity_fixed_metadata_arena", False)
    ]
    policy_window_arena_attn_rows = select_policy_window_arena_rows(
        dispatch_rows, attn_rows)
    arena_metadata_fields = summarize_metadata_fields(arena_attn_rows)
    policy_window_arena_metadata_fields = summarize_metadata_fields(
        policy_window_arena_attn_rows)

    collapse_candidates = []
    for tokens, layouts in sorted(layout_by_total.items()):
        if len(layouts) <= 1:
            continue
        collapse_candidates.append({
            "tokens": tokens,
            "layout_count": len(layouts),
            "layout_examples": [list(x) for x in sorted(layouts)[:5]],
            "dispatch_key_variants": len(by_tokens.get(tokens, set())),
            "dispatch_reasons": dict(reason_by_tokens.get(tokens, Counter())),
        })

    overall_metadata_diagnosis = diagnose_metadata(metadata_fields)
    arena_metadata_diagnosis = diagnose_metadata(arena_metadata_fields)
    policy_window_arena_metadata_diagnosis = diagnose_metadata(
        policy_window_arena_metadata_fields)

    return {
        "dispatcher": dispatcher,
        "runner": runner,
        "attention": attention,
        "dispatch_events": len(dispatch_rows),
        "runner_events": len(runner_rows),
        "attention_events": len(attn_rows),
        "attention_events_all": len(attn_rows_all),
        "arena_attention_events": len(arena_attn_rows),
        "policy_window_arena_attention_events": len(
            policy_window_arena_attn_rows),
        "serving_only": serving_only,
        "tokens_with_multiple_layouts": len(collapse_candidates),
        "collapse_candidates": collapse_candidates[:32],
        "metadata_fields": metadata_fields,
        "arena_metadata_fields": arena_metadata_fields,
        "policy_window_arena_metadata_fields": (
            policy_window_arena_metadata_fields
        ),
        "metadata_diagnosis": overall_metadata_diagnosis,
        "arena_metadata_diagnosis": arena_metadata_diagnosis,
        "policy_window_arena_metadata_diagnosis": (
            policy_window_arena_metadata_diagnosis
        ),
        "diagnosis": {
            "key_collapse_opportunity": len(collapse_candidates) > 0,
            "request_metadata_arena_active": overall_metadata_diagnosis[
                "request_metadata_arena_active"
            ],
            "arena_window_metadata_arena_active": arena_metadata_diagnosis[
                "request_metadata_arena_active"
            ],
            "policy_window_arena_active": (
                policy_window_arena_metadata_diagnosis[
                    "request_metadata_arena_active"
                ]
            ),
            "request_metadata_shapes_fixed": overall_metadata_diagnosis[
                "request_metadata_shapes_fixed"
            ],
            "request_metadata_ptrs_fixed": overall_metadata_diagnosis[
                "request_metadata_ptrs_fixed"
            ],
            "arena_window_request_metadata_shapes_fixed": arena_metadata_diagnosis[
                "request_metadata_shapes_fixed"
            ],
            "arena_window_request_metadata_ptrs_fixed": arena_metadata_diagnosis[
                "request_metadata_ptrs_fixed"
            ],
            "policy_window_request_metadata_shapes_fixed": (
                policy_window_arena_metadata_diagnosis[
                    "request_metadata_shapes_fixed"
                ]
            ),
            "policy_window_request_metadata_ptrs_fixed": (
                policy_window_arena_metadata_diagnosis[
                    "request_metadata_ptrs_fixed"
                ]
            ),
            "token_shape_dynamic": overall_metadata_diagnosis["token_shape_dynamic"],
            "arena_window_token_shape_dynamic": arena_metadata_diagnosis[
                "token_shape_dynamic"
            ],
            "policy_window_token_shape_dynamic": (
                policy_window_arena_metadata_diagnosis["token_shape_dynamic"]
            ),
            "metadata_arena_needed": overall_metadata_diagnosis[
                "metadata_arena_needed"
            ],
            "num_reqs_removed_for_piecewise": any(
                any(key[1] is None for key in variants)
                for variants in by_tokens.values()
            ),
            "remaining_blocker": (
                "MoE expert-routing templates and effective scheduler remain"
                if policy_window_arena_metadata_diagnosis[
                    "request_metadata_arena_active"
                ] and not policy_window_arena_metadata_diagnosis[
                    "token_shape_dynamic"
                ] else
                "token-shape dynamics still require bucket/admission"
                if arena_metadata_diagnosis["request_metadata_arena_active"]
                and arena_metadata_diagnosis["token_shape_dynamic"] else
                "multi-request correctness guard requires integrated metadata arena"
                if len(collapse_candidates) > 0
                and not arena_metadata_diagnosis["request_metadata_arena_active"] else
                "trace did not expose same-token multi-layout opportunity"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatcher")
    parser.add_argument("--runner")
    parser.add_argument("--attention")
    parser.add_argument("--serving-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = summarize(
        args.dispatcher,
        args.runner,
        args.attention,
        serving_only=args.serving_only,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "diagnosis": report["diagnosis"],
        "tokens_with_multiple_layouts": report["tokens_with_multiple_layouts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
