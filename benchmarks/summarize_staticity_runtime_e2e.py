#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def bool_correct(result: dict[str, Any]) -> bool | None:
    if "all_same_outputs_vs_first" in result:
        return bool(result["all_same_outputs_vs_first"])
    values = result.get("same_outputs_vs_first")
    if isinstance(values, list):
        return all(bool(v) for v in values)
    if values is not None:
        return bool(values)
    return None


def summarize_dispatcher(path: str | Path | None) -> dict[str, Any]:
    rows = [row for row in read_jsonl(path) if row.get("kind") == "dispatch"]
    reason_counts = Counter(row.get("reason") for row in rows)
    mode_counts = Counter(row.get("mode") for row in rows)
    action_counts = Counter(row.get("staticity_runtime_action") for row in rows)
    runtime_rows = [row for row in rows if row.get("staticity_runtime_policy")]
    reject_counts = Counter()
    admitted_template_ids = set()
    req_values = []
    for row in runtime_rows:
        admission = row.get("staticity_runtime_admission") or {}
        if admission.get("template_id"):
            admitted_template_ids.add(admission["template_id"])
        if admission.get("reject_reason"):
            reject_counts[admission["reject_reason"]] += 1
        if admission.get("num_reqs") is not None:
            req_values.append(int(admission["num_reqs"]))
    return {
        "path": str(path) if path else None,
        "events": len(rows),
        "runtime_events": len(runtime_rows),
        "mode_counts": dict(mode_counts),
        "reason_counts": dict(reason_counts),
        "runtime_action_counts": dict(action_counts),
        "runtime_reject_counts": dict(reject_counts),
        "runtime_templates_seen": len(admitted_template_ids),
        "num_reqs": {
            "unique": sorted(set(req_values))[:32],
            "max": max(req_values) if req_values else None,
            "multi_req_events": sum(v > 1 for v in req_values),
        },
    }


def summarize_attention(path: str | Path | None) -> dict[str, Any]:
    rows = read_jsonl(path)
    records = []
    field_ptrs: dict[str, set[int]] = defaultdict(set)
    field_shapes: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for row in rows:
        if row.get("component") != "attention_metadata":
            continue
        records.append(row)
        for tensor in row.get("tensors", []):
            name = tensor.get("name")
            if not name:
                continue
            if tensor.get("data_ptr") is not None:
                field_ptrs[name].add(int(tensor["data_ptr"]))
            if tensor.get("shape") is not None:
                field_shapes[name].add(tuple(int(x) for x in tensor["shape"]))
    fields = {}
    for name in sorted(set(field_ptrs) | set(field_shapes)):
        fields[name] = {
            "unique_ptrs": len(field_ptrs.get(name, set())),
            "unique_shapes": len(field_shapes.get(name, set())),
            "shape_examples": [list(shape) for shape in sorted(field_shapes.get(name, set()))[:8]],
        }
    blockers = []
    for name, info in fields.items():
        if info["unique_ptrs"] > 1:
            blockers.append(f"{name}: address dynamic ({info['unique_ptrs']} ptrs)")
        if info["unique_shapes"] > 1:
            blockers.append(f"{name}: shape dynamic ({info['unique_shapes']} shapes)")
    return {
        "path": str(path) if path else None,
        "events": len(records),
        "fields": fields,
        "blockers": blockers,
    }


def summarize_scheduler(path: str | Path | None) -> dict[str, Any]:
    rows = read_jsonl(path)
    waits = [float(row.get("oldest_delay_ms", 0.0)) for row in rows]
    return {
        "path": str(path) if path else None,
        "events": len(rows),
        "enabled_events": sum(1 for row in rows if row.get("template_scheduler_enabled")),
        "promoted_events": sum(1 for row in rows if row.get("staticity_scheduler_promoted")),
        "max_reorder_count": max([int(row.get("reorder_count", 0)) for row in rows], default=0),
        "queue_waiting_max": max([int(row.get("queue_waiting", 0)) for row in rows], default=0),
        "oldest_delay_ms_p95": pct(waits, 95),
        "template_counts": dict(Counter(row.get("template") for row in rows)),
    }


def summarize_vllm(path: str | Path) -> dict[str, Any]:
    data = load_json(path)
    results = []
    baseline = data.get("results", [{}])[0]
    baseline_avg = baseline.get("avg_ms")
    baseline_p95 = baseline.get("p95_ms")
    for result in data.get("results", []):
        avg = result.get("avg_ms")
        p95 = result.get("p95_ms")
        results.append({
            "config": result.get("config"),
            "avg_ms": avg,
            "p50_ms": result.get("p50_ms"),
            "p95_ms": p95,
            "p99_ms": result.get("p99_ms"),
            "speedup_vs_first": result.get("speedup_vs_first"),
            "avg_speedup_vs_baseline": baseline_avg / avg if baseline_avg and avg else None,
            "p95_speedup_vs_baseline": baseline_p95 / p95 if baseline_p95 and p95 else None,
            "correct": bool_correct(result),
            "batch_mode": result.get("batch_mode"),
            "template_scheduler": result.get("template_scheduler"),
            "dispatcher": summarize_dispatcher(result.get("dispatcher_profile")),
            "attention": summarize_attention(result.get("attention_profile")),
            "scheduler": summarize_scheduler(result.get("scheduler_profile")),
        })
    best_correct = [
        row for row in results
        if row["correct"] is not False and row.get("avg_ms") is not None
    ]
    best = min(best_correct, key=lambda row: row["avg_ms"]) if best_correct else None
    return {
        "path": str(path),
        "partial": data.get("partial"),
        "model": data.get("model"),
        "workload": data.get("workload"),
        "workload_stats": data.get("workload_stats"),
        "best_correct_avg": best,
        "results": results,
    }


def summarize_dinfer(path: str | Path) -> dict[str, Any]:
    data = load_json(path)
    return {
        "path": str(path),
        "partial": data.get("partial"),
        "num_samples": data.get("num_samples"),
        "all_same_tokens": data.get("all_same_tokens"),
        "eager_total_s": data.get("eager_total_s"),
        "graph_total_s": data.get("graph_total_s"),
        "graph_total_with_cleanup_s": data.get("graph_total_with_cleanup_s"),
        "speedup": data.get("total_speedup"),
        "speedup_with_cleanup": data.get("total_speedup_with_cleanup"),
        "cleanup_count": data.get("cleanup_count"),
        "admission_policy": data.get("admission_policy"),
        "dynamicity": data.get("dynamicity"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", action="append", default=[])
    parser.add_argument("--dinfer", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = {
        "vllm": [summarize_vllm(path) for path in args.vllm],
        "dinfer": [summarize_dinfer(path) for path in args.dinfer],
    }

    highlights = []
    blockers = []
    for run in report["vllm"]:
        best = run.get("best_correct_avg")
        if best and best.get("avg_speedup_vs_baseline", 0) > 1.0:
            highlights.append({
                "kind": "vllm_avg_win",
                "model": run.get("model"),
                "speedup": best.get("avg_speedup_vs_baseline"),
                "config": best.get("config"),
            })
        for result in run.get("results", []):
            if result.get("correct") is False:
                blockers.append({
                    "kind": "correctness_failure",
                    "path": run.get("path"),
                    "config": result.get("config"),
                })
            sched = result.get("scheduler", {})
            if result.get("template_scheduler") and sched.get("max_reorder_count", 0) == 0:
                blockers.append({
                    "kind": "scheduler_not_exercised",
                    "path": run.get("path"),
                    "config": result.get("config"),
                })
            attn_blockers = result.get("attention", {}).get("blockers", [])
            if attn_blockers:
                blockers.append({
                    "kind": "metadata_dynamicity",
                    "path": run.get("path"),
                    "config": result.get("config"),
                    "blockers": attn_blockers[:8],
                })
    for run in report["dinfer"]:
        if run.get("all_same_tokens") and (run.get("speedup_with_cleanup") or 0) > 1.0:
            highlights.append({
                "kind": "dinfer_correct_speedup",
                "speedup_with_cleanup": run.get("speedup_with_cleanup"),
                "path": run.get("path"),
            })
        if not run.get("all_same_tokens"):
            blockers.append({"kind": "dinfer_token_mismatch", "path": run.get("path")})

    report["highlights"] = highlights
    report["blockers"] = blockers
    report["paper_readiness"] = {
        "planner_admission": "ready for a guarded-runtime claim",
        "metadata_arena": "not ready; evidence is profiling/PoC, not integrated key collapse",
        "scheduler": "not ready until true arrival-trace TTFT shows reorder benefit with correctness",
        "dinfer": "partial; correctness-aware speedup exists but persistent metadata arena is missing",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "highlights": highlights,
        "blockers": blockers[:8],
        "paper_readiness": report["paper_readiness"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
