#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_name(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("+", "_")
        .replace("-", "_")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
    )


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo))


def range_name(tok: int) -> str:
    ranges = [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192)]
    for lo, hi in ranges:
        if lo < tok <= hi:
            return f"({lo},{hi}]"
    return ">8192"


def get_result(data: dict[str, Any], contains: str | None = None, index: int | None = None) -> dict[str, Any]:
    results = data.get("results", [])
    if index is not None:
        return results[index]
    assert contains is not None
    for result in results:
        if contains.lower() in str(result.get("config", "")).lower():
            return result
    raise KeyError(f"no result contains {contains!r}")


def per_req(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = result.get("per_req", [])
    return rows if isinstance(rows, list) else []


def correctness_flags(candidate: dict[str, Any], n: int) -> tuple[list[bool], str]:
    values = candidate.get("same_outputs_vs_reference")
    if isinstance(values, list) and len(values) >= n:
        return [bool(v) for v in values[:n]], "same_outputs_vs_reference"
    if candidate.get("all_same_outputs_vs_reference") is not None:
        return [bool(candidate["all_same_outputs_vs_reference"])] * n, "all_same_outputs_vs_reference"
    values = candidate.get("same_outputs_vs_first")
    if isinstance(values, list) and len(values) >= n:
        return [bool(v) for v in values[:n]], "same_outputs_vs_first"
    if candidate.get("all_same_outputs_vs_first") is not None:
        return [bool(candidate["all_same_outputs_vs_first"])] * n, "all_same_outputs_vs_first"
    return [False] * n, "missing_fail_closed"


def compare_vllm(
    *,
    name: str,
    path: Path,
    baseline_contains: str,
    candidate_contains: str,
    category: str,
    description: str,
) -> dict[str, Any]:
    data = load_json(path)
    baseline = get_result(data, contains=baseline_contains)
    candidate = get_result(data, contains=candidate_contains)
    base_rows = per_req(baseline)
    cand_rows = per_req(candidate)
    n = min(len(base_rows), len(cand_rows))
    correct, correctness_source = correctness_flags(candidate, n)
    rows = []
    by_range: dict[str, list[dict[str, Any]]] = {}
    for idx in range(n):
        tok = int(cand_rows[idx].get("tok", base_rows[idx].get("tok", 0)))
        base_ms = float(base_rows[idx]["ms"])
        cand_ms = float(cand_rows[idx]["ms"])
        delta = cand_ms - base_ms
        row = {
            "idx": idx,
            "tok": tok,
            "range": range_name(tok),
            "baseline_ms": base_ms,
            "candidate_ms": cand_ms,
            "delta_ms": delta,
            "speedup": base_ms / cand_ms if cand_ms else None,
            "correct": bool(correct[idx]),
            "useful": bool(correct[idx] and cand_ms < base_ms),
            "negative": bool((not correct[idx]) or cand_ms >= base_ms),
        }
        rows.append(row)
        by_range.setdefault(row["range"], []).append(row)

    useful = [row for row in rows if row["useful"]]
    negative = [row for row in rows if row["negative"]]
    slower = [row for row in rows if row["correct"] and row["candidate_ms"] >= row["baseline_ms"]]
    mismatches = [row for row in rows if not row["correct"]]
    deltas = [row["delta_ms"] for row in rows]
    range_summary = []
    for rng, items in sorted(by_range.items()):
        range_summary.append({
            "range": rng,
            "n": len(items),
            "avg_baseline_ms": mean(row["baseline_ms"] for row in items),
            "avg_candidate_ms": mean(row["candidate_ms"] for row in items),
            "useful_rate": sum(row["useful"] for row in items) / len(items),
            "negative_rate": sum(row["negative"] for row in items) / len(items),
            "correct_rate": sum(row["correct"] for row in items) / len(items),
        })
    return {
        "name": name,
        "category": category,
        "description": description,
        "path": str(path),
        "baseline_config": baseline.get("config"),
        "candidate_config": candidate.get("config"),
        "correctness_source": correctness_source,
        "correctness_verified": correctness_source != "missing_fail_closed",
        "n": n,
        "baseline_avg_ms": baseline.get("avg_ms"),
        "candidate_avg_ms": candidate.get("avg_ms"),
        "speedup_avg": (
            baseline.get("avg_ms") / candidate.get("avg_ms")
            if baseline.get("avg_ms") and candidate.get("avg_ms")
            else None
        ),
        "baseline_p95_ms": baseline.get("p95_ms"),
        "candidate_p95_ms": candidate.get("p95_ms"),
        "baseline_p99_ms": baseline.get("p99_ms"),
        "candidate_p99_ms": candidate.get("p99_ms"),
        "useful_coverage": len(useful) / n if n else 0.0,
        "negative_graph_rate": len(negative) / n if n else 0.0,
        "slower_than_baseline_rate": len(slower) / n if n else 0.0,
        "correctness_mismatch_rate": len(mismatches) / n if n else 0.0,
        "median_delta_ms": median(deltas) if deltas else None,
        "p95_delta_ms": percentile(deltas, 95),
        "range_summary": range_summary,
        "worst_regressions": sorted(rows, key=lambda row: row["delta_ms"], reverse=True)[:8],
        "best_wins": sorted(rows, key=lambda row: row["delta_ms"])[:8],
    }


def summarize_dinfer_pair(name: str, path: Path, category: str, description: str) -> dict[str, Any]:
    data = load_json(path)
    rows = data.get("rows", [])
    useful = []
    negative = []
    mismatches = []
    slower = []
    records = []
    for row in rows:
        eager = float(row.get("eager_s", 0.0))
        graph = float(row.get("graph_s", 0.0))
        same = bool(row.get("same_tokens"))
        delta = graph - eager
        record = {
            "idx": int(row.get("idx", len(records))),
            "prompt_len": int(row.get("prompt_len", row.get("trace_target_len", 0))),
            "eager_s": eager,
            "graph_s": graph,
            "delta_s": delta,
            "speedup": eager / graph if graph else None,
            "correct": same,
            "useful": bool(same and graph < eager),
            "negative": bool((not same) or graph >= eager),
        }
        records.append(record)
        if record["useful"]:
            useful.append(record)
        if record["negative"]:
            negative.append(record)
        if not same:
            mismatches.append(record)
        if same and graph >= eager:
            slower.append(record)
    n = len(records)
    return {
        "name": name,
        "category": category,
        "description": description,
        "path": str(path),
        "n": n,
        "eager_total_s": data.get("eager_total_s"),
        "graph_total_s": data.get("graph_total_s"),
        "graph_total_with_cleanup_s": data.get("graph_total_with_cleanup_s"),
        "speedup_total": data.get("total_speedup"),
        "speedup_with_cleanup": data.get("total_speedup_with_cleanup"),
        "all_same_tokens": data.get("all_same_tokens"),
        "admission_policy": data.get("admission_policy"),
        "useful_coverage": len(useful) / n if n else 0.0,
        "negative_graph_rate": len(negative) / n if n else 0.0,
        "slower_than_baseline_rate": len(slower) / n if n else 0.0,
        "correctness_mismatch_rate": len(mismatches) / n if n else 0.0,
        "worst_regressions": sorted(records, key=lambda row: row["delta_s"], reverse=True)[:8],
        "best_wins": sorted(records, key=lambda row: row["delta_s"])[:8],
    }


def summarize_keycollapse(path: Path) -> dict[str, Any]:
    data = load_json(path)
    baseline = data["baseline_summary"]
    candidate = data["candidate_summary"]

    def collapse_stats(summary: dict[str, Any]) -> dict[str, Any]:
        candidates = summary.get("collapse_candidates", [])
        variants = [int(item.get("dispatch_key_variants", 0)) for item in candidates]
        layouts = [int(item.get("layout_count", 0)) for item in candidates]
        reasons = Counter()
        for item in candidates:
            reasons.update(item.get("dispatch_reasons", {}))
        return {
            "collapse_candidate_count": len(candidates),
            "avg_dispatch_key_variants": mean(variants) if variants else 0.0,
            "max_dispatch_key_variants": max(variants) if variants else 0,
            "avg_layout_count": mean(layouts) if layouts else 0.0,
            "reason_counts": dict(reasons),
            "diagnosis": summary.get("diagnosis", {}),
        }

    base = collapse_stats(baseline)
    cand = collapse_stats(candidate)
    base_variants = base["avg_dispatch_key_variants"]
    cand_variants = cand["avg_dispatch_key_variants"]
    return {
        "path": str(path),
        "claim_checks": data.get("claim_checks", {}),
        "baseline": base,
        "candidate": cand,
        "avg_key_variant_reduction": (
            base_variants / cand_variants if cand_variants else None
        ),
    }


def svg_bar_chart(title: str, rows: list[dict[str, Any]], fields: list[tuple[str, str]], output: Path) -> None:
    width = 920
    row_h = 42
    group_gap = 18
    label_w = 300
    bar_w = 520
    top = 70
    height = top + len(rows) * (len(fields) * row_h + group_gap) + 40
    colors = {
        "useful_coverage": "#2f9e44",
        "negative_graph_rate": "#e03131",
        "slower_than_baseline_rate": "#f08c00",
        "correctness_mismatch_rate": "#7048e8",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{label_w}" y1="{top-20}" x2="{label_w+bar_w}" y2="{top-20}" stroke="#333"/>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = label_w + tick * bar_w
        parts.append(f'<line x1="{x:.1f}" y1="{top-25}" x2="{x:.1f}" y2="{height-30}" stroke="#eee"/>')
        parts.append(f'<text x="{x-8:.1f}" y="{top-30}" font-family="Arial" font-size="12">{int(tick*100)}%</text>')
    y = top
    for row in rows:
        parts.append(f'<text x="24" y="{y+16}" font-family="Arial" font-size="14" font-weight="700">{html.escape(row["name"])}</text>')
        for field, label in fields:
            val = float(row.get(field, 0.0) or 0.0)
            yy = y + 24
            parts.append(f'<text x="44" y="{yy+16}" font-family="Arial" font-size="13">{html.escape(label)}</text>')
            parts.append(f'<rect x="{label_w}" y="{yy+2}" width="{bar_w}" height="20" fill="#f1f3f5"/>')
            parts.append(
                f'<rect x="{label_w}" y="{yy+2}" width="{max(0.0, min(1.0, val))*bar_w:.1f}" '
                f'height="20" fill="{colors.get(field, "#1971c2")}"/>'
            )
            parts.append(
                f'<text x="{label_w+bar_w+10}" y="{yy+17}" font-family="Arial" font-size="13">{val*100:.1f}%</text>'
            )
            y += row_h
        y += group_gap
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def svg_latency_chart(title: str, rows: list[dict[str, Any]], output: Path) -> None:
    width = 900
    height = 420
    margin_l = 230
    margin_t = 60
    bar_h = 18
    gap = 30
    max_val = max(
        [
            float(row.get("baseline_avg_ms") or row.get("eager_total_s") or 0.0)
            for row in rows
        ] + [
            float(row.get("candidate_avg_ms") or row.get("graph_total_s") or 0.0)
            for row in rows
        ] + [1.0]
    )
    plot_w = 560
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700">{html.escape(title)}</text>',
    ]
    y = margin_t
    for row in rows:
        base = float(row.get("baseline_avg_ms") or row.get("eager_total_s") or 0.0)
        cand = float(row.get("candidate_avg_ms") or row.get("graph_total_s") or 0.0)
        unit = "ms" if "baseline_avg_ms" in row else "s"
        parts.append(f'<text x="24" y="{y+17}" font-family="Arial" font-size="14" font-weight="700">{html.escape(row["name"])}</text>')
        for label, val, color in [("baseline", base, "#868e96"), ("candidate", cand, "#1971c2")]:
            yy = y + (24 if label == "baseline" else 49)
            parts.append(f'<text x="44" y="{yy+14}" font-family="Arial" font-size="13">{label}</text>')
            parts.append(f'<rect x="{margin_l}" y="{yy}" width="{val/max_val*plot_w:.1f}" height="{bar_h}" fill="{color}"/>')
            parts.append(f'<text x="{margin_l+val/max_val*plot_w+8:.1f}" y="{yy+14}" font-family="Arial" font-size="13">{val:.2f} {unit}</text>')
        y += 2 * bar_h + gap + 40
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# CUDA Graph Failure and Useful-Coverage Analysis",
        "",
        "## Key Definitions",
        "",
        "- `negative_graph_rate`: fraction of requests where the graph/static path is slower than the baseline or incorrect.",
        "- `slower_than_baseline_rate`: fraction of token-correct requests where graph/static latency is >= baseline latency.",
        "- `correctness_mismatch_rate`: fraction of requests with token mismatch.",
        "- `useful_coverage`: fraction of requests where graph/static path is both token-correct and faster than the fallback/baseline.",
        "",
        "## vLLM Comparisons",
        "",
        "| Case | Baseline | Candidate | Avg Speedup | Useful Coverage | Negative Rate | Slower Rate | Mismatch Rate |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["vllm_cases"]:
        lines.append(
            "| {name} | {base} | {cand} | {speed:.3f}x | {useful:.1%} | {neg:.1%} | {slow:.1%} | {mis:.1%} |".format(
                name=item["name"],
                base=item["baseline_config"],
                cand=item["candidate_config"],
                speed=item.get("speedup_avg") or 0.0,
                useful=item["useful_coverage"],
                neg=item["negative_graph_rate"],
                slow=item["slower_than_baseline_rate"],
                mis=item["correctness_mismatch_rate"],
            )
        )
    lines.extend([
        "",
        "## dInfer Comparisons",
        "",
        "| Case | Speedup | Useful Coverage | Negative Rate | Slower Rate | Mismatch Rate | All Same Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for item in report["dinfer_cases"]:
        lines.append(
            "| {name} | {speed:.3f}x | {useful:.1%} | {neg:.1%} | {slow:.1%} | {mis:.1%} | {same} |".format(
                name=item["name"],
                speed=item.get("speedup_total") or 0.0,
                useful=item["useful_coverage"],
                neg=item["negative_graph_rate"],
                slow=item["slower_than_baseline_rate"],
                mis=item["correctness_mismatch_rate"],
                same=item.get("all_same_tokens"),
            )
        )
    kc = report["keycollapse"]
    lines.extend([
        "",
        "## Key-Collapse Evidence",
        "",
        f"- Claim checks: `{json.dumps(kc['claim_checks'], ensure_ascii=False)}`",
        f"- Baseline avg dispatch key variants per collapse candidate: `{kc['baseline']['avg_dispatch_key_variants']:.2f}`",
        f"- Candidate avg dispatch key variants per collapse candidate: `{kc['candidate']['avg_dispatch_key_variants']:.2f}`",
        f"- Avg key-variant reduction: `{kc['avg_key_variant_reduction']:.2f}x`",
        "",
        "## Figures",
        "",
        "- `cuda_graph_failure_rates.svg`: negative/useful/correctness rates.",
        "- `cuda_graph_avg_latency.svg`: baseline vs candidate average latency.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/cg_failure_analysis")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vllm_cases = [
        compare_vllm(
            name="8B safe CG vs eager",
            path=Path("results/vllm_qwentrace_llama3_8b_64_safe.json"),
            baseline_contains="Eager",
            candidate_contains="vLLM graph max512 no-CP",
            category="existing_cg",
            description="Default vLLM CUDA Graph improves short/mid requests but still has per-request negatives.",
        ),
        compare_vllm(
            name="long over-capture CG failure",
            path=Path("results/vllm_flowprefill_morspec_8_long_hybrid.json"),
            baseline_contains="vLLM graph max512 CP",
            candidate_contains="Ours hybrid max4096 no-CP",
            category="bad_graph",
            description="Naive long-bucket over-capture is slower than CP+CG because long prefill is compute/padding dominated.",
        ),
        compare_vllm(
            name="235B aggressive arena failure",
            path=Path("results/vllm_qwen3_235b_16_online_arena_windowed_e2e.json"),
            baseline_contains="vLLM graph max512 CP FULL",
            candidate_contains="Single-engine runtime",
            category="bad_admission",
            description="Fixed arena/key-collapse without latency-aware admission regresses short MoE requests.",
        ),
        compare_vllm(
            name="235B admitted mid832",
            path=Path("results/vllm_qwen3_235b_16_mid832_tokenaxis_fixed_e2e.json"),
            baseline_contains="vLLM graph max512 CP",
            candidate_contains="Single-engine runtime",
            category="ours_admission",
            description="Latency-aware admitted 832-token template avoids short/long unstable ranges.",
        ),
        compare_vllm(
            name="235B learned useful-admission",
            path=Path("results/vllm_qwen3_235b_16_useful_auto_fap_e2e.json"),
            baseline_contains="vLLM graph max512 CP",
            candidate_contains="Single-engine runtime",
            category="ours_learned_admission",
            description="Useful-coverage policy learned the profitable 832-token window from measured latency/correctness.",
        ),
        compare_vllm(
            name="235B online refreshed single-engine",
            path=Path("results/vllm_qwen3_235b_16_online_refreshed_e2e.json"),
            baseline_contains="vLLM graph max512 CP",
            candidate_contains="Single-engine runtime",
            category="ours_online_admission",
            description=(
                "Repo-materialized single-engine runtime with fixed-address arena, key collapse, "
                "and online-refreshed admission for the stable 744-805 token window."
            ),
        ),
        compare_vllm(
            name="235B control-plane current",
            path=Path("results/vllm_qwen3_235b_16_control_plane_e2e.json"),
            baseline_contains="vLLM graph max512 CP",
            candidate_contains="Single-engine runtime",
            category="current_control_plane",
            description=(
                "Current control-plane run with fail-closed admission, fixed metadata arena, "
                "key collapse, and MoE routing metadata capture enabled."
            ),
        ),
        compare_vllm(
            name="32B windowed arena",
            path=Path("results/vllm_qwen3_32b_64_online_arena_windowed_e2e.json"),
            baseline_contains="vLLM graph max512 CP FULL",
            candidate_contains="Single-engine runtime",
            category="ours_admission",
            description="Windowed fixed-address arena/key-collapse improves average latency but not tail.",
        ),
        compare_vllm(
            name="32B learned useful-admission",
            path=Path("results/vllm_qwen3_32b_64_useful_auto_e2e.json"),
            baseline_contains="vLLM graph max512 CP",
            candidate_contains="Single-engine runtime",
            category="ours_learned_admission",
            description=(
                "Useful-coverage policy admits only measured-profitable 832/1024 token windows; "
                "it improves the dense 512-1024 range and rejects short/long requests."
            ),
        ),
        compare_vllm(
            name="32B online refreshed single-engine",
            path=Path("results/vllm_qwen3_32b_64_online_refreshed_e2e.json"),
            baseline_contains="vLLM graph max512 CP",
            candidate_contains="Single-engine runtime",
            category="ours_online_admission",
            description=(
                "Repo-materialized single-engine runtime with fixed-address arena, key collapse, "
                "and online-refreshed admission for the stable 858-1012 token window."
            ),
        ),
        compare_vllm(
            name="32B control-plane current",
            path=Path("results/vllm_qwen3_32b_64_control_plane_e2e.json"),
            baseline_contains="vLLM graph max512 CP",
            candidate_contains="Single-engine runtime",
            category="current_control_plane",
            description=(
                "Current dense control-plane run with fail-closed admission, fixed metadata arena, "
                "and 1024-token graph family."
            ),
        ),
    ]

    dinfer_cases = [
        summarize_dinfer_pair(
            "dInfer unsafe graph",
            Path("results/dinfer_qwentrace_llada2_mini_4_graph.json"),
            "correctness_failure",
            "Unvalidated graph is fast but token-incorrect.",
        ),
        summarize_dinfer_pair(
            "dInfer validated every replay",
            Path("results/dinfer_qwentrace_llada2_mini_2_graph_validated.json"),
            "validation_overhead",
            "Correctness validation without admission can be slower than eager.",
        ),
        summarize_dinfer_pair(
            "dInfer admitted runtime",
            Path("results/dinfer_qwentrace_llada2_mini_32_runtime_admission_memguard_final_0428.json"),
            "ours_admission",
            "Decoded-token admission plus memory/template guards yields useful correct graph coverage.",
        ),
        summarize_dinfer_pair(
            "dInfer control-plane current",
            Path("results/dinfer_qwentrace_llada2_mini_32_control_plane_e2e.json"),
            "current_control_plane",
            "Current decoded-token validation/admission run with memory guard cleanup.",
        ),
    ]

    report = {
        "vllm_cases": vllm_cases,
        "dinfer_cases": dinfer_cases,
        "keycollapse": summarize_keycollapse(Path("results/kc_qwen32_fixed_arena_compare.json")),
        "interpretation": {
            "coverage_shortfall_sources": [
                "shape_dynamicity: total tokens, batch size, sequence length, expert token counts",
                "metadata/address_dynamicity: slot_mapping, positions, query_start_loc, seq_lens, block_table, KV/cache pointers",
                "graph_key_explosion: num_reqs/layout/mode/features enter the dispatch key and split graph families",
                "semantic/control_dynamicity: MoE routing, diffusion mask/update sets, early exit, sampling branches",
                "scheduler_dynamicity: natural batches do not align with profitable templates",
            ],
            "main_claim": (
                "The problem is not just insufficient raw graph hit rate. The problem is insufficient useful graph coverage: "
                "many potential graph executions are either unsafe, slower than fallback, or split across metadata/layout keys."
            ),
        },
    }

    (out_dir / "cuda_graph_failure_useful_coverage.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "cuda_graph_failure_useful_coverage.md").write_text(
        markdown_report(report),
        encoding="utf-8",
    )
    chart_rows = [
        {
            "name": item["name"],
            "useful_coverage": item["useful_coverage"],
            "negative_graph_rate": item["negative_graph_rate"],
            "slower_than_baseline_rate": item["slower_than_baseline_rate"],
            "correctness_mismatch_rate": item["correctness_mismatch_rate"],
        }
        for item in vllm_cases + dinfer_cases
    ]
    svg_bar_chart(
        "Useful CUDA Graph Coverage vs Failure Modes",
        chart_rows,
        [
            ("useful_coverage", "useful coverage"),
            ("negative_graph_rate", "negative rate"),
            ("correctness_mismatch_rate", "mismatch rate"),
        ],
        out_dir / "cuda_graph_failure_rates.svg",
    )
    latency_rows = []
    for item in vllm_cases:
        latency_rows.append({
            "name": item["name"],
            "baseline_avg_ms": item.get("baseline_avg_ms") or 0.0,
            "candidate_avg_ms": item.get("candidate_avg_ms") or 0.0,
        })
    for item in dinfer_cases:
        latency_rows.append({
            "name": item["name"],
            "eager_total_s": item.get("eager_total_s") or 0.0,
            "graph_total_s": item.get("graph_total_s") or 0.0,
        })
    svg_latency_chart(
        "Baseline vs Candidate Average Latency",
        latency_rows,
        out_dir / "cuda_graph_avg_latency.svg",
    )
    print(json.dumps({
        "output_dir": str(out_dir),
        "json": str(out_dir / "cuda_graph_failure_useful_coverage.json"),
        "markdown": str(out_dir / "cuda_graph_failure_useful_coverage.md"),
        "figures": [
            str(out_dir / "cuda_graph_failure_rates.svg"),
            str(out_dir / "cuda_graph_avg_latency.svg"),
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
