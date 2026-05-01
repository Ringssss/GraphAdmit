#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prefill_graph.runtime import (  # noqa: E402
    CaptureResult,
    LiveCaptureCallbacks,
    LiveTemplateSpec,
    ReplayResult,
    SameEngineLiveCaptureManager,
    ValidationResult,
    WorkloadDriftDetector,
    WorkloadObservation,
    residual_buckets_for_preset,
)


def load_tokens(path: str | Path, *, limit: int = 0) -> list[int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tokens: list[int] = []
    for req in data.get("requests", []):
        raw = req.get("actual_input_length", req.get("target_input_length", req.get("prompt_len")))
        if raw is not None:
            tokens.append(int(raw))
    return tokens[:limit] if limit else tokens


def template_for(tokens: int, buckets: list[int]) -> int | None:
    for bucket in buckets:
        if tokens <= bucket:
            return int(bucket)
    return None


class SyntheticGraphRuntime:
    def __init__(self, *, base_capture_size: int):
        self.base_capture_size = int(base_capture_size)

    def fallback_latency(self, tokens: int) -> float:
        return 8.0 + 0.065 * tokens

    def graph_latency(self, tokens: int, template_tokens: int) -> float:
        padding = max(0, template_tokens - tokens)
        if template_tokens <= self.base_capture_size:
            return self.fallback_latency(tokens) * 0.92
        # Extra templates save launch/metadata overhead but can lose on padding.
        return 7.0 + 0.041 * tokens + 0.018 * padding

    def static_policy_latency(self, tokens: int, template_tokens: int | None) -> tuple[str, float, bool]:
        if template_tokens is None or template_tokens > self.base_capture_size:
            return "fallback", self.fallback_latency(tokens), False
        return f"tokens={template_tokens}", self.graph_latency(tokens, template_tokens), True


def make_callbacks(runtime: SyntheticGraphRuntime) -> LiveCaptureCallbacks:
    def capture(spec: LiveTemplateSpec, ctx: dict) -> CaptureResult:
        return CaptureResult(
            captured=True,
            capture_ms=4.0,
            warmup_ms=2.0,
            memory_bytes=int((spec.template_tokens or 1) * 4096),
            handle={"template_tokens": spec.template_tokens},
        )

    def replay(spec: LiveTemplateSpec, ctx: dict) -> ReplayResult:
        tokens = int(ctx["tokens"])
        template = int(spec.template_tokens or tokens)
        return ReplayResult(
            output={"token": tokens % 997},
            latency_ms=runtime.graph_latency(tokens, template),
        )

    def fallback(spec: LiveTemplateSpec | None, ctx: dict) -> ReplayResult:
        tokens = int(ctx["tokens"])
        return ReplayResult(
            output={"token": tokens % 997},
            latency_ms=runtime.fallback_latency(tokens),
        )

    return LiveCaptureCallbacks(
        capture=capture,
        replay=replay,
        fallback=fallback,
        validate=lambda graph, fallback, ctx: ValidationResult(correct=graph == fallback),
    )


def summarize(rows: list[dict]) -> dict:
    lat = [float(row["latency_ms"]) for row in rows]
    if not lat:
        return {}
    ordered = sorted(lat)
    return {
        "n": len(rows),
        "avg_ms": sum(lat) / len(lat),
        "p50_ms": percentile(ordered, 50),
        "p95_ms": percentile(ordered, 95),
        "p99_ms": percentile(ordered, 99),
        "graph_used": sum(1 for row in rows if row.get("graph_used")),
        "useful": sum(1 for row in rows if row.get("useful")),
        "fallbacks": sum(1 for row in rows if not row.get("graph_used")),
    }


def percentile(ordered: list[float], pct: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a", required=True)
    parser.add_argument("--phase-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-a", type=int, default=0)
    parser.add_argument("--limit-b", type=int, default=0)
    parser.add_argument("--bucket-preset", default="sglang-pcg")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--base-capture-size", type=int, default=512)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--max-extra-templates", type=int, default=6)
    args = parser.parse_args()

    phase_a = load_tokens(args.phase_a, limit=args.limit_a)
    phase_b = load_tokens(args.phase_b, limit=args.limit_b)
    tokens = phase_a + phase_b
    buckets = residual_buckets_for_preset(args.bucket_preset, max_tokens=args.max_tokens)
    runtime = SyntheticGraphRuntime(base_capture_size=args.base_capture_size)
    manager = SameEngineLiveCaptureManager(
        min_samples=args.min_samples,
        min_useful_rate=0.67,
        min_saving_ms=0.5,
        max_p95_regression_ms=3.0,
        max_templates=args.max_extra_templates,
        max_graph_memory_bytes=64 * 1024 * 1024,
        validation_interval=8,
    )
    for bucket in buckets:
        if bucket <= args.base_capture_size:
            continue
        prev = max([b for b in buckets if b < bucket], default=args.base_capture_size)
        manager.register(
            LiveTemplateSpec(
                f"tokens={bucket}",
                lo=prev,
                hi=bucket,
                template_tokens=bucket,
                action="graph",
                fallback_action="fallback",
            )
        )
    callbacks = make_callbacks(runtime)
    drift = WorkloadDriftDetector(window=16, reference_window=64, min_samples=16)

    static_rows = []
    live_rows = []
    drift_events = []
    drift_actions = []
    for idx, tok in enumerate(tokens):
        bucket = template_for(tok, buckets)
        template_id, static_ms, static_graph = runtime.static_policy_latency(tok, bucket)
        fallback_ms = runtime.fallback_latency(tok)
        static_rows.append({
            "idx": idx,
            "tokens": tok,
            "template_id": template_id,
            "latency_ms": static_ms,
            "fallback_ms": fallback_ms,
            "graph_used": static_graph,
            "useful": static_graph and static_ms < fallback_ms,
        })

        result = manager.run({"tokens": tok}, callbacks)
        graph_used = result.action == "graph"
        live_ms = result.graph_ms if graph_used else result.fallback_ms
        if live_ms is None:
            live_ms = fallback_ms
        useful = bool(graph_used and live_ms < fallback_ms and result.correct is not False)
        decision = drift.observe(
            WorkloadObservation(
                tokens=tok,
                template_id=result.template_id,
                graph_used=graph_used,
                useful=useful,
                latency_ms=live_ms,
                fallback_ms=fallback_ms,
                correct=result.correct is not False,
                timestamp_ms=float(idx),
            )
        )
        if decision.drifted:
            recent_template_ids = [
                row.get("template_id")
                for row in live_rows[-drift.window:]
                if row.get("template_id")
            ]
            if result.template_id:
                recent_template_ids.append(result.template_id)
            affected = manager.apply_drift_decision(
                decision,
                recent_template_ids=recent_template_ids,
                reason_prefix="workload_drift",
            )
            event = {"idx": idx, **decision.__dict__, "affected_templates": affected}
            drift_events.append(event)
            drift_actions.append(event)
        live_rows.append({
            "idx": idx,
            "tokens": tok,
            "template_id": result.template_id,
            "action": result.action,
            "reason": result.reason,
            "latency_ms": live_ms,
            "fallback_ms": fallback_ms,
            "graph_used": graph_used,
            "useful": useful,
            "status": result.status,
        })

    payload = {
        "phase_a": args.phase_a,
        "phase_b": args.phase_b,
        "num_requests": len(tokens),
        "static_policy": summarize(static_rows),
        "same_engine_live_capture": summarize(live_rows),
        "speedup_live_vs_static": (
            summarize(static_rows)["avg_ms"] / summarize(live_rows)["avg_ms"]
            if live_rows else None
        ),
        "manager": manager.summary(include_events=False),
        "drift": drift.summary(),
        "drift_events": drift_events[:64],
        "drift_actions": drift_actions[:64],
        "rows": {
            "static": static_rows,
            "live": live_rows,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "speedup_live_vs_static": payload["speedup_live_vs_static"],
        "static_policy": payload["static_policy"],
        "same_engine_live_capture": payload["same_engine_live_capture"],
        "drift_events": len(drift_events),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
