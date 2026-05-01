#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prefill_graph.runtime import (
    ArenaTemplateRegistry,
    CaptureResult,
    DriftDecision,
    StaticityControlPlane,
    TemplateLifecycle,
    ExpertMetadataCanonicalizer,
    ExpertTrafficTemplate,
    LiveCaptureCallbacks,
    LiveTemplateSpec,
    LiveTemplateStatus,
    MoEDispatchTemplate,
    MoEDispatchTemplateRegistry,
    OnlineSelfLearningAdmissionController,
    PartialGraphTemplateManager,
    ReplayResult,
    SameEngineLiveCaptureManager,
    SlaAwareTemplateScheduler,
    TemplateSchedulingSignal,
    TokenAxisCanonicalizer,
    TokenAxisTemplate,
    ValidationResult,
    WorkloadDriftDetector,
    WorkloadObservation,
    attention_segment_for_mode,
    function_branch_manifest,
    moe_dispatch_manifest,
    moe_segment_for_capacity,
    token_prefill_manifest,
)


def validate_online_admission() -> dict[str, object]:
    controller = OnlineSelfLearningAdmissionController(
        min_samples=3,
        min_useful_rate=0.75,
        min_saving_ms=1.0,
        max_p95_regression_ms=2.0,
        amortization_replays=16,
    )
    observations = [
        (52.0, 70.0),
        (51.0, 69.0),
        (53.0, 71.0),
        (54.0, 70.0),
    ]
    decisions = []
    for graph_ms, fallback_ms in observations:
        decisions.append(
            controller.observe(
                "tokens=832",
                graph_ms=graph_ms,
                fallback_ms=fallback_ms,
                correct=True,
                capture_ms=4.0,
                warmup_ms=2.0,
                metadata={"lo": 744, "hi": 805, "template_tokens": 832, "action": "ours_cp"},
            ).__dict__
        )
    controller.observe(
        "bad=unsafe",
        graph_ms=30.0,
        fallback_ms=60.0,
        correct=False,
        metadata={"lo": 1, "hi": 128, "template_tokens": 128},
    )
    return {
        "decisions": decisions,
        "policy": controller.export_runtime_policy(default_action="cp"),
        "summary": controller.summary(include_decisions=False),
    }


def validate_token_axis() -> dict[str, object]:
    template = TokenAxisTemplate(
        template_tokens=832,
        max_reqs=8,
        min_tokens=744,
        max_tokens=805,
    )
    canonicalizer = TokenAxisCanonicalizer(template)
    positions = list(range(750))
    slot_mapping = list(range(1000, 1750))
    result = canonicalizer.canonicalize(positions=positions, slot_mapping=slot_mapping)
    return {
        "template_rule": template.rule(),
        "original_tokens": result["original_tokens"],
        "template_tokens": result["template_tokens"],
        "active_tokens": sum(result["token_active_mask"]),
        "padded_tokens": len(result["token_active_mask"]) - sum(result["token_active_mask"]),
        "padded_slot_value": result["slot_mapping"][-1],
    }


def validate_moe() -> dict[str, object]:
    template = ExpertTrafficTemplate(capacity_bucket=4, max_experts=4, top_k=1)
    canonicalizer = ExpertMetadataCanonicalizer(template, max_tokens=16)
    expert_ids = [0, 1, 1, 2, 3, 3, 3]
    expert_counts = [1, 2, 1, 3]
    accepted = canonicalizer.canonicalize(
        expert_ids=expert_ids,
        expert_counts=expert_counts,
    )
    rejected = canonicalizer.canonicalize(
        expert_ids=[0] * 8,
        expert_counts=[8, 0, 0, 0],
    )
    return {
        "accepted": {
            "accepted": accepted["accepted"],
            "template_id": accepted["template_id"],
            "expert_counts": accepted["expert_counts"],
            "active_experts": sum(accepted["expert_active_mask"]),
        },
        "rejected": rejected,
    }


def validate_moe_dispatch_templates() -> dict[str, object]:
    admission = OnlineSelfLearningAdmissionController(
        min_samples=2,
        min_useful_rate=1.0,
        min_saving_ms=1.0,
        max_p95_regression_ms=1.0,
    )
    template = MoEDispatchTemplate(
        "moe:cap4:topk1",
        capacity_bucket=4,
        max_experts=4,
        top_k=1,
        action="moe_graph",
        fallback_action="compiled",
        max_tokens=16,
    )
    registry = MoEDispatchTemplateRegistry([template], admission=admission)
    before = registry.decide(
        expert_ids=[0, 1, 1, 2],
        expert_counts=[1, 2, 1, 0],
        tokens=4,
        top_k=1,
    ).__dict__
    registry.observe(
        template.template_id,
        graph_ms=8.0,
        fallback_ms=12.0,
        correct=True,
        metadata={"capacity_bucket": 4},
    )
    registry.observe(
        template.template_id,
        graph_ms=7.0,
        fallback_ms=12.0,
        correct=True,
        metadata={"capacity_bucket": 4},
    )
    after = registry.decide(
        expert_ids=[0, 1, 1, 2],
        expert_counts=[1, 2, 1, 0],
        tokens=4,
        top_k=1,
    ).__dict__
    overflow = registry.decide(
        expert_ids=[0] * 8,
        expert_counts=[8, 0, 0, 0],
        tokens=8,
        top_k=1,
    ).__dict__
    return {
        "before_admission": before,
        "after_admission": after,
        "overflow": overflow,
        "summary": registry.summary(),
    }


def validate_partial_graph() -> dict[str, object]:
    manager = PartialGraphTemplateManager(default_fallback="compiled")
    attn = attention_segment_for_mode("sliding_window", template_tokens=1024)
    moe = moe_segment_for_capacity(64, top_k=1)
    diffusion = function_branch_manifest(
        "diffusion:mask_low:tokens=1024",
        function_name="diffusion",
        branch_field="branch",
        branch_value="mask_low",
        max_tokens=1024,
    )
    manager.register(
        attn,
        guard=lambda ctx: ctx.get("attention_mode") == "sliding_window"
        and int(ctx.get("num_tokens", 0)) <= 1024,
        admitted=True,
    )
    manager.register(
        moe,
        guard=lambda ctx: int(ctx.get("max_expert_count", 10**9)) <= 64,
        admitted=False,
    )
    manager.register(
        attention_segment_for_mode("full", template_tokens=1024),
        guard=lambda ctx: ctx.get("attention_mode") == "full"
        and int(ctx.get("num_tokens", 0)) <= 1024,
        admitted=True,
    )
    decisions = [
        manager.decide({"attention_mode": "sliding_window", "num_tokens": 512}).__dict__,
        manager.decide({"attention_mode": "full", "num_tokens": 512, "max_expert_count": 32}).__dict__,
        manager.decide({"attention_mode": "full", "num_tokens": 512, "max_expert_count": 128}).__dict__,
    ]
    graph_run = manager.run_partial(
        {"attention_mode": "sliding_window", "num_tokens": 512},
        fallback_runner=lambda ctx: {"path": "fallback", "tokens": ctx["num_tokens"]},
    ).__dict__
    fallback_run = manager.run_partial(
        {"attention_mode": "hybrid_unknown", "num_tokens": 512},
        fallback_runner=lambda ctx: {"path": "fallback", "tokens": ctx["num_tokens"]},
    ).__dict__
    return {
        "decisions": decisions,
        "graph_run": graph_run,
        "fallback_run": fallback_run,
        "summary": manager.summary(),
        "diffusion_manifest": diffusion.to_json(),
    }


def validate_control_plane() -> dict[str, object]:
    plane = StaticityControlPlane(default_fallback="compiled")
    dense = token_prefill_manifest(
        "dense:tokens=832:reqs=8",
        template_tokens=832,
        max_reqs=8,
        lo=744,
        hi=805,
    )
    moe = moe_dispatch_manifest(
        "moe:capacity=64:topk=1",
        capacity_bucket=64,
        max_experts=128,
        top_k=1,
    )
    plane.register(dense)
    plane.register(moe)
    rejected_before_admit = plane.decide(
        dense.template_id,
        {"num_tokens": 750, "num_reqs": 4},
    ).__dict__
    plane.set_lifecycle(dense.template_id, TemplateLifecycle.ADMITTED, reason="validated_useful")
    admitted = plane.decide(
        dense.template_id,
        {"num_tokens": 750, "num_reqs": 4},
    ).__dict__
    guard_reject = plane.decide(
        dense.template_id,
        {"num_tokens": 900, "num_reqs": 4},
    ).__dict__
    moe_reject = plane.decide(
        moe.template_id,
        {"max_expert_count": 128, "top_k": 1},
    ).__dict__
    unknown = plane.decide("unknown", {}).__dict__
    return {
        "dense_manifest": dense.to_json(),
        "moe_manifest": moe.to_json(),
        "decisions": {
            "before_admit": rejected_before_admit,
            "admitted": admitted,
            "guard_reject": guard_reject,
            "moe_reject": moe_reject,
            "unknown": unknown,
        },
        "summary": plane.summary(),
    }


def validate_scheduler() -> dict[str, object]:
    signals = {
        "t=832": TemplateSchedulingSignal(
            "t=832",
            admitted=True,
            expected_saving_ms=10.0,
            useful_rate=0.9,
            max_wait_ms=2.0,
        ),
        "t=128": TemplateSchedulingSignal("t=128", admitted=False, expected_saving_ms=0.0),
        "t=drift": TemplateSchedulingSignal(
            "t=drift",
            admitted=True,
            expected_saving_ms=10.0,
            useful_rate=0.9,
            drifted=True,
        ),
    }
    scheduler = SlaAwareTemplateScheduler(
        lambda req: req["template_id"],
        signal_fn=lambda tid: signals.get(tid),
        max_wait_ms=4.0,
        max_batch_size=2,
        sla_p99_budget_ms=20.0,
        adaptive_wait=False,
        min_useful_rate=0.75,
    )
    flushed = []
    flushed.extend(scheduler.add({"id": 1, "template_id": "t=832"}, 0.0))
    flushed.extend(scheduler.add({"id": 2, "template_id": "t=832"}, 1.0))
    flushed.extend(scheduler.add({"id": 3, "template_id": "t=128"}, 2.0))
    flushed.extend(scheduler.add({"id": 4, "template_id": "t=drift"}, 3.0))
    flushed.extend(scheduler.finish())
    return {
        "batches": [
            {
                "template_id": batch.template_id,
                "request_ids": [req["id"] for req in batch.requests],
                "wait_ms": batch.wait_ms,
            }
            for batch in flushed
        ],
        "summary": scheduler.summary(),
    }


def validate_same_engine_live_capture() -> dict[str, object]:
    manager = SameEngineLiveCaptureManager(
        min_samples=2,
        min_useful_rate=1.0,
        min_saving_ms=1.0,
        max_p95_regression_ms=1.0,
        max_templates=1,
        validation_interval=8,
        amortization_replays=8,
    )
    manager.register(
        LiveTemplateSpec(
            "tokens=832",
            lo=744,
            hi=805,
            template_tokens=832,
            action="ours_cp",
            fallback_action="cp",
        )
    )
    callbacks = LiveCaptureCallbacks(
        capture=lambda spec, ctx: CaptureResult(
            captured=True,
            capture_ms=2.0,
            warmup_ms=1.0,
            memory_bytes=4096,
            handle=object(),
        ),
        replay=lambda spec, ctx: ReplayResult(
            output={"token": int(ctx["tokens"]) + 1},
            latency_ms=40.0,
        ),
        fallback=lambda spec, ctx: ReplayResult(
            output={"token": int(ctx["tokens"]) + 1},
            latency_ms=55.0,
        ),
        validate=lambda graph, fallback, ctx: ValidationResult(
            correct=graph == fallback,
        ),
    )
    runs = [
        manager.run({"tokens": tokens}, callbacks).__dict__
        for tokens in (750, 760, 770)
    ]
    policy = manager.export_policy(default_action="cp")
    drift_affected = manager.apply_drift_decision(
        DriftDecision(
            drifted=True,
            reason="token_distribution_shift",
            action="explore_new_templates",
            stats={"new_mean_tokens": 960},
        ),
        recent_template_ids=["tokens=832"],
    )
    manager.register(
        LiveTemplateSpec(
            "tokens=1024",
            lo=805,
            hi=1024,
            template_tokens=1024,
            action="ours_cp",
            fallback_action="cp",
        )
    )
    manager.records["tokens=1024"].status = LiveTemplateStatus.ADMITTED
    shadow_affected = manager.apply_drift_decision(
        DriftDecision(
            drifted=True,
            reason="negative_graph_rate_drift",
            action="increase_shadow_validation",
            stats={"negative_graph_rate": 0.4},
        ),
        recent_template_ids=["tokens=1024"],
    )
    blacklisted = manager.apply_drift_decision(
        DriftDecision(
            drifted=True,
            reason="correctness_drift",
            action="blacklist_recent_templates",
            stats={"correctness_failures": 1},
        ),
        recent_template_ids=["tokens=1024"],
    )
    return {
        "runs": runs,
        "policy": policy,
        "drift_actions": {
            "explore_new_templates": drift_affected,
            "increase_shadow_validation": shadow_affected,
            "blacklist_recent_templates": blacklisted,
        },
        "summary": manager.summary(include_events=False),
    }


def validate_workload_drift() -> dict[str, object]:
    detector = WorkloadDriftDetector(
        window=4,
        reference_window=8,
        min_samples=4,
        max_mean_token_shift=0.30,
    )
    decisions = []
    for _ in range(4):
        decisions.append(
            detector.observe(
                WorkloadObservation(
                    tokens=512,
                    template_id="tokens=512",
                    graph_used=True,
                    useful=True,
                    latency_ms=10.0,
                )
            ).__dict__
        )
    for _ in range(4):
        decisions.append(
            detector.observe(
                WorkloadObservation(
                    tokens=2048,
                    template_id="tokens=2048",
                    graph_used=False,
                    useful=False,
                    latency_ms=50.0,
                )
            ).__dict__
        )
    return {
        "decisions": decisions,
        "summary": detector.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/staticity_runtime_components_validation.json")
    args = parser.parse_args()
    registry = ArenaTemplateRegistry(
        token_templates=[
            TokenAxisTemplate(832, max_reqs=8, min_tokens=744, max_tokens=805),
            TokenAxisTemplate(1024, max_reqs=8, min_tokens=782, max_tokens=1012),
        ],
        expert_templates=[
            ExpertTrafficTemplate(32, max_experts=128, top_k=1),
            ExpertTrafficTemplate(64, max_experts=128, top_k=1),
        ],
    )
    payload = {
        "online_admission": validate_online_admission(),
        "control_plane": validate_control_plane(),
        "token_axis_arena": validate_token_axis(),
        "moe_expert_metadata": validate_moe(),
        "moe_dispatch_templates": validate_moe_dispatch_templates(),
        "partial_graph": validate_partial_graph(),
        "same_engine_live_capture": validate_same_engine_live_capture(),
        "scheduler": validate_scheduler(),
        "workload_drift": validate_workload_drift(),
        "registry": {
            "token_policy_ranges": registry.to_policy_ranges(),
            "token_template_for_790": registry.token_template_for(790).rule(),
            "expert_template_for_counts": registry.expert_template_for([1, 8, 0, 2]).template_id([1, 8, 0, 2]),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out), "checks": list(payload)}, indent=2))


if __name__ == "__main__":
    main()
