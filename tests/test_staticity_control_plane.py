from __future__ import annotations

import pytest

from benchmarks.build_useful_coverage_policy import correctness_flags
from prefill_graph.runtime import (
    CaptureResult,
    LiveCaptureCallbacks,
    LiveTemplateSpec,
    MoEDispatchTemplate,
    MoEDispatchTemplateRegistry,
    ReplayResult,
    SameEngineLiveCaptureManager,
    PartialGraphTemplateManager,
    StaticityControlPlane,
    TemplateLifecycle,
    ValidationResult,
    WorkloadDriftDetector,
    WorkloadObservation,
    attention_segment_for_mode,
    moe_segment_for_capacity,
    moe_dispatch_manifest,
    SlaAwareTemplateScheduler,
    TemplateSchedulingSignal,
    token_prefill_manifest,
)


def test_control_plane_falls_back_until_admitted() -> None:
    plane = StaticityControlPlane(default_fallback="compiled")
    manifest = token_prefill_manifest(
        "dense:832",
        template_tokens=832,
        max_reqs=8,
        lo=744,
        hi=805,
    )
    plane.register(manifest)

    before = plane.decide("dense:832", {"num_tokens": 750, "num_reqs": 4})
    assert not before.admitted
    assert before.action == "compiled"

    plane.set_lifecycle("dense:832", TemplateLifecycle.ADMITTED)
    admitted = plane.decide("dense:832", {"num_tokens": 750, "num_reqs": 4})
    assert admitted.admitted
    assert admitted.action == "graph"

    rejected = plane.decide("dense:832", {"num_tokens": 900, "num_reqs": 4})
    assert not rejected.admitted
    assert rejected.action == "compiled"
    assert rejected.reason == "tokens_outside_template_range"


def test_semantic_moe_manifest_requires_guarded_validation() -> None:
    manifest = moe_dispatch_manifest(
        "moe:64",
        capacity_bucket=64,
        max_experts=128,
        top_k=1,
    )
    assert "expert_ids" in manifest.semantic_fields
    assert manifest.requires_validation
    ok, reason = manifest.guard({"max_expert_count": 32, "top_k": 1})
    assert ok and reason is None
    ok, reason = manifest.guard({"max_expert_count": 128, "top_k": 1})
    assert not ok
    assert reason == "expert_capacity_overflow"


def test_partial_graph_skips_unadmitted_segment() -> None:
    manager = PartialGraphTemplateManager(default_fallback="compiled")
    manager.register(
        moe_segment_for_capacity(64, top_k=1),
        guard=lambda ctx: int(ctx.get("max_expert_count", 0)) <= 64,
        admitted=False,
    )
    manager.register(
        attention_segment_for_mode("full", template_tokens=1024),
        guard=lambda ctx: ctx.get("attention_mode") == "full",
        admitted=True,
    )
    decision = manager.decide({"attention_mode": "full", "max_expert_count": 32})
    assert decision.action == "graph"
    assert decision.template_id == "attention:full:tokens=1024"


def test_partial_graph_run_uses_guarded_runner_or_fallback() -> None:
    manager = PartialGraphTemplateManager(default_fallback="compiled")
    manager.register(
        attention_segment_for_mode("full", template_tokens=1024),
        guard=lambda ctx: ctx.get("attention_mode") == "full",
        runner=lambda ctx: {"path": "graph", "tokens": ctx["num_tokens"]},
        admitted=True,
    )
    graph = manager.run_partial(
        {"attention_mode": "full", "num_tokens": 512},
        fallback_runner=lambda ctx: {"path": "fallback"},
    )
    assert graph.action == "graph"
    assert graph.output["path"] == "graph"

    fallback = manager.run_partial(
        {"attention_mode": "sliding", "num_tokens": 512},
        fallback_runner=lambda ctx: {"path": "fallback"},
    )
    assert fallback.fallback_used
    assert fallback.output["path"] == "fallback"


def test_correctness_flags_fail_closed() -> None:
    with pytest.raises(ValueError):
        correctness_flags({"per_req": []}, 1)


def test_same_engine_live_capture_admits_then_replays() -> None:
    manager = SameEngineLiveCaptureManager(
        min_samples=2,
        min_useful_rate=1.0,
        min_saving_ms=1.0,
        max_p95_regression_ms=1.0,
        max_templates=1,
        validation_interval=8,
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
        capture=lambda spec, ctx: CaptureResult(True, capture_ms=2.0, warmup_ms=1.0, memory_bytes=10, handle=object()),
        replay=lambda spec, ctx: ReplayResult(output=int(ctx["tokens"]) + 1, latency_ms=40.0),
        fallback=lambda spec, ctx: ReplayResult(output=int(ctx["tokens"]) + 1, latency_ms=55.0),
        validate=lambda graph, fallback, ctx: ValidationResult(correct=graph == fallback),
    )

    first = manager.run({"tokens": 750}, callbacks)
    second = manager.run({"tokens": 760}, callbacks)
    third = manager.run({"tokens": 770}, callbacks)

    assert first.action == "fallback"
    assert second.action == "graph_next"
    assert third.action == "ours_cp"
    assert manager.records["tokens=832"].status.value == "admitted"


def test_same_engine_live_capture_blacklists_wrong_graph() -> None:
    manager = SameEngineLiveCaptureManager(min_samples=1)
    manager.register(LiveTemplateSpec("bad", lo=0, hi=128))
    callbacks = LiveCaptureCallbacks(
        capture=lambda spec, ctx: CaptureResult(True, handle=object()),
        replay=lambda spec, ctx: ReplayResult(output="wrong", latency_ms=1.0),
        fallback=lambda spec, ctx: ReplayResult(output="right", latency_ms=2.0),
        validate=lambda graph, fallback, ctx: ValidationResult(correct=graph == fallback, reason="token_mismatch"),
    )
    result = manager.run({"tokens": 64}, callbacks)
    assert result.action == "fallback"
    assert result.reason == "token_mismatch"
    assert manager.records["bad"].status.value == "blacklisted"


def test_moe_dispatch_template_requires_admission_and_capacity() -> None:
    registry = MoEDispatchTemplateRegistry()
    registry.register(
        MoEDispatchTemplate(
            "moe:cap4",
            capacity_bucket=4,
            max_experts=4,
            top_k=1,
            action="moe_graph",
            fallback_action="compiled",
            max_tokens=16,
        )
    )
    decision = registry.decide(
        expert_ids=[0, 1, 1, 2],
        expert_counts=[1, 2, 1, 0],
        tokens=4,
        top_k=1,
        require_admitted=False,
    )
    assert decision.admitted
    assert decision.action == "moe_graph"

    overflow = registry.decide(
        expert_ids=[0] * 8,
        expert_counts=[8, 0, 0, 0],
        tokens=8,
        top_k=1,
        require_admitted=False,
    )
    assert not overflow.admitted
    assert overflow.action == "moe_fallback"


def test_workload_drift_detector_flags_token_shift() -> None:
    detector = WorkloadDriftDetector(
        window=4,
        reference_window=8,
        min_samples=4,
        max_mean_token_shift=0.30,
    )
    for _ in range(4):
        decision = detector.observe(
            WorkloadObservation(
                tokens=512,
                template_id="tokens=512",
                graph_used=True,
                useful=True,
                latency_ms=10.0,
            )
        )
    assert not decision.drifted
    for _ in range(4):
        decision = detector.observe(
            WorkloadObservation(
                tokens=2048,
                template_id="tokens=2048",
                graph_used=False,
                useful=False,
                latency_ms=50.0,
            )
        )
    assert decision.drifted
    assert decision.action == "explore_new_templates"


def test_tail_safe_scheduler_rejects_drifted_signal() -> None:
    signals = {
        "stable": TemplateSchedulingSignal(
            "stable",
            admitted=True,
            expected_saving_ms=10.0,
            useful_rate=0.9,
            max_wait_ms=2.0,
        ),
        "drifted": TemplateSchedulingSignal(
            "drifted",
            admitted=True,
            expected_saving_ms=10.0,
            useful_rate=0.9,
            drifted=True,
        ),
    }
    scheduler = SlaAwareTemplateScheduler(
        lambda req: req["template"],
        signal_fn=lambda tid: signals.get(tid),
        max_wait_ms=4.0,
        max_batch_size=2,
        adaptive_wait=False,
        min_useful_rate=0.75,
    )
    scheduler.add({"id": 1, "template": "stable"}, 0.0)
    scheduler.add({"id": 2, "template": "stable"}, 1.0)
    scheduler.add({"id": 3, "template": "drifted"}, 2.0)
    scheduler.finish()
    summary = scheduler.summary()
    assert summary["drift_rejected"] == 1
    assert summary["max_observed_wait_ms"] <= 2.0
