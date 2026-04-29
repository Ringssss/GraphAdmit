from __future__ import annotations

import pytest

from benchmarks.build_useful_coverage_policy import correctness_flags
from prefill_graph.runtime import (
    PartialGraphTemplateManager,
    StaticityControlPlane,
    TemplateLifecycle,
    attention_segment_for_mode,
    moe_segment_for_capacity,
    moe_dispatch_manifest,
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


def test_correctness_flags_fail_closed() -> None:
    with pytest.raises(ValueError):
        correctness_flags({"per_req": []}, 1)
