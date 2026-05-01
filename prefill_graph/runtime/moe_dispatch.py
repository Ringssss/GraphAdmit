from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .admission import OnlineSelfLearningAdmissionController
from .arena import ExpertMetadataCanonicalizer, ExpertTrafficTemplate


@dataclass(frozen=True)
class MoEDispatchTemplate:
    template_id: str
    capacity_bucket: int
    max_experts: int
    top_k: int = 1
    backend: str = "fused_moe"
    a2a_backend: str = "none"
    min_tokens: int = 0
    max_tokens: int | None = None
    action: str = "moe_graph"
    fallback_action: str = "moe_fallback"
    max_imbalance_ratio: float | None = None
    allow_a2a: bool = False
    priority: int = 0

    def accepts(self, *, expert_counts: list[int], tokens: int, top_k: int, a2a_backend: str = "none") -> tuple[bool, str]:
        if int(top_k) != int(self.top_k):
            return False, "topk_mismatch"
        if tokens <= int(self.min_tokens):
            return False, "tokens_below_template"
        if self.max_tokens is not None and tokens > int(self.max_tokens):
            return False, "tokens_above_template"
        if not expert_counts:
            return False, "missing_expert_counts"
        if max(int(count) for count in expert_counts) > int(self.capacity_bucket):
            return False, "expert_capacity_overflow"
        if len(expert_counts) > int(self.max_experts):
            return False, "too_many_experts"
        if a2a_backend != "none" and not self.allow_a2a:
            return False, "a2a_backend_not_graph_safe"
        if self.max_imbalance_ratio is not None:
            active = [int(count) for count in expert_counts if int(count) > 0]
            if active:
                avg = sum(active) / len(active)
                if avg > 0 and max(active) / avg > float(self.max_imbalance_ratio):
                    return False, "expert_imbalance_overflow"
        return True, "accepted"

    def canonicalizer(self, *, max_tokens: int | None = None) -> ExpertMetadataCanonicalizer:
        return ExpertMetadataCanonicalizer(
            ExpertTrafficTemplate(
                capacity_bucket=self.capacity_bucket,
                max_experts=self.max_experts,
                top_k=self.top_k,
                action=self.action,
                rare_imbalance_fallback=self.fallback_action,
            ),
            max_tokens=max_tokens or self.max_tokens or self.capacity_bucket * self.max_experts,
        )

    def to_json(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class MoEDispatchDecision:
    action: str
    reason: str
    template_id: str | None = None
    admitted: bool = False
    fallback_action: str = "moe_fallback"
    canonical_metadata: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MoEDispatchTemplateRegistry:
    """Template selector for fused-MoE/all-to-all dispatch dynamics."""

    def __init__(
        self,
        templates: list[MoEDispatchTemplate] | None = None,
        *,
        admission: OnlineSelfLearningAdmissionController | None = None,
        default_fallback: str = "moe_fallback",
    ):
        self.templates = sorted(
            list(templates or []),
            key=lambda item: (
                item.capacity_bucket,
                item.max_tokens or 2**63 - 1,
                -item.priority,
                item.template_id,
            ),
        )
        self.admission = admission
        self.default_fallback = default_fallback
        self.dispatch_counts: dict[str, int] = {}

    def register(self, template: MoEDispatchTemplate) -> None:
        self.templates.append(template)
        self.templates.sort(
            key=lambda item: (
                item.capacity_bucket,
                item.max_tokens or 2**63 - 1,
                -item.priority,
                item.template_id,
            )
        )

    def decide(
        self,
        *,
        expert_ids: list[list[int]] | list[int],
        expert_counts: list[int],
        tokens: int,
        top_k: int,
        a2a_backend: str = "none",
        expert_offsets: list[int] | None = None,
        token_permutation: list[list[int]] | list[int] | None = None,
        require_admitted: bool = True,
    ) -> MoEDispatchDecision:
        reject_reasons: dict[str, int] = {}
        for template in self.templates:
            ok, reason = template.accepts(
                expert_counts=expert_counts,
                tokens=tokens,
                top_k=top_k,
                a2a_backend=a2a_backend,
            )
            if not ok:
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                continue
            if self.admission is not None and require_admitted:
                admission_decision = self.admission.decide(template.template_id)
                if not admission_decision.admit:
                    self._count("not_admitted")
                    return MoEDispatchDecision(
                        action=template.fallback_action,
                        reason=admission_decision.reason,
                        template_id=template.template_id,
                        admitted=False,
                        fallback_action=template.fallback_action,
                        metadata={"admission": admission_decision.__dict__},
                    )
            canonical = template.canonicalizer(max_tokens=tokens).canonicalize(
                expert_ids=expert_ids,
                expert_counts=expert_counts,
                expert_offsets=expert_offsets,
                token_permutation=token_permutation,
            )
            if not canonical.get("accepted", False):
                self._count(canonical.get("reason", "canonical_reject"))
                return MoEDispatchDecision(
                    action=template.fallback_action,
                    reason=str(canonical.get("reason", "canonical_reject")),
                    template_id=template.template_id,
                    admitted=False,
                    fallback_action=template.fallback_action,
                    canonical_metadata=canonical,
                )
            self._count("graph")
            return MoEDispatchDecision(
                action=template.action,
                reason="template_selected",
                template_id=template.template_id,
                admitted=True,
                fallback_action=template.fallback_action,
                canonical_metadata=canonical,
                metadata={
                    "capacity_bucket": template.capacity_bucket,
                    "backend": template.backend,
                    "a2a_backend": a2a_backend,
                },
            )
        self._count("fallback")
        return MoEDispatchDecision(
            action=self.default_fallback,
            reason="no_matching_moe_template",
            fallback_action=self.default_fallback,
            metadata={"reject_reasons": reject_reasons},
        )

    def observe(
        self,
        template_id: str,
        *,
        graph_ms: float,
        fallback_ms: float,
        correct: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.admission is None:
            self.admission = OnlineSelfLearningAdmissionController()
        self.admission.observe(
            template_id,
            graph_ms=graph_ms,
            fallback_ms=fallback_ms,
            correct=correct,
            metadata=metadata or {},
        )

    def _count(self, key: str) -> None:
        self.dispatch_counts[key] = self.dispatch_counts.get(key, 0) + 1

    def summary(self) -> dict[str, Any]:
        return {
            "templates": [template.to_json() for template in self.templates],
            "dispatch_counts": dict(sorted(self.dispatch_counts.items())),
            "admission": (
                self.admission.summary(include_decisions=False)
                if self.admission is not None else None
            ),
        }
