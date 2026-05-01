from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


GuardFn = Callable[[dict[str, Any]], bool]
RunnerFn = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class PartialGraphSegment:
    name: str
    template_id: str
    guard_fields: tuple[str, ...] = ()
    static_fields: tuple[str, ...] = ()
    dynamic_value_fields: tuple[str, ...] = ()
    semantic_fields: tuple[str, ...] = ()
    fallback_action: str = "fallback"
    priority: int = 0

    def signature(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "template_id": self.template_id,
            "guard_fields": list(self.guard_fields),
            "static_fields": list(self.static_fields),
            "dynamic_value_fields": list(self.dynamic_value_fields),
            "semantic_fields": list(self.semantic_fields),
            "fallback_action": self.fallback_action,
            "priority": self.priority,
        }


@dataclass
class PartialGraphDecision:
    segment: str | None
    template_id: str | None
    action: str
    reason: str
    fallback_action: str = "fallback"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PartialGraphRunResult:
    action: str
    segment: str | None
    template_id: str | None
    reason: str
    output: Any
    fallback_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class PartialGraphTemplateManager:
    """Guarded partial-graph dispatcher for control-flow dynamic functions.

    The manager does not assume full-function capture is safe.  Adapters
    register graphable segments with guards.  At runtime a context is routed to
    the first admitted segment whose guard passes; otherwise it falls back.
    This is the missing abstraction for hybrid attention, MoE routing branches,
    diffusion step templates, early-exit, and sampling branches.
    """

    def __init__(self, *, default_fallback: str = "fallback"):
        self.default_fallback = default_fallback
        self._segments: list[PartialGraphSegment] = []
        self._guards: dict[str, GuardFn] = {}
        self._runners: dict[str, RunnerFn] = {}
        self._fallback_runners: dict[str, RunnerFn] = {}
        self._admitted: set[str] = set()
        self._blacklist: dict[str, str] = {}
        self._decision_counts: dict[str, int] = {}

    def register(
        self,
        segment: PartialGraphSegment,
        *,
        guard: GuardFn | None = None,
        runner: RunnerFn | None = None,
        fallback_runner: RunnerFn | None = None,
        admitted: bool = False,
    ) -> None:
        self._segments.append(segment)
        self._segments.sort(key=lambda item: item.priority)
        if guard is not None:
            self._guards[segment.template_id] = guard
        if runner is not None:
            self._runners[segment.template_id] = runner
        if fallback_runner is not None:
            self._fallback_runners[segment.template_id] = fallback_runner
        if admitted:
            self._admitted.add(segment.template_id)

    def admit(self, template_id: str) -> None:
        self._admitted.add(template_id)
        self._blacklist.pop(template_id, None)

    def blacklist(self, template_id: str, reason: str) -> None:
        self._admitted.discard(template_id)
        self._blacklist[template_id] = reason

    def decide(self, context: dict[str, Any]) -> PartialGraphDecision:
        for segment in self._segments:
            if segment.template_id in self._blacklist:
                continue
            guard = self._guards.get(segment.template_id)
            if guard is not None and not guard(context):
                continue
            if segment.template_id not in self._admitted:
                self._count("not_admitted")
                # A non-admitted guarded segment must not shadow a later admitted
                # segment.  This keeps partial-graph dispatch composable across
                # hybrid attention, MoE, diffusion, and sampling branches.
                continue
            self._count("graph")
            return PartialGraphDecision(
                segment=segment.name,
                template_id=segment.template_id,
                action="graph",
                reason="guard_passed",
                fallback_action=segment.fallback_action,
                metadata=segment.signature(),
            )
        self._count("fallback")
        return PartialGraphDecision(
            segment=None,
            template_id=None,
            action=self.default_fallback,
            reason="no_admitted_guarded_segment",
            fallback_action=self.default_fallback,
        )

    def run(self, context: dict[str, Any]) -> Any:
        decision = self.decide(context)
        if decision.action != "graph" or decision.template_id is None:
            raise RuntimeError(f"partial graph not admitted: {decision.reason}")
        runner = self._runners.get(decision.template_id)
        if runner is None:
            raise RuntimeError(f"no runner registered for {decision.template_id}")
        return runner(context)

    def run_partial(
        self,
        context: dict[str, Any],
        *,
        fallback_runner: RunnerFn | None = None,
    ) -> PartialGraphRunResult:
        decision = self.decide(context)
        if decision.action == "graph" and decision.template_id is not None:
            runner = self._runners.get(decision.template_id)
            if runner is None:
                runner = self._fallback_runners.get(decision.template_id)
                if runner is None:
                    if fallback_runner is None:
                        raise RuntimeError(
                            f"no runner registered for {decision.template_id}"
                        )
                    runner = fallback_runner
                output = runner(context)
                return PartialGraphRunResult(
                    action=decision.fallback_action,
                    segment=decision.segment,
                    template_id=decision.template_id,
                    reason="missing_graph_runner_fallback",
                    output=output,
                    fallback_used=True,
                    metadata=decision.metadata,
                )
            output = runner(context)
            return PartialGraphRunResult(
                action="graph",
                segment=decision.segment,
                template_id=decision.template_id,
                reason=decision.reason,
                output=output,
                fallback_used=False,
                metadata=decision.metadata,
            )
        if fallback_runner is None:
            raise RuntimeError(f"partial graph fallback required: {decision.reason}")
        output = fallback_runner(context)
        return PartialGraphRunResult(
            action=decision.action,
            segment=decision.segment,
            template_id=decision.template_id,
            reason=decision.reason,
            output=output,
            fallback_used=True,
            metadata=decision.metadata,
        )

    def _count(self, key: str) -> None:
        self._decision_counts[key] = self._decision_counts.get(key, 0) + 1

    def summary(self) -> dict[str, Any]:
        return {
            "default_fallback": self.default_fallback,
            "segments": [segment.signature() for segment in self._segments],
            "admitted": sorted(self._admitted),
            "blacklist": dict(sorted(self._blacklist.items())),
            "runners": sorted(self._runners),
            "fallback_runners": sorted(self._fallback_runners),
            "decision_counts": dict(sorted(self._decision_counts.items())),
        }


def attention_segment_for_mode(mode: str, *, template_tokens: int) -> PartialGraphSegment:
    return PartialGraphSegment(
        name=f"attention_{mode}",
        template_id=f"attention:{mode}:tokens={template_tokens}",
        guard_fields=("attention_mode", "num_tokens"),
        static_fields=("backend", "head_dim"),
        dynamic_value_fields=("positions", "slot_mapping"),
        semantic_fields=("attention_mode",),
        fallback_action="compiled",
    )


def moe_segment_for_capacity(capacity_bucket: int, *, top_k: int) -> PartialGraphSegment:
    return PartialGraphSegment(
        name=f"moe_capacity_{capacity_bucket}",
        template_id=f"moe:capacity={capacity_bucket}:topk={top_k}",
        guard_fields=("max_expert_count", "top_k"),
        static_fields=("num_experts", "hidden_size"),
        dynamic_value_fields=(
            "expert_ids",
            "expert_counts",
            "expert_offsets",
            "token_permutation",
        ),
        semantic_fields=("expert_ids", "expert_counts"),
        fallback_action="compiled",
    )


def diffusion_step_segment(step_bucket: str, *, template_tokens: int) -> PartialGraphSegment:
    return PartialGraphSegment(
        name=f"diffusion_step_{step_bucket}",
        template_id=f"diffusion:{step_bucket}:tokens={template_tokens}",
        guard_fields=("step", "mask_ratio", "num_tokens"),
        static_fields=("block_length",),
        dynamic_value_fields=("mask_positions", "update_indices", "confidence"),
        semantic_fields=("mask_positions", "update_indices"),
        fallback_action="eager",
    )
