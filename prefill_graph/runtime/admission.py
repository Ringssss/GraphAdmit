from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass
class TemplateState:
    key: str
    captures: int = 0
    replays: int = 0
    validation_passes: int = 0
    validation_failures: int = 0
    capture_seconds: float = 0.0
    replay_seconds: float = 0.0
    disabled: bool = False
    disable_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def admitted(self) -> bool:
        return not self.disabled and self.validation_failures == 0 and self.validation_passes > 0


class TemplateAdmissionController:
    def __init__(
        self,
        *,
        max_templates: int = 0,
        min_free_memory_bytes: int = 0,
        admit_after_passes: int = 1,
        min_replay_saving_ms: float = 0.0,
    ):
        self.max_templates = max_templates
        self.min_free_memory_bytes = min_free_memory_bytes
        self.admit_after_passes = admit_after_passes
        self.min_replay_saving_ms = min_replay_saving_ms
        self.templates: dict[str, TemplateState] = {}
        self.rejections: dict[str, int] = {}

    def state(self, key: Any) -> TemplateState:
        stable_key = repr(key)
        if stable_key not in self.templates:
            self.templates[stable_key] = TemplateState(stable_key)
        return self.templates[stable_key]

    def can_capture(self, key: Any, *, free_memory_bytes: int | None = None) -> tuple[bool, str | None]:
        stable_key = repr(key)
        if stable_key in self.templates and self.templates[stable_key].disabled:
            return False, self.templates[stable_key].disable_reason or "template_disabled"
        if stable_key not in self.templates and self.max_templates and len(self.templates) >= self.max_templates:
            self.rejections["max_templates"] = self.rejections.get("max_templates", 0) + 1
            return False, "max_templates"
        if (
            free_memory_bytes is not None
            and self.min_free_memory_bytes
            and free_memory_bytes < self.min_free_memory_bytes
        ):
            self.rejections["memory_guard"] = self.rejections.get("memory_guard", 0) + 1
            return False, "memory_guard"
        return True, None

    def record_capture(self, key: Any, seconds: float) -> None:
        state = self.state(key)
        state.captures += 1
        state.capture_seconds += float(seconds)

    def record_replay(self, key: Any, seconds: float) -> None:
        state = self.state(key)
        state.replays += 1
        state.replay_seconds += float(seconds)

    def record_validation(self, key: Any, passed: bool, *, reason: str | None = None) -> bool:
        state = self.state(key)
        if passed:
            state.validation_passes += 1
            return state.validation_passes >= self.admit_after_passes
        state.validation_failures += 1
        state.disabled = True
        state.disable_reason = reason or "validation_failed"
        return False

    def disable(self, key: Any, reason: str) -> None:
        state = self.state(key)
        state.disabled = True
        state.disable_reason = reason

    def summary(self) -> dict[str, Any]:
        return {
            "max_templates": self.max_templates,
            "min_free_memory_bytes": self.min_free_memory_bytes,
            "admit_after_passes": self.admit_after_passes,
            "num_templates": len(self.templates),
            "num_admitted": sum(1 for state in self.templates.values() if state.admitted),
            "num_disabled": sum(1 for state in self.templates.values() if state.disabled),
            "rejections": dict(sorted(self.rejections.items())),
            "templates": [state.__dict__ for state in self.templates.values()],
        }


@dataclass
class LatencyObservation:
    template_id: str
    graph_ms: float
    fallback_ms: float
    correct: bool = True
    capture_ms: float = 0.0
    warmup_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def delta_ms(self) -> float:
        return float(self.graph_ms) - float(self.fallback_ms)

    @property
    def useful(self) -> bool:
        return self.correct and self.delta_ms < 0.0


@dataclass
class OnlineAdmissionDecision:
    template_id: str
    admit: bool
    action: str
    reason: str
    samples: int
    useful_rate: float
    ewma_saving_ms: float | None
    p95_regression_ms: float | None
    fallback_action: str = "fallback"


@dataclass
class OnlineTemplateStats:
    template_id: str
    samples: int = 0
    useful: int = 0
    regressions: int = 0
    correctness_failures: int = 0
    graph_ewma_ms: float | None = None
    fallback_ewma_ms: float | None = None
    saving_ewma_ms: float | None = None
    capture_ms: float = 0.0
    warmup_ms: float = 0.0
    deltas_ms: list[float] = field(default_factory=list)
    disabled: bool = False
    disable_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def useful_rate(self) -> float:
        return self.useful / self.samples if self.samples else 0.0

    @property
    def p95_regression_ms(self) -> float | None:
        regressions = sorted(delta for delta in self.deltas_ms if delta >= 0.0)
        if not regressions:
            return 0.0 if self.deltas_ms else None
        if len(regressions) == 1:
            return float(regressions[0])
        pos = (len(regressions) - 1) * 0.95
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return float(regressions[lo])
        return float(regressions[lo] * (hi - pos) + regressions[hi] * (pos - lo))

    def update(self, obs: LatencyObservation, *, ewma_alpha: float, max_history: int) -> None:
        self.samples += 1
        self.useful += int(obs.useful)
        self.regressions += int(obs.correct and obs.delta_ms >= 0.0)
        self.correctness_failures += int(not obs.correct)
        self.capture_ms += float(obs.capture_ms)
        self.warmup_ms += float(obs.warmup_ms)
        self.metadata.update(obs.metadata or {})
        if self.graph_ewma_ms is None:
            self.graph_ewma_ms = float(obs.graph_ms)
            self.fallback_ewma_ms = float(obs.fallback_ms)
            self.saving_ewma_ms = float(obs.fallback_ms) - float(obs.graph_ms)
        else:
            alpha = ewma_alpha
            self.graph_ewma_ms = alpha * float(obs.graph_ms) + (1 - alpha) * self.graph_ewma_ms
            self.fallback_ewma_ms = alpha * float(obs.fallback_ms) + (1 - alpha) * self.fallback_ewma_ms
            self.saving_ewma_ms = alpha * (float(obs.fallback_ms) - float(obs.graph_ms)) + (1 - alpha) * self.saving_ewma_ms
        self.deltas_ms.append(obs.delta_ms)
        if len(self.deltas_ms) > max_history:
            self.deltas_ms = self.deltas_ms[-max_history:]

    def to_json(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "samples": self.samples,
            "useful": self.useful,
            "regressions": self.regressions,
            "correctness_failures": self.correctness_failures,
            "useful_rate": self.useful_rate,
            "graph_ewma_ms": self.graph_ewma_ms,
            "fallback_ewma_ms": self.fallback_ewma_ms,
            "saving_ewma_ms": self.saving_ewma_ms,
            "p95_regression_ms": self.p95_regression_ms,
            "capture_ms": self.capture_ms,
            "warmup_ms": self.warmup_ms,
            "disabled": self.disabled,
            "disable_reason": self.disable_reason,
            "metadata": self.metadata,
        }


class OnlineSelfLearningAdmissionController:
    """Online useful-coverage admission for graph templates.

    This controller is intentionally framework-neutral: vLLM, dInfer, MoE, or
    communication adapters feed graph-vs-fallback latency and correctness
    observations.  The controller admits only templates that are correct and
    whose measured replay benefit exceeds capture/warmup amortization and tail
    regression guards.
    """

    def __init__(
        self,
        *,
        min_samples: int = 3,
        min_useful_rate: float = 0.75,
        min_saving_ms: float = 0.5,
        max_p95_regression_ms: float = 2.0,
        max_correctness_failures: int = 0,
        ewma_alpha: float = 0.35,
        amortization_replays: int = 32,
        max_history: int = 128,
        fallback_action: str = "fallback",
    ):
        self.min_samples = int(min_samples)
        self.min_useful_rate = float(min_useful_rate)
        self.min_saving_ms = float(min_saving_ms)
        self.max_p95_regression_ms = float(max_p95_regression_ms)
        self.max_correctness_failures = int(max_correctness_failures)
        self.ewma_alpha = float(ewma_alpha)
        self.amortization_replays = max(1, int(amortization_replays))
        self.max_history = int(max_history)
        self.fallback_action = fallback_action
        self.templates: dict[str, OnlineTemplateStats] = {}
        self.decisions: list[OnlineAdmissionDecision] = []

    def state(self, template_id: Any) -> OnlineTemplateStats:
        key = str(template_id)
        if key not in self.templates:
            self.templates[key] = OnlineTemplateStats(template_id=key)
        return self.templates[key]

    def observe(
        self,
        template_id: Any,
        *,
        graph_ms: float,
        fallback_ms: float,
        correct: bool = True,
        capture_ms: float = 0.0,
        warmup_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> OnlineAdmissionDecision:
        obs = LatencyObservation(
            template_id=str(template_id),
            graph_ms=float(graph_ms),
            fallback_ms=float(fallback_ms),
            correct=bool(correct),
            capture_ms=float(capture_ms),
            warmup_ms=float(warmup_ms),
            metadata=metadata or {},
        )
        stats = self.state(obs.template_id)
        stats.update(obs, ewma_alpha=self.ewma_alpha, max_history=self.max_history)
        if stats.correctness_failures > self.max_correctness_failures:
            stats.disabled = True
            stats.disable_reason = "correctness_failure"
        decision = self.decide(obs.template_id)
        self.decisions.append(decision)
        return decision

    def decide(self, template_id: Any) -> OnlineAdmissionDecision:
        stats = self.state(template_id)
        saving = stats.saving_ewma_ms
        p95_reg = stats.p95_regression_ms
        amortized_cost = (stats.capture_ms + stats.warmup_ms) / self.amortization_replays
        effective_saving = None if saving is None else saving - amortized_cost
        if stats.disabled:
            return self._decision(stats, False, "fallback", stats.disable_reason or "disabled")
        if stats.samples < self.min_samples:
            return self._decision(stats, False, "explore", "explore_until_min_samples")
        if stats.correctness_failures > self.max_correctness_failures:
            stats.disabled = True
            stats.disable_reason = "correctness_failure"
            return self._decision(stats, False, "fallback", "correctness_failure")
        if stats.useful_rate < self.min_useful_rate:
            return self._decision(stats, False, "fallback", "low_useful_rate")
        if effective_saving is None or effective_saving < self.min_saving_ms:
            return self._decision(stats, False, "fallback", "insufficient_latency_saving")
        if p95_reg is not None and p95_reg > self.max_p95_regression_ms:
            return self._decision(stats, False, "fallback", "tail_regression_guard")
        return self._decision(stats, True, "graph", "admitted")

    def disable(self, template_id: Any, reason: str) -> None:
        stats = self.state(template_id)
        stats.disabled = True
        stats.disable_reason = reason

    def _decision(
        self,
        stats: OnlineTemplateStats,
        admit: bool,
        action: str,
        reason: str,
    ) -> OnlineAdmissionDecision:
        return OnlineAdmissionDecision(
            template_id=stats.template_id,
            admit=admit,
            action=action,
            reason=reason,
            samples=stats.samples,
            useful_rate=stats.useful_rate,
            ewma_saving_ms=stats.saving_ewma_ms,
            p95_regression_ms=stats.p95_regression_ms,
            fallback_action=self.fallback_action,
        )

    def export_runtime_policy(self, *, default_action: str = "fallback") -> dict[str, Any]:
        rules = []
        for stats in self.templates.values():
            decision = self.decide(stats.template_id)
            meta = stats.metadata
            if not decision.admit:
                continue
            if "lo" in meta and "hi" in meta:
                rule = {
                    "lo": int(meta["lo"]),
                    "hi": int(meta["hi"]),
                    "action": meta.get("action", "ours_cp"),
                    "n": stats.samples,
                    "reason": f"online-admitted: useful_rate={stats.useful_rate:.2f}, saving_ewma_ms={stats.saving_ewma_ms:.2f}",
                }
                if meta.get("template_tokens") is not None:
                    rule["template_tokens"] = int(meta["template_tokens"])
                rules.append(rule)
        rules.sort(key=lambda row: (row["lo"], row["hi"]))
        return {
            "kind": "online_self_learning_runtime_policy",
            "default_action": default_action,
            "baseline_action": self.fallback_action,
            "correctness_required": True,
            "rules": rules,
            "fixed_metadata_arena_ranges": [
                {
                    "lo": rule["lo"],
                    "hi": rule["hi"],
                    "template_tokens": rule.get("template_tokens"),
                    "action": rule["action"],
                    "n": rule["n"],
                }
                for rule in rules
                if rule.get("template_tokens") is not None
            ],
            "online_admission": self.summary(include_decisions=False),
        }

    def summary(self, *, include_decisions: bool = True) -> dict[str, Any]:
        payload = {
            "mode": "online_self_learning_admission",
            "min_samples": self.min_samples,
            "min_useful_rate": self.min_useful_rate,
            "min_saving_ms": self.min_saving_ms,
            "max_p95_regression_ms": self.max_p95_regression_ms,
            "max_correctness_failures": self.max_correctness_failures,
            "ewma_alpha": self.ewma_alpha,
            "amortization_replays": self.amortization_replays,
            "templates": [state.to_json() for state in self.templates.values()],
        }
        if include_decisions:
            payload["decisions"] = [decision.__dict__ for decision in self.decisions]
        return payload
