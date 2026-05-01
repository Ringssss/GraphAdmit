from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkloadObservation:
    tokens: int
    template_id: str | None
    graph_used: bool
    useful: bool
    latency_ms: float
    fallback_ms: float | None = None
    correct: bool = True
    timestamp_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DriftDecision:
    drifted: bool
    reason: str
    action: str
    stats: dict[str, Any]


class WorkloadDriftDetector:
    """Online drift detector for useful-coverage driven graph policies."""

    def __init__(
        self,
        *,
        window: int = 32,
        reference_window: int = 64,
        min_samples: int = 16,
        max_mean_token_shift: float = 0.35,
        min_useful_rate_drop: float = 0.25,
        max_negative_rate: float = 0.35,
        max_mismatch_rate: float = 0.0,
    ):
        self.window = max(1, int(window))
        self.reference_window = max(self.window, int(reference_window))
        self.min_samples = int(min_samples)
        self.max_mean_token_shift = float(max_mean_token_shift)
        self.min_useful_rate_drop = float(min_useful_rate_drop)
        self.max_negative_rate = float(max_negative_rate)
        self.max_mismatch_rate = float(max_mismatch_rate)
        self.recent: deque[WorkloadObservation] = deque(maxlen=self.window)
        self.reference: deque[WorkloadObservation] = deque(maxlen=self.reference_window)
        self.decisions: list[DriftDecision] = []

    def observe(self, obs: WorkloadObservation) -> DriftDecision:
        self.recent.append(obs)
        self.reference.append(obs)
        decision = self.decide()
        self.decisions.append(decision)
        return decision

    def decide(self) -> DriftDecision:
        if len(self.recent) < self.min_samples or len(self.reference) < self.min_samples:
            return DriftDecision(False, "insufficient_samples", "keep_policy", self.stats())
        stats = self.stats()
        ref_mean = stats["reference_mean_tokens"]
        recent_mean = stats["recent_mean_tokens"]
        token_shift = abs(recent_mean - ref_mean) / max(1.0, ref_mean)
        useful_drop = stats["reference_useful_rate"] - stats["recent_useful_rate"]
        if stats["recent_mismatch_rate"] > self.max_mismatch_rate:
            return DriftDecision(True, "correctness_drift", "blacklist_recent_templates", stats)
        if stats["recent_negative_rate"] > self.max_negative_rate:
            return DriftDecision(True, "negative_graph_rate_drift", "increase_shadow_validation", stats)
        if token_shift > self.max_mean_token_shift:
            return DriftDecision(True, "token_distribution_shift", "explore_new_templates", stats)
        if useful_drop > self.min_useful_rate_drop:
            return DriftDecision(True, "useful_coverage_drop", "refresh_admission", stats)
        return DriftDecision(False, "stable", "keep_policy", stats)

    def stats(self) -> dict[str, Any]:
        recent = list(self.recent)
        reference = list(self.reference)
        return {
            "recent_samples": len(recent),
            "reference_samples": len(reference),
            "recent_mean_tokens": mean([obs.tokens for obs in recent]),
            "reference_mean_tokens": mean([obs.tokens for obs in reference]),
            "recent_useful_rate": rate([obs.useful for obs in recent]),
            "reference_useful_rate": rate([obs.useful for obs in reference]),
            "recent_negative_rate": rate([
                obs.graph_used and obs.correct and not obs.useful for obs in recent
            ]),
            "reference_negative_rate": rate([
                obs.graph_used and obs.correct and not obs.useful for obs in reference
            ]),
            "recent_mismatch_rate": rate([not obs.correct for obs in recent]),
            "reference_mismatch_rate": rate([not obs.correct for obs in reference]),
            "recent_templates": histogram([obs.template_id or "fallback" for obs in recent]),
            "reference_templates": histogram([obs.template_id or "fallback" for obs in reference]),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "reference_window": self.reference_window,
            "min_samples": self.min_samples,
            "stats": self.stats(),
            "last_decision": self.decisions[-1].__dict__ if self.decisions else None,
            "num_drift_decisions": sum(1 for item in self.decisions if item.drifted),
        }


def mean(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def rate(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for item in values if item) / len(values)


def histogram(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))
