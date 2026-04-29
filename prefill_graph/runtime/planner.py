from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ActionStats:
    avg: float | None = None
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    total: float | None = None
    correct: bool | None = None

    @classmethod
    def from_ms_stats(cls, stats: dict[str, Any], *, correct: bool | None = None) -> "ActionStats":
        return cls(
            avg=stats.get("avg_ms"),
            p50=stats.get("p50_ms"),
            p95=stats.get("p95_ms"),
            p99=stats.get("p99_ms"),
            correct=correct,
        )

    @classmethod
    def from_s_stats(cls, stats: dict[str, Any], *, correct: bool | None = None) -> "ActionStats":
        return cls(
            avg=stats.get("avg_s"),
            p50=stats.get("p50_s"),
            p95=stats.get("p95_s"),
            p99=stats.get("p99_s"),
            total=stats.get("total_s"),
            correct=correct,
        )


@dataclass(frozen=True)
class RequestContext:
    idx: int
    tokens: int
    mode: str = "prefill"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanDecision:
    action: str
    reason: str
    rule: dict[str, Any] | None = None
    expected_stats: ActionStats | None = None
    admitted: bool = True
    fallback_action: str | None = None


@dataclass
class RuntimePolicy:
    rules: list[dict[str, Any]]
    default_action: str
    baseline_action: str
    action_stats: dict[str, ActionStats]
    static_actions: set[str] = field(default_factory=set)
    correctness_required: bool = True
    latency_margin_pct: float = 0.0
    tail_guard_pct: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "RuntimePolicy":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "runtime_policy" in data:
            data = data["runtime_policy"]
        baseline = "default" if "default" in data.get("baseline_stats", {}) else "eager"
        baseline_stats = data.get("baseline_stats")
        if baseline_stats is None and isinstance(data.get("source_summary"), dict):
            baseline_stats = data["source_summary"].get("baseline_stats")
        baseline_stats = baseline_stats or {}
        if "default" in baseline_stats:
            baseline = "default"
        elif "cp" in baseline_stats:
            baseline = "cp"
        action_stats = {
            action: ActionStats.from_ms_stats(stats)
            for action, stats in baseline_stats.items()
        }
        if "best_policy_stats" in data:
            action_stats["policy"] = ActionStats.from_s_stats(data["best_policy_stats"])
        if "policy_stats" in data:
            action_stats["policy"] = ActionStats.from_ms_stats(data["policy_stats"])
        rules = data.get("rules") or []
        if not rules and "rows" in data:
            rules = cls._rules_from_rows(data["rows"], token_key="prompt_len")
        default_action = data.get("default_action") or (rules[-1]["action"] if rules else baseline)
        return cls(
            rules=rules,
            default_action=default_action,
            baseline_action=data.get("baseline_action", baseline),
            action_stats=action_stats,
            static_actions=set(data.get("static_actions") or []),
            correctness_required=bool(
                data.get("require_same_output", data.get("require_correct", True))
            ),
            latency_margin_pct=float(data.get("prefer_fallback_margin_pct", 0.0) or 0.0),
            metadata={
                key: value
                for key, value in data.items()
                if key not in {"rows", "baseline_stats", "rules"}
            },
        )

    @staticmethod
    def _rules_from_rows(rows: list[dict[str, Any]], token_key: str) -> list[dict[str, Any]]:
        if not rows:
            return []
        sorted_rows = sorted(rows, key=lambda row: (int(row.get(token_key, row.get("tok", 0))), int(row.get("idx", 0))))
        rules: list[dict[str, Any]] = []
        prev_hi = 0
        current_action = None
        current_lo = 0
        count = 0
        for row in sorted_rows:
            tok = int(row.get(token_key, row.get("tok", 0)))
            action = row["action"]
            if current_action is None:
                current_action = action
                current_lo = prev_hi
                count = 1
            elif action == current_action:
                count += 1
            else:
                rules.append({"lo": current_lo, "hi": prev_hi, "action": current_action, "n": count})
                current_lo = prev_hi
                current_action = action
                count = 1
            prev_hi = tok
        if current_action is not None:
            rules.append({"lo": current_lo, "hi": prev_hi, "action": current_action, "n": count})
        return rules

    def to_json(self) -> dict[str, Any]:
        return {
            "rules": self.rules,
            "default_action": self.default_action,
            "baseline_action": self.baseline_action,
            "static_actions": sorted(self.static_actions),
            "correctness_required": self.correctness_required,
            "latency_margin_pct": self.latency_margin_pct,
            "tail_guard_pct": self.tail_guard_pct,
            "action_stats": {
                action: stats.__dict__
                for action, stats in self.action_stats.items()
            },
            "metadata": self.metadata,
        }


class RuntimePlanner:
    def __init__(self, policy: RuntimePolicy):
        self.policy = policy
        self.decisions = 0
        self.action_counts: dict[str, int] = {}

    def choose(self, ctx: RequestContext) -> PlanDecision:
        action = self.policy.default_action
        matched = None
        for rule in self.policy.rules:
            lo = int(rule.get("lo", -1))
            hi = int(rule.get("hi", 2**63 - 1))
            if lo < ctx.tokens <= hi:
                action = str(rule["action"])
                matched = rule
                break
        stats = self.policy.action_stats.get(action)
        decision = PlanDecision(
            action=action,
            reason="range_rule" if matched else "default_action",
            rule=matched,
            expected_stats=stats,
        )
        self.decisions += 1
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        return decision

    def choose_with_admission(
        self,
        ctx: RequestContext,
        *,
        candidate_correct: bool | None = None,
        fallback_action: str | None = None,
    ) -> PlanDecision:
        decision = self.choose(ctx)
        if self.policy.correctness_required and candidate_correct is False:
            fallback = fallback_action or self.policy.baseline_action
            return PlanDecision(
                action=fallback,
                reason="correctness_rejected",
                rule=decision.rule,
                expected_stats=self.policy.action_stats.get(fallback),
                admitted=False,
                fallback_action=fallback,
            )
        return decision

    def summary(self) -> dict[str, Any]:
        return {
            "num_decisions": self.decisions,
            "action_counts": dict(sorted(self.action_counts.items())),
            "policy": self.policy.to_json(),
        }
