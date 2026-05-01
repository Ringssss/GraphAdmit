from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Any

from .drift import DriftDecision


class LiveTemplateStatus(str, Enum):
    CANDIDATE = "candidate"
    CAPTURING = "capturing"
    SHADOW_VALIDATING = "shadow_validating"
    ADMITTED = "admitted"
    BLACKLISTED = "blacklisted"
    EVICTED = "evicted"
    FALLBACK_ONLY = "fallback_only"


@dataclass(frozen=True)
class LiveTemplateSpec:
    template_id: str
    lo: int = 0
    hi: int = 2**63 - 1
    template_tokens: int | None = None
    action: str = "graph"
    fallback_action: str = "fallback"
    capture_key: str | None = None
    max_memory_bytes: int = 0
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def accepts(self, context: dict[str, Any]) -> bool:
        raw_tokens = context.get("tokens", context.get("num_tokens"))
        if raw_tokens is None:
            return True
        tokens = int(raw_tokens)
        return int(self.lo) < tokens <= int(self.hi)


@dataclass
class CaptureResult:
    captured: bool
    capture_ms: float = 0.0
    warmup_ms: float = 0.0
    memory_bytes: int = 0
    reason: str = "captured"
    handle: Any = None


@dataclass
class ReplayResult:
    output: Any
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    correct: bool
    reason: str = "validated"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveCaptureCallbacks:
    capture: Callable[[LiveTemplateSpec, dict[str, Any]], CaptureResult]
    replay: Callable[[LiveTemplateSpec, dict[str, Any]], ReplayResult]
    fallback: Callable[[LiveTemplateSpec | None, dict[str, Any]], ReplayResult]
    validate: Callable[[Any, Any, dict[str, Any]], ValidationResult] | None = None
    evict: Callable[[LiveTemplateSpec, Any], None] | None = None
    should_shadow: Callable[[LiveTemplateSpec, int], bool] | None = None


@dataclass
class LiveTemplateRecord:
    spec: LiveTemplateSpec
    status: LiveTemplateStatus = LiveTemplateStatus.CANDIDATE
    attempts: int = 0
    captures: int = 0
    replays: int = 0
    fallbacks: int = 0
    shadow_validations: int = 0
    useful: int = 0
    regressions: int = 0
    correctness_failures: int = 0
    capture_ms: float = 0.0
    warmup_ms: float = 0.0
    memory_bytes: int = 0
    graph_ms: list[float] = field(default_factory=list)
    fallback_ms: list[float] = field(default_factory=list)
    deltas_ms: list[float] = field(default_factory=list)
    handle: Any = None
    last_reason: str | None = None
    created_s: float = field(default_factory=time.time)
    last_used_s: float = field(default_factory=time.time)

    @property
    def samples(self) -> int:
        return len(self.deltas_ms)

    @property
    def useful_rate(self) -> float:
        return self.useful / self.samples if self.samples else 0.0

    @property
    def avg_saving_ms(self) -> float | None:
        if not self.deltas_ms:
            return None
        return -sum(self.deltas_ms) / len(self.deltas_ms)

    @property
    def p95_regression_ms(self) -> float:
        regressions = sorted(delta for delta in self.deltas_ms if delta >= 0.0)
        if not regressions:
            return 0.0
        return percentile(regressions, 95.0)

    @property
    def amortized_capture_ms(self) -> float:
        return self.capture_ms + self.warmup_ms

    def observe(
        self,
        *,
        graph_ms: float,
        fallback_ms: float,
        correct: bool,
        max_history: int,
    ) -> None:
        delta = float(graph_ms) - float(fallback_ms)
        self.graph_ms.append(float(graph_ms))
        self.fallback_ms.append(float(fallback_ms))
        self.deltas_ms.append(delta)
        if len(self.deltas_ms) > max_history:
            self.graph_ms = self.graph_ms[-max_history:]
            self.fallback_ms = self.fallback_ms[-max_history:]
            self.deltas_ms = self.deltas_ms[-max_history:]
        self.useful += int(correct and delta < 0.0)
        self.regressions += int(correct and delta >= 0.0)
        self.correctness_failures += int(not correct)
        self.last_used_s = time.time()

    def score(self) -> float:
        saving = self.avg_saving_ms
        if saving is None:
            return -1.0
        return max(0.0, saving) * max(1, self.samples) - self.memory_bytes / 1e9

    def to_json(self) -> dict[str, Any]:
        return {
            "template_id": self.spec.template_id,
            "status": self.status.value,
            "attempts": self.attempts,
            "captures": self.captures,
            "replays": self.replays,
            "fallbacks": self.fallbacks,
            "shadow_validations": self.shadow_validations,
            "samples": self.samples,
            "useful": self.useful,
            "regressions": self.regressions,
            "correctness_failures": self.correctness_failures,
            "useful_rate": self.useful_rate,
            "avg_saving_ms": self.avg_saving_ms,
            "p95_regression_ms": self.p95_regression_ms,
            "capture_ms": self.capture_ms,
            "warmup_ms": self.warmup_ms,
            "memory_bytes": self.memory_bytes,
            "last_reason": self.last_reason,
            "last_used_s": self.last_used_s,
            "spec": self.spec.__dict__,
        }


@dataclass
class LiveCaptureRunResult:
    template_id: str | None
    action: str
    reason: str
    output: Any
    graph_ms: float | None = None
    fallback_ms: float | None = None
    correct: bool | None = None
    status: str = LiveTemplateStatus.FALLBACK_ONLY.value
    shadow_validated: bool = False
    evicted: list[str] = field(default_factory=list)


class SameEngineLiveCaptureManager:
    """Fail-closed live capture/admission/eviction manager.

    The manager is framework-neutral.  A serving adapter provides callbacks for
    capture, graph replay, fallback, and validation.  The manager never returns
    graph output before a candidate has passed shadow validation unless the
    template is already admitted and no shadow sample is requested.
    """

    def __init__(
        self,
        *,
        min_samples: int = 2,
        min_useful_rate: float = 0.75,
        min_saving_ms: float = 0.5,
        max_p95_regression_ms: float = 2.0,
        max_correctness_failures: int = 0,
        max_templates: int = 0,
        max_graph_memory_bytes: int = 0,
        validation_interval: int = 16,
        amortization_replays: int = 32,
        max_history: int = 128,
    ):
        self.min_samples = int(min_samples)
        self.min_useful_rate = float(min_useful_rate)
        self.min_saving_ms = float(min_saving_ms)
        self.max_p95_regression_ms = float(max_p95_regression_ms)
        self.max_correctness_failures = int(max_correctness_failures)
        self.max_templates = int(max_templates)
        self.max_graph_memory_bytes = int(max_graph_memory_bytes)
        self.validation_interval = max(1, int(validation_interval))
        self.amortization_replays = max(1, int(amortization_replays))
        self.max_history = int(max_history)
        self.records: dict[str, LiveTemplateRecord] = {}
        self.events: list[dict[str, Any]] = []
        self.shadow_validation_multiplier = 1

    def register(self, spec: LiveTemplateSpec) -> LiveTemplateRecord:
        record = self.records.get(spec.template_id)
        if record is None:
            record = LiveTemplateRecord(spec=spec)
            self.records[spec.template_id] = record
        else:
            record.spec = spec
        return record

    def select(self, context: dict[str, Any]) -> LiveTemplateRecord | None:
        candidates = [
            record for record in self.records.values()
            if record.status != LiveTemplateStatus.EVICTED and record.spec.accepts(context)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda record: (
                record.spec.template_tokens or 2**63 - 1,
                -record.spec.priority,
                record.spec.template_id,
            ),
        )

    def run(
        self,
        context: dict[str, Any],
        callbacks: LiveCaptureCallbacks,
    ) -> LiveCaptureRunResult:
        record = self.select(context)
        if record is None:
            fallback = callbacks.fallback(None, context)
            return LiveCaptureRunResult(
                template_id=None,
                action="fallback",
                reason="no_candidate_template",
                output=fallback.output,
                fallback_ms=fallback.latency_ms,
            )

        record.attempts += 1
        record.last_used_s = time.time()
        evicted: list[str] = []

        if record.status in {
            LiveTemplateStatus.BLACKLISTED,
            LiveTemplateStatus.EVICTED,
            LiveTemplateStatus.FALLBACK_ONLY,
        }:
            record.fallbacks += 1
            fallback = callbacks.fallback(record.spec, context)
            return LiveCaptureRunResult(
                template_id=record.spec.template_id,
                action="fallback",
                reason=record.last_reason or record.status.value,
                output=fallback.output,
                fallback_ms=fallback.latency_ms,
                status=record.status.value,
            )

        if record.handle is None:
            record.status = LiveTemplateStatus.CAPTURING
            capture = callbacks.capture(record.spec, context)
            record.captures += int(capture.captured)
            record.capture_ms += float(capture.capture_ms)
            record.warmup_ms += float(capture.warmup_ms)
            record.memory_bytes = int(capture.memory_bytes)
            record.handle = capture.handle
            if not capture.captured:
                record.status = LiveTemplateStatus.FALLBACK_ONLY
                record.last_reason = capture.reason
                fallback = callbacks.fallback(record.spec, context)
                return LiveCaptureRunResult(
                    template_id=record.spec.template_id,
                    action="fallback",
                    reason=capture.reason,
                    output=fallback.output,
                    fallback_ms=fallback.latency_ms,
                    status=record.status.value,
                )
            record.status = LiveTemplateStatus.SHADOW_VALIDATING
            evicted.extend(self.evict_if_needed(callbacks.evict))

        if record.status == LiveTemplateStatus.SHADOW_VALIDATING:
            graph = callbacks.replay(record.spec, context)
            fallback = callbacks.fallback(record.spec, context)
            validation = self._validate(callbacks, graph.output, fallback.output, context)
            record.replays += 1
            record.fallbacks += 1
            record.shadow_validations += 1
            record.observe(
                graph_ms=graph.latency_ms,
                fallback_ms=fallback.latency_ms,
                correct=validation.correct,
                max_history=self.max_history,
            )
            admitted, reason = self._admission(record)
            if not validation.correct:
                record.status = LiveTemplateStatus.BLACKLISTED
                record.last_reason = validation.reason
                return LiveCaptureRunResult(
                    template_id=record.spec.template_id,
                    action="fallback",
                    reason=validation.reason,
                    output=fallback.output,
                    graph_ms=graph.latency_ms,
                    fallback_ms=fallback.latency_ms,
                    correct=False,
                    status=record.status.value,
                    shadow_validated=True,
                    evicted=evicted,
                )
            if admitted:
                record.status = LiveTemplateStatus.ADMITTED
                record.last_reason = reason
            else:
                record.last_reason = reason
            return LiveCaptureRunResult(
                template_id=record.spec.template_id,
                action="fallback" if not admitted else "graph_next",
                reason=reason,
                output=fallback.output,
                graph_ms=graph.latency_ms,
                fallback_ms=fallback.latency_ms,
                correct=True,
                status=record.status.value,
                shadow_validated=True,
                evicted=evicted,
            )

        shadow = self._should_shadow(record, callbacks)
        graph = callbacks.replay(record.spec, context)
        record.replays += 1
        if not shadow:
            return LiveCaptureRunResult(
                template_id=record.spec.template_id,
                action=record.spec.action,
                reason="admitted_replay",
                output=graph.output,
                graph_ms=graph.latency_ms,
                status=record.status.value,
                evicted=evicted,
            )

        fallback = callbacks.fallback(record.spec, context)
        validation = self._validate(callbacks, graph.output, fallback.output, context)
        record.fallbacks += 1
        record.shadow_validations += 1
        record.observe(
            graph_ms=graph.latency_ms,
            fallback_ms=fallback.latency_ms,
            correct=validation.correct,
            max_history=self.max_history,
        )
        admitted, reason = self._admission(record)
        if not validation.correct:
            record.status = LiveTemplateStatus.BLACKLISTED
            record.last_reason = validation.reason
            return LiveCaptureRunResult(
                template_id=record.spec.template_id,
                action="fallback",
                reason=validation.reason,
                output=fallback.output,
                graph_ms=graph.latency_ms,
                fallback_ms=fallback.latency_ms,
                correct=False,
                status=record.status.value,
                shadow_validated=True,
                evicted=evicted,
            )
        if not admitted:
            record.status = LiveTemplateStatus.SHADOW_VALIDATING
            record.last_reason = reason
            return LiveCaptureRunResult(
                template_id=record.spec.template_id,
                action="fallback",
                reason=reason,
                output=fallback.output,
                graph_ms=graph.latency_ms,
                fallback_ms=fallback.latency_ms,
                correct=True,
                status=record.status.value,
                shadow_validated=True,
                evicted=evicted,
            )
        return LiveCaptureRunResult(
            template_id=record.spec.template_id,
            action=record.spec.action,
            reason="admitted_shadow_validated",
            output=graph.output,
            graph_ms=graph.latency_ms,
            fallback_ms=fallback.latency_ms,
            correct=True,
            status=record.status.value,
            shadow_validated=True,
            evicted=evicted,
        )

    def observe(
        self,
        template_id: str,
        *,
        graph_ms: float,
        fallback_ms: float,
        correct: bool,
    ) -> tuple[bool, str]:
        record = self.records[template_id]
        record.observe(
            graph_ms=graph_ms,
            fallback_ms=fallback_ms,
            correct=correct,
            max_history=self.max_history,
        )
        admitted, reason = self._admission(record)
        if admitted:
            record.status = LiveTemplateStatus.ADMITTED
        elif record.correctness_failures > self.max_correctness_failures:
            record.status = LiveTemplateStatus.BLACKLISTED
        record.last_reason = reason
        return admitted, reason

    def evict_if_needed(
        self,
        evict_fn: Callable[[LiveTemplateSpec, Any], None] | None = None,
    ) -> list[str]:
        evicted: list[str] = []
        while self._over_budget():
            victim = self._select_victim()
            if victim is None:
                break
            if evict_fn is not None and victim.handle is not None:
                evict_fn(victim.spec, victim.handle)
            victim.status = LiveTemplateStatus.EVICTED
            victim.last_reason = "evicted_budget"
            evicted.append(victim.spec.template_id)
        return evicted

    def apply_drift_decision(
        self,
        decision: DriftDecision | dict[str, Any],
        *,
        recent_template_ids: list[str | None] | None = None,
        reason_prefix: str = "drift",
    ) -> list[str]:
        """Apply workload-drift feedback to live template state.

        Drift detection is only useful if it changes execution policy.  This
        method connects a detector decision to the fail-closed live manager:
        correctness drift blacklists recent templates, negative graph drift
        increases shadow validation cadence, and distribution/useful-coverage
        drift moves admitted templates back to shadow validation.
        """
        if isinstance(decision, DriftDecision):
            drifted = decision.drifted
            action = decision.action
            reason = decision.reason
        else:
            drifted = bool(decision.get("drifted", False))
            action = str(decision.get("action", "keep_policy"))
            reason = str(decision.get("reason", "unknown"))
        if not drifted:
            return []

        recent = {
            str(item)
            for item in (recent_template_ids or [])
            if item is not None and str(item) in self.records
        }
        affected: list[str] = []
        full_reason = f"{reason_prefix}:{reason}"

        if action == "blacklist_recent_templates":
            for template_id in sorted(recent):
                record = self.records[template_id]
                if record.status == LiveTemplateStatus.EVICTED:
                    continue
                record.status = LiveTemplateStatus.BLACKLISTED
                record.last_reason = full_reason
                affected.append(template_id)
        elif action == "increase_shadow_validation":
            self.shadow_validation_multiplier = min(
                self.shadow_validation_multiplier * 2,
                16,
            )
            for template_id in sorted(recent):
                record = self.records[template_id]
                if record.status == LiveTemplateStatus.ADMITTED:
                    record.last_reason = full_reason
                    affected.append(template_id)
        elif action in {"explore_new_templates", "refresh_admission"}:
            targets = recent or {
                template_id
                for template_id, record in self.records.items()
                if record.status == LiveTemplateStatus.ADMITTED
            }
            for template_id in sorted(targets):
                record = self.records[template_id]
                if record.status == LiveTemplateStatus.ADMITTED:
                    record.status = LiveTemplateStatus.SHADOW_VALIDATING
                    record.last_reason = full_reason
                    affected.append(template_id)

        self.events.append({
            "event": "drift_action",
            "reason": reason,
            "action": action,
            "affected": affected,
            "shadow_validation_multiplier": self.shadow_validation_multiplier,
        })
        return affected

    def _over_budget(self) -> bool:
        active = [
            record for record in self.records.values()
            if record.status not in {LiveTemplateStatus.EVICTED, LiveTemplateStatus.BLACKLISTED}
            and record.handle is not None
        ]
        if self.max_templates and len(active) > self.max_templates:
            return True
        if self.max_graph_memory_bytes:
            total = sum(record.memory_bytes for record in active)
            if total > self.max_graph_memory_bytes:
                return True
        return False

    def _select_victim(self) -> LiveTemplateRecord | None:
        candidates = [
            record for record in self.records.values()
            if record.handle is not None
            and record.status not in {LiveTemplateStatus.EVICTED, LiveTemplateStatus.BLACKLISTED}
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda record: (
                record.status == LiveTemplateStatus.ADMITTED,
                record.score(),
                record.last_used_s,
            ),
        )

    def _should_shadow(
        self,
        record: LiveTemplateRecord,
        callbacks: LiveCaptureCallbacks,
    ) -> bool:
        if callbacks.should_shadow is not None:
            return bool(callbacks.should_shadow(record.spec, record.replays))
        interval = max(
            1,
            self.validation_interval // max(1, self.shadow_validation_multiplier),
        )
        return record.replays > 0 and record.replays % interval == 0

    def _validate(
        self,
        callbacks: LiveCaptureCallbacks,
        graph_output: Any,
        fallback_output: Any,
        context: dict[str, Any],
    ) -> ValidationResult:
        if callbacks.validate is None:
            return ValidationResult(correct=graph_output == fallback_output)
        return callbacks.validate(graph_output, fallback_output, context)

    def _admission(self, record: LiveTemplateRecord) -> tuple[bool, str]:
        if record.correctness_failures > self.max_correctness_failures:
            return False, "correctness_failure"
        if record.samples < self.min_samples:
            return False, "explore_until_min_samples"
        if record.useful_rate < self.min_useful_rate:
            return False, "low_useful_rate"
        saving = record.avg_saving_ms
        amortized = record.amortized_capture_ms / self.amortization_replays
        if saving is None or saving - amortized < self.min_saving_ms:
            return False, "insufficient_latency_saving"
        if record.p95_regression_ms > self.max_p95_regression_ms:
            return False, "tail_regression_guard"
        return True, "admitted"

    def export_policy(self, *, default_action: str = "fallback") -> dict[str, Any]:
        rules = []
        for record in self.records.values():
            if record.status != LiveTemplateStatus.ADMITTED:
                continue
            rule = {
                "lo": int(record.spec.lo),
                "hi": int(record.spec.hi),
                "action": record.spec.action,
                "n": record.samples,
                "reason": (
                    "same-engine-live-admitted: "
                    f"useful_rate={record.useful_rate:.2f}, "
                    f"avg_saving_ms={record.avg_saving_ms or 0.0:.2f}"
                ),
            }
            if record.spec.template_tokens is not None:
                rule["template_tokens"] = int(record.spec.template_tokens)
            rules.append(rule)
        return {
            "kind": "same_engine_live_capture_policy",
            "default_action": default_action,
            "correctness_required": True,
            "rules": sorted(rules, key=lambda row: (row["lo"], row["hi"])),
            "live_capture": self.summary(include_events=False),
        }

    def summary(self, *, include_events: bool = True) -> dict[str, Any]:
        active_memory = sum(
            record.memory_bytes for record in self.records.values()
            if record.status not in {LiveTemplateStatus.EVICTED, LiveTemplateStatus.BLACKLISTED}
        )
        payload = {
            "mode": "same_engine_live_capture",
            "min_samples": self.min_samples,
            "min_useful_rate": self.min_useful_rate,
            "min_saving_ms": self.min_saving_ms,
            "max_p95_regression_ms": self.max_p95_regression_ms,
            "max_templates": self.max_templates,
            "max_graph_memory_bytes": self.max_graph_memory_bytes,
            "active_memory_bytes": active_memory,
            "shadow_validation_multiplier": self.shadow_validation_multiplier,
            "templates": [record.to_json() for record in self.records.values()],
        }
        if include_events:
            payload["events"] = self.events
        return payload


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(x) for x in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * float(pct) / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)
