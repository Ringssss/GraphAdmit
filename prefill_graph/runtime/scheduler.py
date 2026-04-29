from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ScheduledBatch:
    template_id: str
    requests: list[dict[str, Any]]
    flush_time_ms: float
    wait_ms: list[float]


class TemplateAwareScheduler:
    """Bounded-wait scheduler that groups requests by static template."""

    def __init__(
        self,
        template_fn: Callable[[dict[str, Any]], str],
        *,
        max_wait_ms: float = 0.0,
        max_batch_size: int = 0,
        adaptive_wait: bool = False,
        adaptive_min_samples: int = 4,
        adaptive_min_hit_rate: float = 0.5,
    ):
        self.template_fn = template_fn
        self.max_wait_ms = max_wait_ms
        self.max_batch_size = max_batch_size
        self.adaptive_wait = adaptive_wait
        self.adaptive_min_samples = adaptive_min_samples
        self.adaptive_min_hit_rate = adaptive_min_hit_rate
        self.queue: deque[dict[str, Any]] = deque()
        self.batches: list[ScheduledBatch] = []
        self._last_arrival_by_template: dict[str, float] = {}
        self._gaps_by_template: dict[str, list[float]] = {}
        self._adaptive_enabled = 0
        self._adaptive_disabled = 0

    def add(self, request: dict[str, Any], arrival_ms: float) -> list[ScheduledBatch]:
        item = dict(request)
        item["_arrival_ms"] = float(arrival_ms)
        item["_template_id"] = self.template_fn(request)
        wait_budget_ms = request.get("scheduler_wait_budget_ms", self.max_wait_ms)
        wait_budget_ms = self._effective_wait_budget(
            item["_template_id"],
            item["_arrival_ms"],
            float(wait_budget_ms),
        )
        item["_wait_budget_ms"] = wait_budget_ms
        item["_deadline_ms"] = item["_arrival_ms"] + item["_wait_budget_ms"]
        self._record_arrival(item["_template_id"], item["_arrival_ms"])
        self.queue.append(item)
        return self._flush_ready(float(arrival_ms))

    def _effective_wait_budget(self, template_id: str, arrival_ms: float, requested_wait_ms: float) -> float:
        if requested_wait_ms <= 0.0 or not self.adaptive_wait:
            return max(0.0, requested_wait_ms)
        gaps = self._gaps_by_template.get(template_id, [])
        if len(gaps) < self.adaptive_min_samples:
            self._adaptive_disabled += 1
            return 0.0
        hits = sum(1 for gap in gaps if gap <= requested_wait_ms)
        hit_rate = hits / len(gaps)
        if hit_rate >= self.adaptive_min_hit_rate:
            self._adaptive_enabled += 1
            return max(0.0, requested_wait_ms)
        self._adaptive_disabled += 1
        return 0.0

    def _record_arrival(self, template_id: str, arrival_ms: float) -> None:
        previous = self._last_arrival_by_template.get(template_id)
        if previous is not None:
            self._gaps_by_template.setdefault(template_id, []).append(arrival_ms - previous)
        self._last_arrival_by_template[template_id] = arrival_ms

    def finish(self) -> list[ScheduledBatch]:
        flushed: list[ScheduledBatch] = []
        while self.queue:
            flushed.append(self._flush_template(self.queue[0]["_template_id"], self.queue[0]["_deadline_ms"]))
        return flushed

    def _flush_ready(self, now_ms: float) -> list[ScheduledBatch]:
        flushed: list[ScheduledBatch] = []
        while self.queue:
            first = self.queue[0]
            template_id = first["_template_id"]
            same_template = [item for item in self.queue if item["_template_id"] == template_id]
            if self.max_batch_size and len(same_template) >= self.max_batch_size:
                flushed.append(self._flush_template(template_id, now_ms))
                continue
            if first["_deadline_ms"] <= now_ms:
                flushed.append(self._flush_template(template_id, first["_deadline_ms"]))
                continue
            break
        return flushed

    def _flush_template(self, template_id: str, flush_time_ms: float) -> ScheduledBatch:
        selected = []
        kept = deque()
        while self.queue:
            item = self.queue.popleft()
            if (
                item["_template_id"] == template_id
                and item["_arrival_ms"] <= flush_time_ms
                and (not self.max_batch_size or len(selected) < self.max_batch_size)
            ):
                selected.append(item)
            else:
                kept.append(item)
        self.queue = kept
        waits = [float(flush_time_ms - item["_arrival_ms"]) for item in selected]
        wait_budgets = [float(item.get("_wait_budget_ms", self.max_wait_ms)) for item in selected]
        requests = [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in selected
        ]
        batch = ScheduledBatch(template_id, requests, float(flush_time_ms), waits)
        for request, wait_budget in zip(batch.requests, wait_budgets):
            request["scheduler_wait_budget_ms"] = wait_budget
        self.batches.append(batch)
        return batch

    def summary(self) -> dict[str, Any]:
        waits = [wait for batch in self.batches for wait in batch.wait_ms]
        wait_budgets = [
            float(request.get("scheduler_wait_budget_ms", self.max_wait_ms))
            for batch in self.batches
            for request in batch.requests
        ]
        return {
            "max_wait_ms": self.max_wait_ms,
            "max_batch_size": self.max_batch_size,
            "adaptive_wait": self.adaptive_wait,
            "adaptive_min_samples": self.adaptive_min_samples,
            "adaptive_min_hit_rate": self.adaptive_min_hit_rate,
            "adaptive_enabled": self._adaptive_enabled,
            "adaptive_disabled": self._adaptive_disabled,
            "num_batches": len(self.batches),
            "num_requests": sum(len(batch.requests) for batch in self.batches),
            "avg_batch_size": (
                sum(len(batch.requests) for batch in self.batches) / len(self.batches)
                if self.batches
                else 0.0
            ),
            "max_observed_wait_ms": max(waits) if waits else 0.0,
            "avg_wait_budget_ms": (
                sum(wait_budgets) / len(wait_budgets)
                if wait_budgets
                else 0.0
            ),
        }


@dataclass(frozen=True)
class TemplateSchedulingSignal:
    template_id: str
    admitted: bool
    expected_saving_ms: float
    p95_regression_ms: float = 0.0
    min_batch_size: int = 1


class SlaAwareTemplateScheduler(TemplateAwareScheduler):
    """Template scheduler that only waits when admission and SLA allow it."""

    def __init__(
        self,
        template_fn: Callable[[dict[str, Any]], str],
        *,
        signal_fn: Callable[[str], TemplateSchedulingSignal | None],
        max_wait_ms: float = 0.0,
        max_batch_size: int = 0,
        sla_p99_budget_ms: float | None = None,
        wait_fraction_of_saving: float = 0.5,
        adaptive_wait: bool = True,
        adaptive_min_samples: int = 4,
        adaptive_min_hit_rate: float = 0.5,
    ):
        super().__init__(
            template_fn,
            max_wait_ms=max_wait_ms,
            max_batch_size=max_batch_size,
            adaptive_wait=adaptive_wait,
            adaptive_min_samples=adaptive_min_samples,
            adaptive_min_hit_rate=adaptive_min_hit_rate,
        )
        self.signal_fn = signal_fn
        self.sla_p99_budget_ms = sla_p99_budget_ms
        self.wait_fraction_of_saving = float(wait_fraction_of_saving)
        self._sla_rejected = 0
        self._not_admitted_rejected = 0
        self._saving_rejected = 0

    def _effective_wait_budget(self, template_id: str, arrival_ms: float, requested_wait_ms: float) -> float:
        signal = self.signal_fn(template_id)
        if signal is None or not signal.admitted:
            self._not_admitted_rejected += 1
            return 0.0
        if signal.expected_saving_ms <= 0.0:
            self._saving_rejected += 1
            return 0.0
        wait_budget = min(
            requested_wait_ms,
            signal.expected_saving_ms * self.wait_fraction_of_saving,
        )
        if self.sla_p99_budget_ms is not None:
            request_deadline = self.sla_p99_budget_ms - max(0.0, signal.p95_regression_ms)
            if request_deadline <= 0.0:
                self._sla_rejected += 1
                return 0.0
            wait_budget = min(wait_budget, request_deadline)
        return super()._effective_wait_budget(template_id, arrival_ms, max(0.0, wait_budget))

    def summary(self) -> dict[str, Any]:
        payload = super().summary()
        payload.update({
            "sla_p99_budget_ms": self.sla_p99_budget_ms,
            "wait_fraction_of_saving": self.wait_fraction_of_saving,
            "sla_rejected": self._sla_rejected,
            "not_admitted_rejected": self._not_admitted_rejected,
            "saving_rejected": self._saving_rejected,
        })
        return payload
