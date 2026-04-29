from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import mean
from typing import Any, Iterable


DEFAULT_RESIDUAL_BUCKETS = [
    512,
    640,
    768,
    832,
    896,
    1024,
    1280,
    1536,
    2048,
    3072,
    4096,
]


def sglang_piecewise_token_buckets(max_tokens: int = 4096) -> list[int]:
    """Return the token buckets used by SGLang-style piecewise CUDA Graphs."""
    max_tokens = int(max_tokens)
    buckets: list[int] = []
    buckets.extend(range(4, min(32, max_tokens) + 1, 4))
    buckets.extend(range(48, min(256, max_tokens) + 1, 16))
    buckets.extend(range(288, min(512, max_tokens) + 1, 32))
    buckets.extend(range(576, min(1024, max_tokens) + 1, 64))
    buckets.extend(range(1280, min(4096, max_tokens) + 1, 256))
    if max_tokens > 4096:
        buckets.extend(range(4608, max_tokens + 1, 512))
    return sorted({int(bucket) for bucket in buckets if int(bucket) <= max_tokens})


SGLANG_PCG_RESIDUAL_BUCKETS = sglang_piecewise_token_buckets(4096)


def residual_buckets_for_preset(
    preset: str,
    *,
    max_tokens: int = 4096,
) -> list[int]:
    """Resolve a named residual-capture bucket preset."""
    name = str(preset or "default").strip().lower()
    if name == "default":
        buckets = DEFAULT_RESIDUAL_BUCKETS
    elif name in {"sglang", "sglang-pcg", "pcg"}:
        buckets = sglang_piecewise_token_buckets(max_tokens)
    elif name in {"default+sglang", "default+sglang-pcg", "combined"}:
        buckets = sorted(
            set(DEFAULT_RESIDUAL_BUCKETS)
            | set(sglang_piecewise_token_buckets(max_tokens))
        )
    else:
        raise ValueError(f"unknown residual bucket preset: {preset!r}")
    return [int(bucket) for bucket in buckets if int(bucket) <= int(max_tokens)]


@dataclass
class ResidualCaptureObservation:
    idx: int
    tokens: int
    fallback_ms: float
    graph_ms: float
    correct: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def delta_ms(self) -> float:
        return float(self.graph_ms) - float(self.fallback_ms)

    @property
    def saving_ms(self) -> float:
        return -self.delta_ms

    @property
    def useful(self) -> bool:
        return bool(self.correct) and self.delta_ms < 0.0


@dataclass
class ResidualCaptureCandidate:
    lo: int
    hi: int
    template_tokens: int
    observations: list[ResidualCaptureObservation]
    capture_ms: float = 0.0
    warmup_ms: float = 0.0
    amortization_replays: int = 32
    action: str = "ours_cp"
    source: str = "residual"
    admit: bool = False
    reason: str = "not_evaluated"

    @property
    def n(self) -> int:
        return len(self.observations)

    @property
    def wins(self) -> int:
        return sum(1 for obs in self.observations if obs.useful)

    @property
    def losses(self) -> int:
        return self.n - self.wins

    @property
    def mismatches(self) -> int:
        return sum(1 for obs in self.observations if not obs.correct)

    @property
    def useful_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def avg_delta_ms(self) -> float | None:
        if not self.observations:
            return None
        return mean(obs.delta_ms for obs in self.observations)

    @property
    def avg_saving_ms(self) -> float | None:
        avg = self.avg_delta_ms
        return None if avg is None else -avg

    @property
    def p95_regression_ms(self) -> float | None:
        regressions = sorted(
            obs.delta_ms for obs in self.observations
            if obs.correct and obs.delta_ms >= 0.0
        )
        if not regressions:
            return 0.0 if self.observations else None
        return percentile(regressions, 95)

    @property
    def max_regression_ms(self) -> float | None:
        if not self.observations:
            return None
        return max(obs.delta_ms for obs in self.observations)

    @property
    def p95_graph_ms(self) -> float | None:
        if not self.observations:
            return None
        return percentile([obs.graph_ms for obs in self.observations], 95)

    @property
    def p95_fallback_ms(self) -> float | None:
        if not self.observations:
            return None
        return percentile([obs.fallback_ms for obs in self.observations], 95)

    @property
    def p95_saving_ms(self) -> float | None:
        graph = self.p95_graph_ms
        fallback = self.p95_fallback_ms
        if graph is None or fallback is None:
            return None
        return float(fallback) - float(graph)

    @property
    def amortized_capture_ms(self) -> float:
        return (float(self.capture_ms) + float(self.warmup_ms)) / max(
            1, int(self.amortization_replays)
        )

    @property
    def effective_saving_ms(self) -> float | None:
        saving = self.avg_saving_ms
        if saving is None:
            return None
        return saving - self.amortized_capture_ms

    @property
    def score(self) -> float:
        saving = self.effective_saving_ms
        if saving is None:
            return 0.0
        return max(0.0, saving) * self.n

    def to_rule(self) -> dict[str, Any]:
        return {
            "lo": int(self.lo),
            "hi": int(self.hi),
            "action": self.action,
            "n": int(self.n),
            "template_tokens": int(self.template_tokens),
            "reason": (
                "residual-capture admitted: "
                f"useful_rate={self.useful_rate:.2f}, "
                f"avg_saving_ms={self.avg_saving_ms or 0.0:.2f}, "
                f"effective_saving_ms={self.effective_saving_ms or 0.0:.2f}, "
                f"p95_saving_ms={self.p95_saving_ms or 0.0:.2f}"
            ),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "lo": int(self.lo),
            "hi": int(self.hi),
            "template_tokens": int(self.template_tokens),
            "n": int(self.n),
            "wins": int(self.wins),
            "losses": int(self.losses),
            "mismatches": int(self.mismatches),
            "useful_rate": float(self.useful_rate),
            "avg_delta_ms": self.avg_delta_ms,
            "avg_saving_ms": self.avg_saving_ms,
            "effective_saving_ms": self.effective_saving_ms,
            "p95_graph_ms": self.p95_graph_ms,
            "p95_fallback_ms": self.p95_fallback_ms,
            "p95_saving_ms": self.p95_saving_ms,
            "p95_regression_ms": self.p95_regression_ms,
            "max_regression_ms": self.max_regression_ms,
            "capture_ms": float(self.capture_ms),
            "warmup_ms": float(self.warmup_ms),
            "amortized_capture_ms": float(self.amortized_capture_ms),
            "amortization_replays": int(self.amortization_replays),
            "action": self.action,
            "source": self.source,
            "admit": bool(self.admit),
            "reason": self.reason,
            "tokens": [int(obs.tokens) for obs in self.observations],
            "indices": [int(obs.idx) for obs in self.observations],
        }


@dataclass
class ResidualCapturePlan:
    policy: dict[str, Any]
    admitted: list[ResidualCaptureCandidate]
    rejected: list[ResidualCaptureCandidate]
    observations: list[ResidualCaptureObservation]
    residual_observations: list[ResidualCaptureObservation]
    seed_policy: dict[str, Any] | None = None

    @property
    def extra_capture_sizes(self) -> list[int]:
        base = int(self.policy.get("single_engine_base_capture_size", 512) or 512)
        sizes = {
            int(rule["template_tokens"])
            for rule in self.policy.get("rules", [])
            if isinstance(rule, dict)
            and rule.get("template_tokens") is not None
            and int(rule.get("template_tokens", 0)) > base
            and str(rule.get("action", "")).endswith("cp")
        }
        return sorted(sizes)

    @property
    def admitted_residual_capture_sizes(self) -> list[int]:
        base = int(self.policy.get("single_engine_base_capture_size", 512) or 512)
        return sorted({
            int(candidate.template_tokens)
            for candidate in self.admitted
            if int(candidate.template_tokens) > base
        })

    def to_json(self) -> dict[str, Any]:
        return {
            "runtime_policy": self.policy,
            "analysis": {
                "observations": [obs_to_json(obs) for obs in self.observations],
                "residual_observations": [
                    obs_to_json(obs) for obs in self.residual_observations
                ],
                "admitted": [candidate.to_json() for candidate in self.admitted],
                "rejected": [candidate.to_json() for candidate in self.rejected],
                "extra_capture_sizes": self.extra_capture_sizes,
                "admitted_residual_capture_sizes": self.admitted_residual_capture_sizes,
                "global_stats": observation_stats(self.observations),
                "residual_stats": observation_stats(self.residual_observations),
            },
        }


class ResidualCapturePlanner:
    """Learn next-round graph templates from correct but residual misses.

    The planner is deliberately fail-closed: correctness failures are never
    admitted, and candidates must beat fallback after capture/warmup
    amortization.  It can preserve an existing runtime policy and only replace
    fallback windows where residual exploration found a profitable template.
    """

    def __init__(
        self,
        *,
        template_buckets: Iterable[int] = DEFAULT_RESIDUAL_BUCKETS,
        min_samples: int = 2,
        min_useful_rate: float = 0.75,
        min_saving_ms: float = 0.5,
        min_p95_saving_ms: float | None = None,
        max_p95_regression_ms: float = 2.0,
        max_regression_ms: float = 5.0,
        min_template_tokens: int = 0,
        require_exact_template_evidence: bool = True,
        tail_token_threshold: int = 0,
        tail_min_samples: int | None = None,
        tail_min_useful_rate: float | None = None,
        tail_min_saving_ms: float | None = None,
        tail_min_p95_saving_ms: float | None = None,
        tail_max_p95_regression_ms: float | None = None,
        tail_max_regression_ms: float | None = None,
        max_segments: int = 4,
        capture_ms_per_template: float = 0.0,
        warmup_ms_per_template: float = 0.0,
        amortization_replays: int = 32,
        graph_action: str = "ours_cp",
        default_action: str = "cp",
        base_capture_size: int = 512,
        residual_only: bool = True,
        require_candidate_graph: bool = True,
    ):
        self.template_buckets = sorted({int(x) for x in template_buckets})
        self.min_samples = int(min_samples)
        self.min_useful_rate = float(min_useful_rate)
        self.min_saving_ms = float(min_saving_ms)
        self.min_p95_saving_ms = (
            None if min_p95_saving_ms is None else float(min_p95_saving_ms)
        )
        self.max_p95_regression_ms = float(max_p95_regression_ms)
        self.max_regression_ms = float(max_regression_ms)
        self.min_template_tokens = int(min_template_tokens or 0)
        self.require_exact_template_evidence = bool(require_exact_template_evidence)
        self.tail_token_threshold = int(tail_token_threshold or 0)
        self.tail_min_samples = (
            None if tail_min_samples is None else int(tail_min_samples)
        )
        self.tail_min_useful_rate = (
            None if tail_min_useful_rate is None
            else float(tail_min_useful_rate)
        )
        self.tail_min_saving_ms = (
            None if tail_min_saving_ms is None else float(tail_min_saving_ms)
        )
        self.tail_min_p95_saving_ms = (
            None if tail_min_p95_saving_ms is None
            else float(tail_min_p95_saving_ms)
        )
        self.tail_max_p95_regression_ms = (
            None if tail_max_p95_regression_ms is None
            else float(tail_max_p95_regression_ms)
        )
        self.tail_max_regression_ms = (
            None if tail_max_regression_ms is None
            else float(tail_max_regression_ms)
        )
        self.max_segments = int(max_segments)
        self.capture_ms_per_template = float(capture_ms_per_template)
        self.warmup_ms_per_template = float(warmup_ms_per_template)
        self.amortization_replays = int(amortization_replays)
        self.graph_action = graph_action
        self.default_action = default_action
        self.base_capture_size = int(base_capture_size)
        self.residual_only = bool(residual_only)
        self.require_candidate_graph = bool(require_candidate_graph)

    def plan(
        self,
        observations: Iterable[ResidualCaptureObservation | dict[str, Any]],
        *,
        seed_policy: dict[str, Any] | None = None,
    ) -> ResidualCapturePlan:
        obs = [
            coerce_observation(item, idx=idx)
            for idx, item in enumerate(observations)
        ]
        residual = [
            item for item in obs
            if self._is_residual_observation(item, seed_policy)
        ]
        candidate_pool_source = residual if self.residual_only else obs
        candidate_pool = [
            item for item in candidate_pool_source
            if self._candidate_graph_observed(item)
        ]
        candidates = self._generate_candidates(candidate_pool)
        selected, rejected = self._select_candidates(candidates)
        policy = self._build_policy(obs, selected, seed_policy)
        return ResidualCapturePlan(
            policy=policy,
            admitted=selected,
            rejected=rejected,
            observations=obs,
            residual_observations=residual,
            seed_policy=seed_policy,
        )

    def _candidate_graph_observed(self, obs: ResidualCaptureObservation) -> bool:
        if not self.require_candidate_graph:
            return True
        if "candidate_graph_allowed" not in obs.metadata:
            return True
        return bool(obs.metadata.get("candidate_graph_allowed"))

    def _is_residual_observation(
        self,
        obs: ResidualCaptureObservation,
        seed_policy: dict[str, Any] | None,
    ) -> bool:
        if not self.residual_only or not seed_policy:
            return True
        allowed, rule = policy_graph_covers(
            seed_policy,
            obs.tokens,
            base_capture_size=self.base_capture_size,
        )
        if not allowed:
            return True
        if rule is None:
            return False
        return False

    def _generate_candidates(
        self,
        observations: list[ResidualCaptureObservation],
    ) -> list[ResidualCaptureCandidate]:
        ordered = sorted(observations, key=lambda item: (item.tokens, item.idx))
        n = len(ordered)
        candidates: list[ResidualCaptureCandidate] = []
        for left in range(n):
            for right in range(left, n):
                lo = int(ordered[left].tokens) - 1
                hi = int(ordered[right].tokens)
                slice_obs = [
                    obs for obs in ordered
                    if lo < int(obs.tokens) <= hi
                ]
                template = template_for_tokens(hi, self.template_buckets)
                if template is None:
                    continue
                if int(template) < self.min_template_tokens:
                    continue
                if (
                    self.require_exact_template_evidence
                    and int(template) > int(self.base_capture_size)
                    and not all(
                        obs.metadata.get("candidate_template_tokens") is not None
                        and int(obs.metadata.get("candidate_template_tokens")) == int(template)
                        for obs in slice_obs
                    )
                ):
                    continue
                candidate = ResidualCaptureCandidate(
                    lo=lo,
                    hi=hi,
                    template_tokens=template,
                    observations=slice_obs,
                    capture_ms=self.capture_ms_per_template,
                    warmup_ms=self.warmup_ms_per_template,
                    amortization_replays=self.amortization_replays,
                    action=self.graph_action,
                )
                candidate.admit, candidate.reason = self._evaluate(candidate)
                if candidate.admit:
                    candidates.append(candidate)
        return candidates

    def _is_tail_candidate(self, candidate: ResidualCaptureCandidate) -> bool:
        return (
            self.tail_token_threshold > 0
            and int(candidate.hi) >= self.tail_token_threshold
        )

    def _thresholds_for(
        self,
        candidate: ResidualCaptureCandidate,
    ) -> dict[str, float | int | None]:
        if not self._is_tail_candidate(candidate):
            return {
                "min_samples": self.min_samples,
                "min_useful_rate": self.min_useful_rate,
                "min_saving_ms": self.min_saving_ms,
                "min_p95_saving_ms": self.min_p95_saving_ms,
                "max_p95_regression_ms": self.max_p95_regression_ms,
                "max_regression_ms": self.max_regression_ms,
            }
        return {
            "min_samples": (
                self.min_samples
                if self.tail_min_samples is None
                else self.tail_min_samples
            ),
            "min_useful_rate": (
                self.min_useful_rate
                if self.tail_min_useful_rate is None
                else self.tail_min_useful_rate
            ),
            "min_saving_ms": (
                self.min_saving_ms
                if self.tail_min_saving_ms is None
                else self.tail_min_saving_ms
            ),
            "min_p95_saving_ms": (
                self.min_p95_saving_ms
                if self.tail_min_p95_saving_ms is None
                else self.tail_min_p95_saving_ms
            ),
            "max_p95_regression_ms": (
                self.max_p95_regression_ms
                if self.tail_max_p95_regression_ms is None
                else self.tail_max_p95_regression_ms
            ),
            "max_regression_ms": (
                self.max_regression_ms
                if self.tail_max_regression_ms is None
                else self.tail_max_regression_ms
            ),
        }

    def _evaluate(self, candidate: ResidualCaptureCandidate) -> tuple[bool, str]:
        thresholds = self._thresholds_for(candidate)
        if candidate.n < int(thresholds["min_samples"] or 0):
            return False, "insufficient_samples"
        if candidate.mismatches:
            return False, "correctness_failure"
        if candidate.useful_rate < float(thresholds["min_useful_rate"] or 0.0):
            return False, "low_useful_rate"
        saving = candidate.effective_saving_ms
        if saving is None or saving < float(thresholds["min_saving_ms"] or 0.0):
            return False, "insufficient_latency_saving"
        min_p95_saving = thresholds["min_p95_saving_ms"]
        if min_p95_saving is not None:
            p95_saving = candidate.p95_saving_ms
            if p95_saving is None or p95_saving < float(min_p95_saving):
                return False, "insufficient_p95_latency_saving"
        p95 = candidate.p95_regression_ms
        max_p95_reg = thresholds["max_p95_regression_ms"]
        if p95 is not None and p95 > float(max_p95_reg):
            return False, "tail_regression_guard"
        max_reg = candidate.max_regression_ms
        max_reg_limit = thresholds["max_regression_ms"]
        if max_reg is not None and max_reg > float(max_reg_limit):
            return False, "max_regression_guard"
        return True, "admitted"

    def _select_candidates(
        self,
        candidates: list[ResidualCaptureCandidate],
    ) -> tuple[list[ResidualCaptureCandidate], list[ResidualCaptureCandidate]]:
        if not candidates:
            return [], []

        intervals = sorted(candidates, key=lambda item: (item.hi, item.lo, -item.score))
        prev: list[int] = []
        for idx, item in enumerate(intervals):
            j = idx - 1
            while j >= 0 and intervals[j].hi > item.lo:
                j -= 1
            prev.append(j)

        max_segments = max(1, self.max_segments)
        dp = [[0.0] * (len(intervals) + 1) for _ in range(max_segments + 1)]
        take = [[False] * (len(intervals) + 1) for _ in range(max_segments + 1)]
        for k in range(1, max_segments + 1):
            for i in range(1, len(intervals) + 1):
                skip = dp[k][i - 1]
                item = intervals[i - 1]
                with_item = item.score + dp[k - 1][prev[i - 1] + 1]
                if with_item > skip:
                    dp[k][i] = with_item
                    take[k][i] = True
                else:
                    dp[k][i] = skip

        selected: list[ResidualCaptureCandidate] = []
        k = max_segments
        i = len(intervals)
        while k > 0 and i > 0:
            if take[k][i]:
                item = intervals[i - 1]
                selected.append(item)
                i = prev[i - 1] + 1
                k -= 1
            else:
                i -= 1
        selected = sorted(selected, key=lambda item: (item.lo, item.hi))
        selected_ids = {(item.lo, item.hi, item.template_tokens) for item in selected}
        rejected = [
            item for item in intervals
            if (item.lo, item.hi, item.template_tokens) not in selected_ids
        ]
        for item in rejected:
            if item.reason == "admitted":
                item.admit = False
                item.reason = "not_selected_by_interval_dp"
        return selected, rejected

    def _build_policy(
        self,
        observations: list[ResidualCaptureObservation],
        selected: list[ResidualCaptureCandidate],
        seed_policy: dict[str, Any] | None,
    ) -> dict[str, Any]:
        seed = unwrap_runtime_policy(seed_policy)
        seed_rules = normalize_rules(seed) if seed else []
        max_tok = max([obs.tokens for obs in observations], default=max(self.template_buckets))
        boundaries = {0, max_tok}
        for rule in seed_rules:
            bounds = range_bounds(rule)
            if bounds is None:
                continue
            boundaries.update(bounds)
        for candidate in selected:
            boundaries.add(candidate.lo)
            boundaries.add(candidate.hi)
        sorted_bounds = sorted(boundaries)

        rules: list[dict[str, Any]] = []
        for left, right in zip(sorted_bounds, sorted_bounds[1:]):
            if left == right:
                continue
            residual_rule = first_covering_candidate(selected, left, right)
            if residual_rule is not None:
                rule = residual_rule.to_rule()
                rule["lo"] = int(left)
                rule["hi"] = int(right)
                rule["n"] = count_obs_in_range(observations, left, right)
            else:
                seed_rule = first_covering_rule(seed_rules, left, right)
                if seed_rule is not None:
                    rule = copy_policy_rule(seed_rule, left, right)
                    rule.setdefault(
                        "reason",
                        "preserved seed runtime policy rule",
                    )
                else:
                    rule = {
                        "lo": int(left),
                        "hi": int(right),
                        "action": self.default_action,
                        "n": count_obs_in_range(observations, left, right),
                        "reason": "fallback: no admitted residual graph template",
                    }
            rule.setdefault("n", count_obs_in_range(observations, left, right))
            rules.append(rule)

        graph_actions = ["default", "ours", "cp", self.graph_action]
        graph_actions = dedupe_preserve_order(graph_actions)
        arena_ranges = [
            {
                "lo": int(rule["lo"]),
                "hi": int(rule["hi"]),
                "template_tokens": int(rule["template_tokens"]),
                "action": rule.get("action", self.graph_action),
                "n": int(rule.get("n", 0)),
            }
            for rule in rules
            if rule.get("template_tokens") is not None
            and str(rule.get("action", "")) in graph_actions
        ]
        extra_sizes = sorted({
            int(row["template_tokens"])
            for row in arena_ranges
            if int(row["template_tokens"]) > self.base_capture_size
        })
        policy = dict(seed or {})
        policy.update({
            "kind": "residual_capture_runtime_policy",
            "default_action": self.default_action,
            "baseline_action": self.default_action,
            "correctness_required": True,
            "rules": rules,
            "fixed_metadata_arena_ranges": arena_ranges,
            "single_engine_graph_actions": graph_actions,
            "single_engine_fallback_actions": [
                "eager",
                "compile",
                "compiled",
                "fallback",
                "none",
            ],
            "single_engine_allow_multi_req_extra": True,
            "single_engine_requires_fixed_metadata_arena": True,
            "single_engine_max_extra_templates": max(1, len(extra_sizes)),
            "single_engine_min_rule_n": 0,
            "single_engine_base_capture_size": self.base_capture_size,
            "residual_capture": {
                "mode": "next_round_residual_capture",
                "residual_only": self.residual_only,
                "require_candidate_graph": self.require_candidate_graph,
                "template_buckets": self.template_buckets,
                "extra_capture_sizes": extra_sizes,
                "min_samples": self.min_samples,
                "min_useful_rate": self.min_useful_rate,
                "min_saving_ms": self.min_saving_ms,
                "min_p95_saving_ms": self.min_p95_saving_ms,
                "max_p95_regression_ms": self.max_p95_regression_ms,
                "max_regression_ms": self.max_regression_ms,
                "min_template_tokens": self.min_template_tokens,
                "require_exact_template_evidence": self.require_exact_template_evidence,
                "tail_token_threshold": self.tail_token_threshold,
                "tail_min_samples": self.tail_min_samples,
                "tail_min_useful_rate": self.tail_min_useful_rate,
                "tail_min_saving_ms": self.tail_min_saving_ms,
                "tail_min_p95_saving_ms": self.tail_min_p95_saving_ms,
                "tail_max_p95_regression_ms": self.tail_max_p95_regression_ms,
                "tail_max_regression_ms": self.tail_max_regression_ms,
                "max_segments": self.max_segments,
                "capture_ms_per_template": self.capture_ms_per_template,
                "warmup_ms_per_template": self.warmup_ms_per_template,
                "amortization_replays": self.amortization_replays,
                "admitted": [candidate.to_json() for candidate in selected],
            },
            "online_admission": {
                "mode": "online_self_learning_admission",
                "source": "residual_capture_planner",
                "min_samples": self.min_samples,
                "min_useful_rate": self.min_useful_rate,
                "min_saving_ms": self.min_saving_ms,
                "max_p95_regression_ms": self.max_p95_regression_ms,
                "min_template_tokens": self.min_template_tokens,
                "require_exact_template_evidence": self.require_exact_template_evidence,
                "tail_token_threshold": self.tail_token_threshold,
                "tail_min_samples": self.tail_min_samples,
                "tail_min_useful_rate": self.tail_min_useful_rate,
                "tail_min_saving_ms": self.tail_min_saving_ms,
                "tail_min_p95_saving_ms": self.tail_min_p95_saving_ms,
                "tail_max_p95_regression_ms": self.tail_max_p95_regression_ms,
                "tail_max_regression_ms": self.tail_max_regression_ms,
                "max_correctness_failures": 0,
                "amortization_replays": self.amortization_replays,
                "templates": [
                    candidate_to_online_template(candidate)
                    for candidate in selected
                ],
            },
        })
        return policy


def coerce_observation(
    item: ResidualCaptureObservation | dict[str, Any],
    *,
    idx: int,
) -> ResidualCaptureObservation:
    if isinstance(item, ResidualCaptureObservation):
        return item
    tokens = item.get("tokens", item.get("tok", item.get("num_tokens", 0)))
    fallback = item.get("fallback_ms", item.get("baseline_ms"))
    graph = item.get("graph_ms", item.get("candidate_ms", item.get("ms")))
    if fallback is None or graph is None:
        raise ValueError(f"observation {idx} missing fallback_ms/graph_ms: {item}")
    return ResidualCaptureObservation(
        idx=int(item.get("idx", idx)),
        tokens=int(tokens),
        fallback_ms=float(fallback),
        graph_ms=float(graph),
        correct=bool(item.get("correct", item.get("token_correct", True))),
        metadata=dict(item.get("metadata") or {}),
    )


def template_for_tokens(tokens: int, buckets: Iterable[int]) -> int | None:
    for bucket in sorted(int(x) for x in buckets):
        if int(tokens) <= bucket:
            return int(bucket)
    return None


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - pos) + values[hi] * (pos - lo))


def observation_stats(observations: list[ResidualCaptureObservation]) -> dict[str, Any]:
    if not observations:
        return {
            "n": 0,
            "wins": 0,
            "losses": 0,
            "mismatches": 0,
            "useful_rate": 0.0,
            "avg_delta_ms": None,
            "avg_saving_ms": None,
            "p95_regression_ms": None,
            "max_regression_ms": None,
        }
    candidate = ResidualCaptureCandidate(
        lo=min(obs.tokens for obs in observations) - 1,
        hi=max(obs.tokens for obs in observations),
        template_tokens=max(obs.tokens for obs in observations),
        observations=observations,
    )
    return {
        "n": candidate.n,
        "wins": candidate.wins,
        "losses": candidate.losses,
        "mismatches": candidate.mismatches,
        "useful_rate": candidate.useful_rate,
        "avg_delta_ms": candidate.avg_delta_ms,
        "avg_saving_ms": candidate.avg_saving_ms,
        "p95_graph_ms": candidate.p95_graph_ms,
        "p95_fallback_ms": candidate.p95_fallback_ms,
        "p95_saving_ms": candidate.p95_saving_ms,
        "p95_regression_ms": candidate.p95_regression_ms,
        "max_regression_ms": candidate.max_regression_ms,
    }


def obs_to_json(obs: ResidualCaptureObservation) -> dict[str, Any]:
    return {
        "idx": int(obs.idx),
        "tokens": int(obs.tokens),
        "fallback_ms": float(obs.fallback_ms),
        "graph_ms": float(obs.graph_ms),
        "delta_ms": float(obs.delta_ms),
        "saving_ms": float(obs.saving_ms),
        "correct": bool(obs.correct),
        "useful": bool(obs.useful),
        "metadata": dict(obs.metadata),
    }


def unwrap_runtime_policy(policy: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(policy, dict):
        return None
    nested = policy.get("runtime_policy")
    if isinstance(nested, dict):
        return nested
    return policy


def range_bounds(rule: dict[str, Any]) -> tuple[int, int] | None:
    try:
        return int(rule.get("lo", rule.get("low"))), int(rule.get("hi", rule.get("high")))
    except (TypeError, ValueError):
        return None


def normalize_rules(policy: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(policy, dict):
        return []
    raw = policy.get("rules", [])
    if isinstance(raw, list):
        return [rule for rule in raw if isinstance(rule, dict) and range_bounds(rule)]
    if not isinstance(raw, dict):
        return []
    rules: list[dict[str, Any]] = []
    for name, body in raw.items():
        if not isinstance(body, dict) or "," not in str(name):
            continue
        try:
            left, right = str(name).strip().split(",", 1)
            lo = int(left.strip().lstrip("(").lstrip("["))
            hi = int(right.strip().rstrip("]").rstrip(")"))
        except ValueError:
            continue
        rule = dict(body)
        rule.setdefault("lo", lo)
        rule.setdefault("hi", hi)
        rules.append(rule)
    return rules


def policy_graph_covers(
    policy: dict[str, Any] | None,
    tokens: int,
    *,
    base_capture_size: int = 512,
) -> tuple[bool, dict[str, Any] | None]:
    runtime_policy = unwrap_runtime_policy(policy)
    if not runtime_policy:
        return False, None
    graph_actions = set(
        runtime_policy.get("single_engine_graph_actions")
        or ["default", "ours", "cp", "ours_cp"]
    )
    default_action = str(runtime_policy.get("default_action", "default"))
    matched = None
    action = default_action
    for rule in normalize_rules(runtime_policy):
        bounds = range_bounds(rule)
        if bounds is None:
            continue
        lo, hi = bounds
        if lo < int(tokens) <= hi:
            matched = rule
            action = str(rule.get("action", default_action))
            break
    if action not in graph_actions:
        return False, matched
    if action in {"default", "cp"} and int(tokens) > int(
        runtime_policy.get("single_engine_base_capture_size", base_capture_size)
    ):
        return False, matched
    return True, matched


def first_covering_candidate(
    candidates: list[ResidualCaptureCandidate],
    left: int,
    right: int,
) -> ResidualCaptureCandidate | None:
    for candidate in candidates:
        if int(candidate.lo) <= int(left) and int(candidate.hi) >= int(right):
            return candidate
    return None


def first_covering_rule(
    rules: list[dict[str, Any]],
    left: int,
    right: int,
) -> dict[str, Any] | None:
    for rule in rules:
        bounds = range_bounds(rule)
        if bounds is None:
            continue
        lo, hi = bounds
        if int(lo) <= int(left) and int(hi) >= int(right):
            return rule
    return None


def copy_policy_rule(rule: dict[str, Any], left: int, right: int) -> dict[str, Any]:
    copied = dict(rule)
    copied["lo"] = int(left)
    copied["hi"] = int(right)
    return copied


def count_obs_in_range(
    observations: list[ResidualCaptureObservation],
    left: int,
    right: int,
) -> int:
    return sum(1 for obs in observations if int(left) < int(obs.tokens) <= int(right))


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def candidate_to_online_template(candidate: ResidualCaptureCandidate) -> dict[str, Any]:
    return {
        "template_id": f"tokens={int(candidate.template_tokens)}",
        "samples": int(candidate.n),
        "useful": int(candidate.wins),
        "regressions": sum(
            1 for obs in candidate.observations
            if obs.correct and obs.delta_ms >= 0.0
        ),
        "correctness_failures": int(candidate.mismatches),
        "useful_rate": float(candidate.useful_rate),
        "graph_ewma_ms": mean(obs.graph_ms for obs in candidate.observations),
        "fallback_ewma_ms": mean(obs.fallback_ms for obs in candidate.observations),
        "saving_ewma_ms": candidate.avg_saving_ms,
        "p95_regression_ms": candidate.p95_regression_ms,
        "capture_ms": float(candidate.capture_ms),
        "warmup_ms": float(candidate.warmup_ms),
        "disabled": False,
        "disable_reason": None,
        "metadata": {
            "lo": int(candidate.lo),
            "hi": int(candidate.hi),
            "template_tokens": int(candidate.template_tokens),
            "action": candidate.action,
        },
    }
