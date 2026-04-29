from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DynamicityKind(str, Enum):
    SHAPE = "shape"
    METADATA = "metadata"
    ADDRESS = "address"
    CONTROL = "control"
    SCHEDULER = "scheduler"
    FEATURE = "feature"


@dataclass
class StaticityDecision:
    field: str
    kind: DynamicityKind
    semantic: bool
    in_graph_key: bool
    canonicalizable: bool
    method: str
    risk: str
    priority: str
    reason: str


@dataclass
class DynamicFieldProfile:
    field: str
    values_seen: int = 0
    unique_values: set[str] = field(default_factory=set)
    in_graph_key: bool = False
    semantic: bool | None = None
    examples: list[Any] = field(default_factory=list)

    def observe(
        self,
        value: Any,
        *,
        in_graph_key: bool = False,
        semantic: bool | None = None,
        max_examples: int = 5,
    ) -> None:
        self.values_seen += 1
        self.unique_values.add(self._stable_value(value))
        self.in_graph_key = self.in_graph_key or in_graph_key
        if semantic is not None:
            self.semantic = semantic
        if len(self.examples) < max_examples:
            self.examples.append(value)

    @staticmethod
    def _stable_value(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return repr(tuple(value))
        if isinstance(value, dict):
            return repr(sorted(value.items()))
        return repr(value)

    def to_json(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "values_seen": self.values_seen,
            "unique_count": len(self.unique_values),
            "in_graph_key": self.in_graph_key,
            "semantic": self.semantic,
            "examples": self.examples,
        }


class DynamicityAnalyzer:
    """Classifies dynamic fields into recoverable and non-recoverable staticity."""

    FIELD_CATALOG: dict[str, tuple[DynamicityKind, bool, bool, str, str, str]] = {
        "num_tokens": (
            DynamicityKind.SHAPE,
            True,
            True,
            "bucket/pad only if measured graph replay wins",
            "padding FLOPs and large compute-bound graphs",
            "P1",
        ),
        "num_reqs": (
            DynamicityKind.SCHEDULER,
            False,
            True,
            "collapse out of key after metadata arena",
            "wrong if attention metadata is not canonicalized",
            "P0",
        ),
        "cu_seqlens": (
            DynamicityKind.METADATA,
            False,
            True,
            "persistent device buffer + copy-in",
            "stale metadata corrupts attention",
            "P0",
        ),
        "query_start_loc": (
            DynamicityKind.METADATA,
            False,
            True,
            "persistent device buffer + copy-in",
            "stale metadata corrupts attention",
            "P0",
        ),
        "positions": (
            DynamicityKind.METADATA,
            False,
            True,
            "fixed position arena + copy-in",
            "position mismatch changes logits",
            "P0",
        ),
        "slot_mapping": (
            DynamicityKind.ADDRESS,
            False,
            True,
            "fixed slot-mapping arena + copy-in",
            "wrong KV slot writes",
            "P0",
        ),
        "block_table": (
            DynamicityKind.ADDRESS,
            False,
            True,
            "static block-table arena with cleared padded rows",
            "stale block ids corrupt KV state",
            "P0",
        ),
        "prefix_cache": (
            DynamicityKind.METADATA,
            True,
            True,
            "template by prefix-cache mode and arena for block ids",
            "semantic cache hits cannot be faked",
            "P1",
        ),
        "expert_ids": (
            DynamicityKind.CONTROL,
            True,
            True,
            "dynamic values in fixed expert-id buffer",
            "must not freeze routing decisions",
            "P0",
        ),
        "expert_counts": (
            DynamicityKind.SHAPE,
            True,
            True,
            "capacity buckets + fallback for rare imbalance",
            "padding/imbalance can dominate",
            "P0",
        ),
        "expert_offsets": (
            DynamicityKind.METADATA,
            False,
            True,
            "fixed offset/permutation arena + copy-in",
            "wrong token permutation",
            "P0",
        ),
        "mask_positions": (
            DynamicityKind.CONTROL,
            True,
            True,
            "dynamic values in fixed mask/update arena",
            "diffusion semantics change if frozen",
            "P0",
        ),
        "update_indices": (
            DynamicityKind.CONTROL,
            True,
            True,
            "dynamic values in fixed update arena",
            "diffusion semantics change if frozen",
            "P0",
        ),
        "kv_cache": (
            DynamicityKind.ADDRESS,
            True,
            True,
            "persistent KV arena + decoded/KV validation",
            "state aliasing creates silent token drift",
            "P0",
        ),
        "feature_flags": (
            DynamicityKind.FEATURE,
            True,
            True,
            "fast/slow path split",
            "feature cross-product graph explosion",
            "P1",
        ),
    }

    def classify(self, profile: DynamicFieldProfile) -> StaticityDecision:
        key = profile.field
        catalog = self.FIELD_CATALOG.get(key)
        if catalog is None:
            kind = DynamicityKind.METADATA if not profile.semantic else DynamicityKind.CONTROL
            semantic = bool(profile.semantic)
            canonicalizable = not semantic
            method = "profile further; only remove from key after equivalence proof"
            risk = "unknown field semantics"
            priority = "P2"
        else:
            kind, default_semantic, canonicalizable, method, risk, priority = catalog
            semantic = profile.semantic if profile.semantic is not None else default_semantic
            canonicalizable = canonicalizable and (not semantic or kind in {
                DynamicityKind.SHAPE,
                DynamicityKind.CONTROL,
                DynamicityKind.FEATURE,
                DynamicityKind.ADDRESS,
            })
        unique_count = len(profile.unique_values)
        if unique_count <= 1:
            reason = "observed static in this trace"
        elif not profile.in_graph_key:
            reason = "dynamic but not currently known to be in graph key"
        elif canonicalizable:
            reason = "dynamic field is recoverable if representation/address is canonicalized"
        else:
            reason = "semantic dynamicity must stay guarded or fallback"
        return StaticityDecision(
            field=profile.field,
            kind=kind,
            semantic=semantic,
            in_graph_key=profile.in_graph_key,
            canonicalizable=canonicalizable,
            method=method,
            risk=risk,
            priority=priority,
            reason=reason,
        )

    def analyze(self, profiles: list[DynamicFieldProfile]) -> list[StaticityDecision]:
        return sorted(
            (self.classify(profile) for profile in profiles),
            key=lambda decision: (decision.priority, not decision.in_graph_key, decision.field),
        )

    @staticmethod
    def decisions_to_json(decisions: list[StaticityDecision]) -> list[dict[str, Any]]:
        return [
            {
                "field": d.field,
                "kind": d.kind.value,
                "semantic": d.semantic,
                "in_graph_key": d.in_graph_key,
                "canonicalizable": d.canonicalizable,
                "method": d.method,
                "risk": d.risk,
                "priority": d.priority,
                "reason": d.reason,
            }
            for d in decisions
        ]
