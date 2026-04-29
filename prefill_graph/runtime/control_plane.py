from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any


class DynamicAxis(str, Enum):
    TOKEN = "token"
    REQUEST = "request"
    BLOCK = "block"
    EXPERT = "expert"
    STEP = "step"
    LAYER = "layer"
    RANK = "rank"
    FUNCTION = "function"
    RESOURCE = "resource"
    RNG = "rng"


class FieldRole(str, Enum):
    STATIC = "static"
    DYNAMIC_VALUE = "dynamic_value"
    SEMANTIC = "semantic"
    ADDRESS_STABLE = "address_stable"
    SHAPE_GUARDED = "shape_guarded"
    RESOURCE = "resource"


class TemplateLifecycle(str, Enum):
    CANDIDATE = "candidate"
    CAPTURING = "capturing"
    SHADOW_VALIDATING = "shadow_validating"
    ADMITTED = "admitted"
    PROBATION = "probation"
    BLACKLISTED = "blacklisted"
    EVICTED = "evicted"
    FALLBACK_ONLY = "fallback_only"


@dataclass(frozen=True)
class FieldContract:
    name: str
    axis: DynamicAxis
    role: FieldRole
    dtype: str | None = None
    max_shape: tuple[int, ...] | None = None
    padding_value: Any = None
    guard: str | None = None
    validator: str | None = None
    notes: str = ""

    @property
    def semantic(self) -> bool:
        return self.role == FieldRole.SEMANTIC

    @property
    def requires_guard(self) -> bool:
        return self.role in {
            FieldRole.SEMANTIC,
            FieldRole.SHAPE_GUARDED,
            FieldRole.RESOURCE,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "axis": self.axis.value,
            "role": self.role.value,
            "dtype": self.dtype,
            "max_shape": list(self.max_shape) if self.max_shape is not None else None,
            "padding_value": self.padding_value,
            "guard": self.guard,
            "validator": self.validator,
            "notes": self.notes,
            "semantic": self.semantic,
            "requires_guard": self.requires_guard,
        }


@dataclass(frozen=True)
class GuardSpec:
    name: str
    field: str
    op: str
    value: Any
    reason: str

    def evaluate(self, context: dict[str, Any]) -> tuple[bool, str | None]:
        actual = context.get(self.field)
        try:
            if self.op == "eq":
                ok = actual == self.value
            elif self.op == "neq":
                ok = actual != self.value
            elif self.op == "lte":
                ok = actual is not None and actual <= self.value
            elif self.op == "lt":
                ok = actual is not None and actual < self.value
            elif self.op == "gte":
                ok = actual is not None and actual >= self.value
            elif self.op == "gt":
                ok = actual is not None and actual > self.value
            elif self.op == "in":
                ok = actual in self.value
            elif self.op == "range_open_closed":
                lo, hi = self.value
                ok = actual is not None and lo < actual <= hi
            elif self.op == "present":
                ok = self.field in context and actual is not None
            else:
                return False, f"unsupported_guard_op:{self.op}"
        except TypeError:
            ok = False
        if ok:
            return True, None
        return False, self.reason or f"guard_failed:{self.name}"

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "field": self.field,
            "op": self.op,
            "value": self.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ValidatorSpec:
    name: str
    mode: str
    cadence: str = "before_admit"
    atol: float = 0.0
    rtol: float = 0.0
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "cadence": self.cadence,
            "atol": self.atol,
            "rtol": self.rtol,
            "notes": self.notes,
        }


@dataclass
class TemplateManifest:
    template_id: str
    axes: set[DynamicAxis]
    fields: list[FieldContract]
    guards: list[GuardSpec] = field(default_factory=list)
    validators: list[ValidatorSpec] = field(default_factory=list)
    fallback_action: str = "fallback"
    max_templates: int | None = None
    max_memory_bytes: int | None = None
    notes: str = ""

    def guard(self, context: dict[str, Any]) -> tuple[bool, str | None]:
        for guard in self.guards:
            ok, reason = guard.evaluate(context)
            if not ok:
                return False, reason
        return True, None

    @property
    def semantic_fields(self) -> list[str]:
        return [field.name for field in self.fields if field.semantic]

    @property
    def requires_validation(self) -> bool:
        return bool(self.validators or self.semantic_fields)

    def to_json(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "axes": sorted(axis.value for axis in self.axes),
            "fields": [field.to_json() for field in self.fields],
            "guards": [guard.to_json() for guard in self.guards],
            "validators": [validator.to_json() for validator in self.validators],
            "fallback_action": self.fallback_action,
            "max_templates": self.max_templates,
            "max_memory_bytes": self.max_memory_bytes,
            "semantic_fields": self.semantic_fields,
            "requires_validation": self.requires_validation,
            "notes": self.notes,
        }


@dataclass
class TemplateRuntimeRecord:
    manifest: TemplateManifest
    lifecycle: TemplateLifecycle = TemplateLifecycle.CANDIDATE
    attempts: int = 0
    guard_failures: int = 0
    validation_passes: int = 0
    validation_failures: int = 0
    useful_samples: int = 0
    negative_samples: int = 0
    last_reason: str | None = None
    last_updated_s: float = field(default_factory=time.time)

    def transition(self, lifecycle: TemplateLifecycle, reason: str | None = None) -> None:
        self.lifecycle = lifecycle
        self.last_reason = reason
        self.last_updated_s = time.time()

    def to_json(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.to_json(),
            "lifecycle": self.lifecycle.value,
            "attempts": self.attempts,
            "guard_failures": self.guard_failures,
            "validation_passes": self.validation_passes,
            "validation_failures": self.validation_failures,
            "useful_samples": self.useful_samples,
            "negative_samples": self.negative_samples,
            "last_reason": self.last_reason,
            "last_updated_s": self.last_updated_s,
        }


@dataclass(frozen=True)
class DispatchDecision:
    template_id: str | None
    action: str
    lifecycle: str
    admitted: bool
    reason: str
    fallback_action: str
    requires_validation: bool = False


class StaticityControlPlane:
    """Safety control plane for arbitrary serving dynamicity.

    The control plane is deliberately conservative: a template must have an
    explicit manifest, pass all guards, and reach ADMITTED lifecycle before it
    can return a graph action.  Unknown, semantic, or resource-dynamic cases
    become bounded fallback instead of implicit graph execution.
    """

    def __init__(self, *, default_fallback: str = "fallback"):
        self.default_fallback = default_fallback
        self.records: dict[str, TemplateRuntimeRecord] = {}
        self.decision_counts: dict[str, int] = {}

    def register(self, manifest: TemplateManifest) -> TemplateRuntimeRecord:
        record = self.records.get(manifest.template_id)
        if record is None:
            record = TemplateRuntimeRecord(manifest=manifest)
            self.records[manifest.template_id] = record
        else:
            record.manifest = manifest
        return record

    def set_lifecycle(
        self,
        template_id: str,
        lifecycle: TemplateLifecycle,
        *,
        reason: str | None = None,
    ) -> None:
        record = self.records[template_id]
        record.transition(lifecycle, reason)

    def record_validation(self, template_id: str, passed: bool, *, useful: bool | None = None) -> None:
        record = self.records[template_id]
        if passed:
            record.validation_passes += 1
        else:
            record.validation_failures += 1
            record.transition(TemplateLifecycle.BLACKLISTED, "validation_failed")
        if useful is True:
            record.useful_samples += 1
        elif useful is False:
            record.negative_samples += 1
        record.last_updated_s = time.time()

    def decide(self, template_id: str, context: dict[str, Any]) -> DispatchDecision:
        record = self.records.get(template_id)
        if record is None:
            self._count("unknown_template")
            return DispatchDecision(
                template_id=None,
                action=self.default_fallback,
                lifecycle=TemplateLifecycle.FALLBACK_ONLY.value,
                admitted=False,
                reason="unknown_template",
                fallback_action=self.default_fallback,
            )
        record.attempts += 1
        ok, reason = record.manifest.guard(context)
        if not ok:
            record.guard_failures += 1
            record.last_reason = reason
            self._count("guard_reject")
            return DispatchDecision(
                template_id=template_id,
                action=record.manifest.fallback_action,
                lifecycle=record.lifecycle.value,
                admitted=False,
                reason=reason or "guard_failed",
                fallback_action=record.manifest.fallback_action,
                requires_validation=record.manifest.requires_validation,
            )
        if record.lifecycle != TemplateLifecycle.ADMITTED:
            self._count(record.lifecycle.value)
            return DispatchDecision(
                template_id=template_id,
                action=record.manifest.fallback_action,
                lifecycle=record.lifecycle.value,
                admitted=False,
                reason=f"template_{record.lifecycle.value}",
                fallback_action=record.manifest.fallback_action,
                requires_validation=record.manifest.requires_validation,
            )
        self._count("graph")
        return DispatchDecision(
            template_id=template_id,
            action="graph",
            lifecycle=record.lifecycle.value,
            admitted=True,
            reason="admitted",
            fallback_action=record.manifest.fallback_action,
            requires_validation=record.manifest.requires_validation,
        )

    def _count(self, key: str) -> None:
        self.decision_counts[key] = self.decision_counts.get(key, 0) + 1

    def summary(self) -> dict[str, Any]:
        return {
            "default_fallback": self.default_fallback,
            "num_templates": len(self.records),
            "decision_counts": dict(sorted(self.decision_counts.items())),
            "templates": [record.to_json() for record in self.records.values()],
        }


def token_prefill_manifest(
    template_id: str,
    *,
    template_tokens: int,
    max_reqs: int,
    lo: int = 0,
    hi: int | None = None,
    fallback_action: str = "compiled",
) -> TemplateManifest:
    upper = hi if hi is not None else template_tokens
    return TemplateManifest(
        template_id=template_id,
        axes={DynamicAxis.TOKEN, DynamicAxis.REQUEST, DynamicAxis.BLOCK},
        fields=[
            FieldContract("num_tokens", DynamicAxis.TOKEN, FieldRole.SHAPE_GUARDED, guard="token_range"),
            FieldContract("num_reqs", DynamicAxis.REQUEST, FieldRole.SHAPE_GUARDED, guard="max_reqs"),
            FieldContract("positions", DynamicAxis.TOKEN, FieldRole.DYNAMIC_VALUE, "int64", (template_tokens,)),
            FieldContract("slot_mapping", DynamicAxis.TOKEN, FieldRole.DYNAMIC_VALUE, "int64", (template_tokens,)),
            FieldContract("query_start_loc", DynamicAxis.REQUEST, FieldRole.DYNAMIC_VALUE, "int32", (max_reqs + 1,)),
            FieldContract("seq_lens", DynamicAxis.REQUEST, FieldRole.DYNAMIC_VALUE, "int32", (max_reqs,)),
            FieldContract("block_table", DynamicAxis.BLOCK, FieldRole.ADDRESS_STABLE, "int32", None),
        ],
        guards=[
            GuardSpec("token_range", "num_tokens", "range_open_closed", (lo, upper), "tokens_outside_template_range"),
            GuardSpec("max_reqs", "num_reqs", "lte", max_reqs, "num_reqs_exceeds_template"),
        ],
        validators=[
            ValidatorSpec("token_identity", "token_ids", cadence="before_admit_then_periodic"),
        ],
        fallback_action=fallback_action,
        notes="Dense/mixed prefill metadata virtualization manifest.",
    )


def moe_dispatch_manifest(
    template_id: str,
    *,
    capacity_bucket: int,
    max_experts: int,
    top_k: int,
    fallback_action: str = "compiled",
) -> TemplateManifest:
    return TemplateManifest(
        template_id=template_id,
        axes={DynamicAxis.EXPERT, DynamicAxis.TOKEN, DynamicAxis.RANK},
        fields=[
            FieldContract("expert_ids", DynamicAxis.EXPERT, FieldRole.SEMANTIC, "int32", None, validator="token_identity"),
            FieldContract("expert_counts", DynamicAxis.EXPERT, FieldRole.SEMANTIC, "int32", (max_experts,), guard="capacity_bucket"),
            FieldContract("expert_offsets", DynamicAxis.EXPERT, FieldRole.DYNAMIC_VALUE, "int32", (max_experts + 1,)),
            FieldContract("token_permutation", DynamicAxis.TOKEN, FieldRole.DYNAMIC_VALUE, "int32", None),
            FieldContract("all_to_all_splits", DynamicAxis.RANK, FieldRole.SHAPE_GUARDED, "int32", None),
        ],
        guards=[
            GuardSpec("capacity_bucket", "max_expert_count", "lte", capacity_bucket, "expert_capacity_overflow"),
            GuardSpec("top_k", "top_k", "eq", top_k, "top_k_mismatch"),
        ],
        validators=[
            ValidatorSpec("token_identity", "token_ids", cadence="before_admit_then_periodic"),
            ValidatorSpec("traffic_template", "metadata_invariants", cadence="every_replay"),
        ],
        fallback_action=fallback_action,
        notes="MoE dispatch-level template manifest; expert values remain semantic dynamic.",
    )


def function_branch_manifest(
    template_id: str,
    *,
    function_name: str,
    branch_field: str,
    branch_value: Any,
    max_tokens: int,
    fallback_action: str = "eager",
) -> TemplateManifest:
    return TemplateManifest(
        template_id=template_id,
        axes={DynamicAxis.FUNCTION, DynamicAxis.TOKEN, DynamicAxis.STEP},
        fields=[
            FieldContract(branch_field, DynamicAxis.FUNCTION, FieldRole.SEMANTIC, guard="branch_guard"),
            FieldContract("num_tokens", DynamicAxis.TOKEN, FieldRole.SHAPE_GUARDED, guard="max_tokens"),
            FieldContract("step", DynamicAxis.STEP, FieldRole.SEMANTIC, guard="step_guard"),
        ],
        guards=[
            GuardSpec("branch_guard", branch_field, "eq", branch_value, "semantic_branch_mismatch"),
            GuardSpec("max_tokens", "num_tokens", "lte", max_tokens, "tokens_exceed_function_template"),
        ],
        validators=[
            ValidatorSpec("segment_output", "segment_checksum", cadence="before_admit_then_periodic"),
        ],
        fallback_action=fallback_action,
        notes=f"Function-level partial graph manifest for {function_name}.",
    )
