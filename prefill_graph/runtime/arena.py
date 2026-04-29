from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArenaFieldSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str = "cuda"
    semantic_value: bool = False


class CanonicalMetadataArena:
    """Fixed-address metadata arena.

    The arena keeps addresses stable and lets callers copy dynamic values into
    those addresses before graph replay.  Torch is imported lazily so policy
    tooling can run on CPU-only Python.
    """

    def __init__(self, specs: list[ArenaFieldSpec]):
        self.specs = {spec.name: spec for spec in specs}
        self.buffers: dict[str, Any] = {}

    def allocate(self) -> None:
        import torch

        for spec in self.specs.values():
            dtype = getattr(torch, spec.dtype) if isinstance(spec.dtype, str) else spec.dtype
            self.buffers[spec.name] = torch.zeros(spec.shape, dtype=dtype, device=spec.device)

    def copy_in(self, name: str, value: Any, *, zero_fill: bool = True) -> None:
        if name not in self.buffers:
            raise KeyError(f"arena field {name!r} is not allocated")
        import torch

        target = self.buffers[name]
        source = value.to(target.device, dtype=target.dtype) if torch.is_tensor(value) else torch.as_tensor(value, dtype=target.dtype, device=target.device)
        if source.numel() > target.numel():
            raise ValueError(f"source for {name!r} has {source.numel()} elements, arena only has {target.numel()}")
        flat_target = target.reshape(-1)
        flat_source = source.reshape(-1)
        if zero_fill:
            flat_target.zero_()
        flat_target[: flat_source.numel()].copy_(flat_source)

    def descriptors(self) -> dict[str, dict[str, Any]]:
        out = {}
        for name, spec in self.specs.items():
            buffer = self.buffers.get(name)
            out[name] = {
                "shape": list(spec.shape),
                "dtype": spec.dtype,
                "device": spec.device,
                "semantic_value": spec.semantic_value,
                "allocated": buffer is not None,
                "data_ptr": int(buffer.data_ptr()) if buffer is not None else None,
            }
        return out

    @staticmethod
    def vllm_prefill_specs(max_tokens: int, max_reqs: int, *, device: str = "cuda") -> list[ArenaFieldSpec]:
        return [
            ArenaFieldSpec("query_start_loc", (max_reqs + 1,), "int32", device),
            ArenaFieldSpec("seq_lens", (max_reqs,), "int32", device),
            ArenaFieldSpec("positions", (max_tokens,), "int64", device),
            ArenaFieldSpec("slot_mapping", (max_tokens,), "int64", device),
            ArenaFieldSpec("request_indices", (max_reqs,), "int32", device),
        ]

    @staticmethod
    def vllm_token_axis_specs(
        max_tokens: int,
        *,
        device: str = "cuda",
        include_mask: bool = True,
    ) -> list[ArenaFieldSpec]:
        specs = [
            ArenaFieldSpec("positions", (max_tokens,), "int64", device),
            ArenaFieldSpec("slot_mapping", (max_tokens,), "int64", device),
            ArenaFieldSpec("token_active_mask", (max_tokens,), "bool", device),
        ]
        if not include_mask:
            specs = specs[:2]
        return specs

    @staticmethod
    def vllm_request_axis_specs(
        max_reqs: int,
        *,
        max_blocks_per_req: int = 0,
        device: str = "cuda",
    ) -> list[ArenaFieldSpec]:
        specs = [
            ArenaFieldSpec("query_start_loc", (max_reqs + 1,), "int32", device),
            ArenaFieldSpec("seq_lens", (max_reqs,), "int32", device),
            ArenaFieldSpec("request_indices", (max_reqs,), "int32", device),
            ArenaFieldSpec("is_prefilling", (max_reqs,), "bool", device),
        ]
        if max_blocks_per_req:
            specs.append(
                ArenaFieldSpec(
                    "block_table",
                    (max_reqs, max_blocks_per_req),
                    "int32",
                    device,
                )
            )
        return specs

    @staticmethod
    def moe_expert_specs(
        max_tokens: int,
        max_experts: int,
        *,
        top_k: int = 1,
        device: str = "cuda",
    ) -> list[ArenaFieldSpec]:
        return [
            ArenaFieldSpec(
                "expert_ids",
                (max_tokens, top_k),
                "int32",
                device,
                semantic_value=True,
            ),
            ArenaFieldSpec(
                "expert_counts",
                (max_experts,),
                "int32",
                device,
                semantic_value=True,
            ),
            ArenaFieldSpec(
                "expert_offsets",
                (max_experts + 1,),
                "int32",
                device,
            ),
            ArenaFieldSpec(
                "token_permutation",
                (max_tokens, top_k),
                "int32",
                device,
            ),
            ArenaFieldSpec(
                "expert_active_mask",
                (max_experts,),
                "bool",
                device,
            ),
        ]

    @staticmethod
    def dinfer_diffusion_specs(max_tokens: int, max_experts: int = 0, *, device: str = "cuda") -> list[ArenaFieldSpec]:
        specs = [
            ArenaFieldSpec("mask_positions", (max_tokens,), "int64", device, semantic_value=True),
            ArenaFieldSpec("update_indices", (max_tokens,), "int64", device, semantic_value=True),
            ArenaFieldSpec("confidence", (max_tokens,), "float32", device, semantic_value=True),
            ArenaFieldSpec("replace_position", (2,), "int64", device, semantic_value=False),
        ]
        if max_experts:
            specs.extend(
                [
                    ArenaFieldSpec("expert_ids", (max_tokens,), "int32", device, semantic_value=True),
                    ArenaFieldSpec("expert_offsets", (max_experts + 1,), "int32", device),
                    ArenaFieldSpec("expert_counts", (max_experts,), "int32", device, semantic_value=True),
                ]
            )
        return specs


@dataclass(frozen=True)
class TokenAxisTemplate:
    template_tokens: int
    max_reqs: int
    action: str = "ours_cp"
    min_tokens: int = 0
    max_tokens: int | None = None
    padded_mask_value: int = -1

    def accepts(self, tokens: int) -> bool:
        if tokens <= self.min_tokens:
            return False
        if self.max_tokens is not None and tokens > self.max_tokens:
            return False
        return tokens <= self.template_tokens

    def rule(self) -> dict[str, Any]:
        return {
            "lo": self.min_tokens,
            "hi": self.max_tokens or self.template_tokens,
            "action": self.action,
            "template_tokens": self.template_tokens,
            "fixed_token_axis": True,
            "fixed_request_axis": True,
        }


@dataclass
class TokenAxisCanonicalizer:
    """CPU-side shape canonicalizer for token-axis metadata.

    Framework adapters use this to determine template shape and padded metadata
    before copying values into fixed device buffers.  It deliberately does not
    hide semantic values: padded slots are masked and inactive.
    """

    template: TokenAxisTemplate

    def canonicalize(
        self,
        *,
        positions: list[int],
        slot_mapping: list[int],
    ) -> dict[str, Any]:
        tokens = len(positions)
        if len(slot_mapping) != tokens:
            raise ValueError("positions and slot_mapping must have the same length")
        if not self.template.accepts(tokens):
            raise ValueError(
                f"{tokens} tokens do not fit template {self.template.template_tokens}"
            )
        pad = self.template.template_tokens - tokens
        return {
            "template_tokens": self.template.template_tokens,
            "original_tokens": tokens,
            "positions": list(positions) + [0] * pad,
            "slot_mapping": list(slot_mapping) + [self.template.padded_mask_value] * pad,
            "token_active_mask": [True] * tokens + [False] * pad,
        }


@dataclass(frozen=True)
class ExpertTrafficTemplate:
    capacity_bucket: int
    max_experts: int
    top_k: int = 1
    action: str = "moe_graph"
    rare_imbalance_fallback: str = "fallback"

    def accepts(self, expert_counts: list[int]) -> bool:
        return bool(expert_counts) and max(expert_counts) <= self.capacity_bucket

    def template_id(self, expert_counts: list[int]) -> str:
        active = sum(1 for count in expert_counts if count > 0)
        return (
            f"moe:capacity={self.capacity_bucket}:active={active}:"
            f"experts={self.max_experts}:topk={self.top_k}"
        )


@dataclass
class ExpertMetadataCanonicalizer:
    template: ExpertTrafficTemplate
    max_tokens: int

    def canonicalize(
        self,
        *,
        expert_ids: list[list[int]] | list[int],
        expert_counts: list[int],
        expert_offsets: list[int] | None = None,
        token_permutation: list[list[int]] | list[int] | None = None,
    ) -> dict[str, Any]:
        if not self.template.accepts(expert_counts):
            return {
                "accepted": False,
                "reason": "expert_capacity_overflow",
                "fallback_action": self.template.rare_imbalance_fallback,
                "template_id": self.template.template_id(expert_counts),
            }
        normalized_ids = self._normalize_2d(expert_ids, self.template.top_k)
        token_count = len(normalized_ids)
        if token_count > self.max_tokens:
            return {
                "accepted": False,
                "reason": "token_capacity_overflow",
                "fallback_action": self.template.rare_imbalance_fallback,
                "template_id": self.template.template_id(expert_counts),
            }
        offsets = expert_offsets
        if offsets is None:
            offsets = [0]
            running = 0
            for count in expert_counts:
                running += int(count)
                offsets.append(running)
        permutation = (
            self._normalize_2d(token_permutation, self.template.top_k)
            if token_permutation is not None
            else [[idx] * self.template.top_k for idx in range(token_count)]
        )
        pad = self.max_tokens - token_count
        return {
            "accepted": True,
            "template_id": self.template.template_id(expert_counts),
            "capacity_bucket": self.template.capacity_bucket,
            "original_tokens": token_count,
            "expert_ids": normalized_ids + [[-1] * self.template.top_k for _ in range(pad)],
            "expert_counts": list(expert_counts)
            + [0] * max(0, self.template.max_experts - len(expert_counts)),
            "expert_offsets": list(offsets)
            + [offsets[-1] if offsets else 0]
            * max(0, self.template.max_experts + 1 - len(offsets)),
            "token_permutation": permutation
            + [[-1] * self.template.top_k for _ in range(pad)],
            "expert_active_mask": [
                count > 0 for count in list(expert_counts)[: self.template.max_experts]
            ]
            + [False] * max(0, self.template.max_experts - len(expert_counts)),
        }

    @staticmethod
    def _normalize_2d(values: list[list[int]] | list[int] | None, width: int) -> list[list[int]]:
        if values is None:
            return []
        if not values:
            return []
        first = values[0]  # type: ignore[index]
        if isinstance(first, list):
            rows = values  # type: ignore[assignment]
            return [
                [int(x) for x in row[:width]] + [-1] * max(0, width - len(row))
                for row in rows  # type: ignore[union-attr]
            ]
        return [[int(x)] + [-1] * max(0, width - 1) for x in values]  # type: ignore[arg-type]


@dataclass
class ArenaTemplateRegistry:
    token_templates: list[TokenAxisTemplate] = field(default_factory=list)
    expert_templates: list[ExpertTrafficTemplate] = field(default_factory=list)

    def token_template_for(self, tokens: int) -> TokenAxisTemplate | None:
        candidates = [template for template in self.token_templates if template.accepts(tokens)]
        if not candidates:
            return None
        return min(candidates, key=lambda template: template.template_tokens)

    def expert_template_for(self, expert_counts: list[int]) -> ExpertTrafficTemplate | None:
        candidates = [
            template for template in self.expert_templates if template.accepts(expert_counts)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda template: template.capacity_bucket)

    def to_policy_ranges(self) -> list[dict[str, Any]]:
        return [template.rule() for template in self.token_templates]
