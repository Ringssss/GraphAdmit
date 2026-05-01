from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prefill_graph.runtime import residual_buckets_for_preset


def make_exploration_policy(
    *,
    bucket_preset: str = "sglang-pcg",
    max_tokens: int = 4096,
    base_capture_size: int = 512,
    min_tokens: int = 0,
    default_action: str = "cp",
    graph_action: str = "ours_cp",
    max_extra_templates: int = 0,
    live_admission: bool = True,
    live_min_samples: int = 2,
    live_min_useful_rate: float = 0.67,
    live_min_saving_ms: float = 0.5,
    live_max_p95_regression_ms: float | None = 5.0,
    live_capture: bool = False,
) -> dict[str, Any]:
    """Build a fail-closed GraphAdmit exploration policy.

    The policy admits no template permanently by construction.  It exposes
    residual capture candidates to the patched serving engine, and the engine's
    live admission path decides whether each template becomes useful enough to
    replay.  Unsupported or unprofitable windows fall back to ``default_action``.
    """

    buckets = residual_buckets_for_preset(bucket_preset, max_tokens=max_tokens)
    rules: list[dict[str, Any]] = []
    arena_ranges: list[dict[str, Any]] = []
    left = 0
    for bucket in buckets:
        bucket = int(bucket)
        if bucket <= int(base_capture_size):
            left = max(left, bucket)
            continue
        hi = min(bucket, int(max_tokens))
        if hi <= left:
            continue
        action = default_action if (min_tokens and hi <= min_tokens) else graph_action
        rule = {
            "lo": int(left),
            "hi": int(hi),
            "action": action,
            "n": 0,
            "template_tokens": bucket,
            "reason": "online exploration candidate; live admission decides replay",
        }
        rules.append(rule)
        if action == graph_action:
            arena_ranges.append(
                {
                    "lo": int(left),
                    "hi": int(hi),
                    "template_tokens": bucket,
                    "action": action,
                    "n": 0,
                }
            )
        left = hi
        if left >= int(max_tokens):
            break
    if left < int(max_tokens):
        rules.append(
            {
                "lo": int(left),
                "hi": int(max_tokens),
                "action": default_action,
                "n": 0,
                "reason": "outside residual exploration capture budget",
            }
        )

    return {
        "runtime_policy": {
            "kind": "graphadmit_online_exploration_policy",
            "default_action": default_action,
            "baseline_action": default_action,
            "correctness_required": True,
            "rules": rules,
            "fixed_metadata_arena_ranges": arena_ranges,
            "single_engine_graph_actions": ["default", "ours", "cp", graph_action],
            "single_engine_fallback_actions": [
                "eager",
                "compile",
                "compiled",
                "fallback",
                "none",
            ],
            "single_engine_allow_multi_req_extra": True,
            "single_engine_requires_fixed_metadata_arena": True,
            "single_engine_max_extra_templates": (
                int(max_extra_templates) if max_extra_templates else max(1, len(arena_ranges))
            ),
            "single_engine_min_rule_n": 0,
            "single_engine_base_capture_size": int(base_capture_size),
            "live_admission": {
                "enabled": bool(live_admission),
                "explore_until_min_samples": True,
                "min_samples": int(live_min_samples),
                "min_useful_rate": float(live_min_useful_rate),
                "min_saving_ms": float(live_min_saving_ms),
                "max_p95_regression_ms": (
                    None
                    if live_max_p95_regression_ms is None
                    else float(live_max_p95_regression_ms)
                ),
            },
            "live_capture": {
                "enabled": bool(live_capture),
                "mode": "same_engine_lazy_piecewise_capture",
                "scope": "piecewise_prefill_templates",
                "fallback": default_action,
                "note": (
                    "FULL-graph templates remain init-time captured; PIECEWISE "
                    "extra templates may be captured on first admitted/explored "
                    "runtime hit and later controlled by live admission."
                ),
            },
            "residual_capture": {
                "mode": (
                    "same_engine_lazy_capture"
                    if live_capture else "broad_online_exploration"
                ),
                "bucket_preset": bucket_preset,
                "template_buckets": buckets,
                "extra_capture_sizes": [
                    int(row["template_tokens"]) for row in arena_ranges
                ],
            },
        }
    }


def write_policy(policy: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_policy(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
