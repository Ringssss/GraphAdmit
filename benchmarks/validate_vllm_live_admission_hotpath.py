#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


def write_policy(path: Path) -> None:
    policy = {
        "runtime_policy": {
            "kind": "hotpath_live_admission_validation",
            "default_action": "cp",
            "baseline_action": "cp",
            "single_engine_graph_actions": ["default", "cp", "ours", "ours_cp"],
            "single_engine_fallback_actions": ["eager", "compile", "compiled", "fallback", "none"],
            "single_engine_base_capture_size": 512,
            "single_engine_max_extra_templates": 4,
            "rules": [
                {
                    "lo": 700,
                    "hi": 830,
                    "action": "ours_cp",
                    "n": 6,
                    "template_tokens": 832,
                },
                {
                    "lo": 830,
                    "hi": 1050,
                    "action": "ours_cp",
                    "n": 6,
                    "template_tokens": 1024,
                },
            ],
            "online_admission": {
                "mode": "online_self_learning_admission",
                "min_samples": 2,
                "min_useful_rate": 0.75,
                "min_saving_ms": 1.0,
                "max_p95_regression_ms": 5.0,
                "max_correctness_failures": 0,
                "templates": [
                    {
                        "template_id": "tokens=832",
                        "samples": 3,
                        "useful": 3,
                        "regressions": 0,
                        "correctness_failures": 0,
                        "saving_ewma_ms": 12.0,
                        "p95_regression_ms": 0.0,
                    }
                ],
            },
        }
    }
    path.write_text(json.dumps(policy, indent=2), encoding="utf-8")


def write_observations(path: Path) -> None:
    rows = [
        {"template_id": "tokens=1024", "graph_ms": 60.0, "fallback_ms": 55.0, "correct": True},
        {"template_id": "tokens=1024", "graph_ms": 61.0, "fallback_ms": 56.0, "correct": True},
    ]
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/vllm_live_admission_hotpath_validation.json")
    parser.add_argument("--workdir", default="results/live_admission_hotpath")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    policy_path = workdir / "policy.json"
    obs_path = workdir / "observations.jsonl"
    write_policy(policy_path)
    write_observations(obs_path)

    os.environ["STATICITY_VLLM_RUNTIME_POLICY"] = str(policy_path)
    os.environ["STATICITY_VLLM_RUNTIME_ACTIVE"] = "1"
    os.environ["STATICITY_VLLM_LIVE_ADMISSION"] = "1"
    os.environ["STATICITY_VLLM_LIVE_OBSERVATIONS"] = str(obs_path)
    os.environ["STATICITY_VLLM_BASE_CAPTURE_SIZE"] = "512"

    dispatcher = importlib.import_module("vllm.v1.cudagraph_dispatcher")
    dispatcher._STATICITY_RUNTIME_POLICY_CACHE = None
    dispatcher._STATICITY_RUNTIME_POLICY_PATH = None

    admitted = dispatcher._staticity_runtime_allows_graph(750, 1, False)
    rejected = dispatcher._staticity_runtime_allows_graph(900, 1, False)
    default_safe = dispatcher._staticity_runtime_allows_graph(128, 1, False)

    payload = {
        "policy": str(policy_path),
        "observations": str(obs_path),
        "admitted_832": {
            "allowed": admitted[0],
            "action": admitted[1],
            "reason": admitted[3],
            "admission": admitted[4],
        },
        "rejected_1024": {
            "allowed": rejected[0],
            "action": rejected[1],
            "reason": rejected[3],
            "admission": rejected[4],
        },
        "default_safe_path": {
            "allowed": default_safe[0],
            "action": default_safe[1],
            "reason": default_safe[3],
            "admission": default_safe[4],
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
