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
                {
                    "lo": 1050,
                    "hi": 1300,
                    "action": "ours_cp",
                    "n": 6,
                    "template_tokens": 1280,
                },
                {
                    "lo": 1300,
                    "hi": 1500,
                    "action": "ours_cp",
                    "n": 6,
                    "template_tokens": 1536,
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
        {"template_id": "tokens=832", "graph_ms": 42.0, "fallback_ms": 55.0, "correct": True, "source": "live_graph_replay"},
        {"template_id": "tokens=832", "graph_ms": 43.0, "fallback_ms": 56.0, "correct": True, "source": "live_graph_replay"},
        {"template_id": "tokens=1024", "graph_ms": 60.0, "fallback_ms": 55.0, "correct": True, "source": "graph_replay_shadow"},
        {"template_id": "tokens=1024", "graph_ms": 61.0, "fallback_ms": 56.0, "correct": True, "source": "graph_replay_shadow"},
        {
            "template_id": "ours_cp:1050:1300:template=1280:reqs=1",
            "template_aliases": ["tokens=1280"],
            "graph_ms": 40.0,
            "fallback_ms": 80.0,
            "correct": False,
            "source": "graph_replay_shadow",
        },
        {
            "template_id": "tokens=1536",
            "graph_ms": 30.0,
            "fallback_ms": 70.0,
            "correct": True,
            "source": "graph_replay_shadow",
        },
        {
            "template_id": "tokens=1536",
            "graph_ms": 31.0,
            "fallback_ms": 71.0,
            "correct": True,
            "source": "graph_replay_shadow",
        },
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
    os.environ["STATICITY_VLLM_LIVE_EXPLORE"] = "1"
    os.environ["STATICITY_VLLM_LIVE_OBSERVATIONS"] = str(obs_path)
    os.environ["STATICITY_VLLM_BASE_CAPTURE_SIZE"] = "512"
    os.environ["STATICITY_VLLM_TRUST_SEEDED_ONLINE_ADMISSION"] = "0"
    os.environ["STATICITY_VLLM_LIVE_CAPTURE"] = "1"
    os.environ.pop("STATICITY_VLLM_LIVE_CAPTURE_ONLY_EXPLORE", None)

    dispatcher = importlib.import_module("vllm.v1.cudagraph_dispatcher")
    dispatcher._STATICITY_RUNTIME_POLICY_CACHE = None
    dispatcher._STATICITY_RUNTIME_POLICY_PATH = None

    admitted = dispatcher._staticity_runtime_allows_graph(750, 1, False)
    rejected = dispatcher._staticity_runtime_allows_graph(900, 1, False)
    blacklisted_range = dispatcher._staticity_runtime_allows_graph(1100, 1, False)
    exploring = dispatcher._staticity_runtime_allows_graph(1400, 1, False)
    shadow_positive_not_admitted = dispatcher._staticity_runtime_allows_graph(1450, 1, False)
    default_safe = dispatcher._staticity_runtime_allows_graph(128, 1, False)
    os.environ["STATICITY_VLLM_LIVE_CAPTURE_ONLY_EXPLORE"] = "1"
    dispatcher._STATICITY_RUNTIME_POLICY_CACHE = None
    dispatcher._STATICITY_RUNTIME_POLICY_PATH = None
    capture_only = dispatcher._staticity_runtime_allows_graph(1400, 1, False)
    empty_obs_path = workdir / "empty_observations.jsonl"
    empty_obs_path.write_text("", encoding="utf-8")
    os.environ["STATICITY_VLLM_LIVE_OBSERVATIONS"] = str(empty_obs_path)
    os.environ.pop("STATICITY_VLLM_LIVE_CAPTURE_ONLY_EXPLORE", None)
    dispatcher._STATICITY_RUNTIME_POLICY_CACHE = None
    dispatcher._STATICITY_RUNTIME_POLICY_PATH = None
    seeded_ignored = dispatcher._staticity_runtime_allows_graph(750, 1, False)

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
        "exploring_1280": {
            "allowed": exploring[0],
            "action": exploring[1],
            "reason": exploring[3],
            "admission": exploring[4],
        },
        "blacklisted_range_1280": {
            "allowed": blacklisted_range[0],
            "action": blacklisted_range[1],
            "reason": blacklisted_range[3],
            "admission": blacklisted_range[4],
        },
        "shadow_positive_not_admitted_1536": {
            "allowed": shadow_positive_not_admitted[0],
            "action": shadow_positive_not_admitted[1],
            "reason": shadow_positive_not_admitted[3],
            "admission": shadow_positive_not_admitted[4],
        },
        "capture_only_debug_1280": {
            "allowed": capture_only[0],
            "action": capture_only[1],
            "reason": capture_only[3],
            "admission": capture_only[4],
        },
        "default_safe_path": {
            "allowed": default_safe[0],
            "action": default_safe[1],
            "reason": default_safe[3],
            "admission": default_safe[4],
        },
        "seeded_policy_ignored_without_live_observations": {
            "allowed": seeded_ignored[0],
            "action": seeded_ignored[1],
            "reason": seeded_ignored[3],
            "admission": seeded_ignored[4],
        },
    }
    assert admitted[0] is True, admitted
    assert rejected[0] is False, rejected
    assert blacklisted_range[0] is False, blacklisted_range
    assert blacklisted_range[3] == "live_correctness_failure", blacklisted_range
    assert exploring[0] is False, exploring
    assert exploring[3] == "live_explore_fallback_until_admitted", exploring
    assert shadow_positive_not_admitted[0] is False, shadow_positive_not_admitted
    assert shadow_positive_not_admitted[3] == "live_explore_fallback_until_admitted", shadow_positive_not_admitted
    assert capture_only[0] is True, capture_only
    assert capture_only[3] == "live_capture_only_until_admitted", capture_only
    assert capture_only[4]["live_capture_only"] is True, capture_only
    assert default_safe[0] is True, default_safe
    assert seeded_ignored[0] is False, seeded_ignored
    assert seeded_ignored[3] == "live_explore_fallback_until_admitted", seeded_ignored
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
