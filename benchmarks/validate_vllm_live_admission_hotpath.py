#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    probe_dispatch_path = workdir / "probe_dispatch.jsonl"
    probe_obs_path = workdir / "probe_observations.jsonl"
    write_policy(policy_path)
    write_observations(obs_path)
    if probe_obs_path.exists():
        probe_obs_path.unlink()
    probe_dispatch_rows = [
        {
            "kind": "dispatch",
            "mode": "NONE",
            "requested_num_tokens": 896,
            "reason": "live_explore_fallback_until_admitted",
            "staticity_runtime_action": "ours_cp",
            "staticity_runtime_reason": "live_explore_fallback_until_admitted",
            "staticity_runtime_rule": {
                "lo": 832,
                "hi": 896,
                "action": "ours_cp",
                "template_tokens": 896,
            },
            "staticity_runtime_admission": {
                "template_id": "ours_cp:832:896:template=896:reqs=1",
                "template_tokens": 896,
            },
        },
        {
            "kind": "dispatch",
            "mode": "PIECEWISE",
            "requested_num_tokens": 960,
            "reason": "live_explore_replay_until_admitted",
            "staticity_runtime_action": "ours_cp",
            "staticity_runtime_reason": "live_explore_replay_until_admitted",
            "staticity_runtime_rule": {
                "lo": 896,
                "hi": 960,
                "action": "ours_cp",
                "template_tokens": 960,
            },
            "staticity_runtime_admission": {
                "template_id": "ours_cp:896:960:template=960:reqs=1",
                "template_tokens": 960,
            },
        },
        {
            "kind": "dispatch",
            "mode": "PIECEWISE",
            "requested_num_tokens": 1024,
            "reason": "live_explore_replay_until_admitted",
            "staticity_runtime_action": "ours_cp",
            "staticity_runtime_reason": "live_explore_replay_until_admitted",
            "staticity_runtime_rule": {
                "lo": 960,
                "hi": 1024,
                "action": "ours_cp",
                "template_tokens": 1024,
            },
            "staticity_runtime_admission": {
                "template_id": "ours_cp:960:1024:template=1024:reqs=1",
                "template_tokens": 1024,
            },
        },
    ]
    probe_dispatch_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in probe_dispatch_rows),
        encoding="utf-8",
    )

    from benchmarks.vllm_flowprefill_workload import (
        _append_probe_dispatch_crash_blacklist,
    )

    probe_blacklisted = _append_probe_dispatch_crash_blacklist(
        str(probe_obs_path),
        dispatch_profile=str(probe_dispatch_path),
        returncode=1,
    )
    probe_rows = [
        json.loads(line)
        for line in probe_obs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    os.environ["STATICITY_VLLM_RUNTIME_POLICY"] = str(policy_path)
    os.environ["STATICITY_VLLM_RUNTIME_ACTIVE"] = "1"
    os.environ["STATICITY_VLLM_LIVE_ADMISSION"] = "1"
    os.environ["STATICITY_VLLM_LIVE_EXPLORE"] = "1"
    os.environ["STATICITY_VLLM_LIVE_OBSERVATIONS"] = str(obs_path)
    os.environ["STATICITY_VLLM_BASE_CAPTURE_SIZE"] = "512"
    os.environ["STATICITY_VLLM_TRUST_SEEDED_ONLINE_ADMISSION"] = "0"
    os.environ["STATICITY_VLLM_LIVE_CAPTURE"] = "1"
    os.environ.pop("STATICITY_VLLM_LIVE_CAPTURE_ONLY_EXPLORE", None)
    control_path = workdir / "runtime_control.json"
    control_path.write_text("{}", encoding="utf-8")
    os.environ["STATICITY_VLLM_RUNTIME_CONTROL"] = str(control_path)

    dispatcher = importlib.import_module("vllm.v1.cudagraph_dispatcher")
    dispatcher._STATICITY_RUNTIME_POLICY_CACHE = None
    dispatcher._STATICITY_RUNTIME_POLICY_PATH = None

    admitted = dispatcher._staticity_runtime_allows_graph(750, 1, False)
    rejected = dispatcher._staticity_runtime_allows_graph(900, 1, False)
    blacklisted_range = dispatcher._staticity_runtime_allows_graph(1100, 1, False)
    exploring = dispatcher._staticity_runtime_allows_graph(1400, 1, False)
    shadow_positive_not_admitted = dispatcher._staticity_runtime_allows_graph(1450, 1, False)
    default_safe = dispatcher._staticity_runtime_allows_graph(128, 1, False)
    old_graphadmit_only = os.environ.get("STATICITY_VLLM_GRAPHADMIT_ONLY")
    os.environ["STATICITY_VLLM_GRAPHADMIT_ONLY"] = "1"
    try:
        dispatcher._STATICITY_RUNTIME_POLICY_CACHE = None
        dispatcher._STATICITY_RUNTIME_POLICY_PATH = None
        graphadmit_only_default = dispatcher._staticity_runtime_allows_graph(128, 1, False)
    finally:
        if old_graphadmit_only is None:
            os.environ.pop("STATICITY_VLLM_GRAPHADMIT_ONLY", None)
        else:
            os.environ["STATICITY_VLLM_GRAPHADMIT_ONLY"] = old_graphadmit_only
        dispatcher._STATICITY_RUNTIME_POLICY_CACHE = None
        dispatcher._STATICITY_RUNTIME_POLICY_PATH = None
    os.environ["STATICITY_VLLM_FORCE_RUNTIME_FALLBACK"] = "1"
    forced_runtime_fallback = dispatcher._staticity_runtime_allows_graph(750, 1, False)
    os.environ.pop("STATICITY_VLLM_FORCE_RUNTIME_FALLBACK", None)
    control_path.write_text(
        json.dumps({"force_runtime_fallback": True}),
        encoding="utf-8",
    )
    control_forced_runtime_fallback = dispatcher._staticity_runtime_allows_graph(750, 1, False)
    control_path.write_text("{}", encoding="utf-8")
    os.environ["STATICITY_VLLM_LIVE_CAPTURE_ONLY_EXPLORE"] = "1"
    dispatcher._STATICITY_RUNTIME_POLICY_CACHE = None
    dispatcher._STATICITY_RUNTIME_POLICY_PATH = None
    capture_only = dispatcher._staticity_runtime_allows_graph(1400, 1, False)
    os.environ.pop("STATICITY_VLLM_LIVE_CAPTURE_ONLY_EXPLORE", None)
    control_path.write_text(
        json.dumps({"unsafe_live_explore_replay": True}),
        encoding="utf-8",
    )
    dispatcher._STATICITY_RUNTIME_POLICY_CACHE = None
    dispatcher._STATICITY_RUNTIME_POLICY_PATH = None
    exploratory_replay = dispatcher._staticity_runtime_allows_graph(1400, 1, False)
    control_path.write_text("{}", encoding="utf-8")
    empty_obs_path = workdir / "empty_observations.jsonl"
    empty_obs_path.write_text("", encoding="utf-8")
    os.environ["STATICITY_VLLM_LIVE_OBSERVATIONS"] = str(empty_obs_path)
    dispatcher._STATICITY_RUNTIME_POLICY_CACHE = None
    dispatcher._STATICITY_RUNTIME_POLICY_PATH = None
    seeded_ignored = dispatcher._staticity_runtime_allows_graph(750, 1, False)

    from vllm.config import CUDAGraphMode
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    class FakeCompilationConfig:
        cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
        max_cudagraph_capture_size = 1024
        cudagraph_capture_sizes = [128, 512, 832, 1024]
        compile_sizes = None
        cudagraph_specialize_lora = True

        @staticmethod
        def is_attention_compiled_piecewise() -> bool:
            return True

    fake_config = SimpleNamespace(
        compilation_config=FakeCompilationConfig(),
        speculative_config=None,
        lora_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    )
    os.environ["STATICITY_VLLM_LIVE_OBSERVATIONS"] = str(obs_path)
    os.environ["STATICITY_VLLM_GRAPHADMIT_ONLY"] = "1"
    dispatcher._STATICITY_RUNTIME_POLICY_CACHE = None
    dispatcher._STATICITY_RUNTIME_POLICY_PATH = None
    strict_dispatcher = CudagraphDispatcher(fake_config)
    strict_dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL_AND_PIECEWISE)
    strict_default_dispatch = strict_dispatcher.dispatch(128, 1, False)
    strict_admitted_dispatch = strict_dispatcher.dispatch(750, 1, False)
    os.environ.pop("STATICITY_VLLM_GRAPHADMIT_ONLY", None)
    dispatcher._STATICITY_RUNTIME_POLICY_CACHE = None
    dispatcher._STATICITY_RUNTIME_POLICY_PATH = None

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
        "exploratory_replay_not_admitted": {
            "allowed": exploratory_replay[0],
            "action": exploratory_replay[1],
            "reason": exploratory_replay[3],
            "admission": exploratory_replay[4],
        },
        "default_safe_path": {
            "allowed": default_safe[0],
            "action": default_safe[1],
            "reason": default_safe[3],
            "admission": default_safe[4],
        },
        "graphadmit_only_default_rejected": {
            "allowed": graphadmit_only_default[0],
            "action": graphadmit_only_default[1],
            "reason": graphadmit_only_default[3],
            "admission": graphadmit_only_default[4],
        },
        "forced_runtime_fallback": {
            "allowed": forced_runtime_fallback[0],
            "action": forced_runtime_fallback[1],
            "reason": forced_runtime_fallback[3],
            "admission": forced_runtime_fallback[4],
        },
        "control_forced_runtime_fallback": {
            "allowed": control_forced_runtime_fallback[0],
            "action": control_forced_runtime_fallback[1],
            "reason": control_forced_runtime_fallback[3],
            "admission": control_forced_runtime_fallback[4],
        },
        "probe_dispatch_crash_blacklist": {
            "written": probe_blacklisted,
            "templates": [row["template_id"] for row in probe_rows],
            "modes": [row["validation_mode"] for row in probe_rows],
        },
        "seeded_policy_ignored_without_live_observations": {
            "allowed": seeded_ignored[0],
            "action": seeded_ignored[1],
            "reason": seeded_ignored[3],
            "admission": seeded_ignored[4],
        },
        "graphadmit_only_dispatch": {
            "default_mode": strict_default_dispatch[0].name,
            "default_desc": strict_default_dispatch[1].__dict__,
            "admitted_mode": strict_admitted_dispatch[0].name,
            "admitted_desc": strict_admitted_dispatch[1].__dict__,
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
    assert exploratory_replay[0] is True, exploratory_replay
    assert exploratory_replay[3] == "live_explore_replay_until_admitted", exploratory_replay
    assert exploratory_replay[4]["admitted_templates"] == 0, exploratory_replay
    assert default_safe[0] is True, default_safe
    assert graphadmit_only_default[0] is False, graphadmit_only_default
    assert (
        graphadmit_only_default[3] == "graphadmit_only_non_admitted_default"
    ), graphadmit_only_default
    assert forced_runtime_fallback[0] is False, forced_runtime_fallback
    assert forced_runtime_fallback[3] == "forced_shadow_fallback", forced_runtime_fallback
    assert control_forced_runtime_fallback[0] is False, control_forced_runtime_fallback
    assert control_forced_runtime_fallback[3] == "forced_shadow_fallback", control_forced_runtime_fallback
    assert probe_blacklisted == 1, probe_rows
    assert {row["template_id"] for row in probe_rows} == {
        "ours_cp:960:1024:template=1024:reqs=1",
    }
    assert all(
        row["validation_mode"] == "isolated_probe_dispatch_crash_blacklist"
        for row in probe_rows
    )
    assert all(
        row["probe_blacklist_scope"] == "last_graph_dispatch"
        for row in probe_rows
    )
    assert seeded_ignored[0] is False, seeded_ignored
    assert seeded_ignored[3] == "live_explore_fallback_until_admitted", seeded_ignored
    assert strict_default_dispatch[0] == CUDAGraphMode.NONE, strict_default_dispatch
    assert strict_admitted_dispatch[0] == CUDAGraphMode.PIECEWISE, strict_admitted_dispatch
    assert strict_admitted_dispatch[1].num_tokens == 832, strict_admitted_dispatch
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
