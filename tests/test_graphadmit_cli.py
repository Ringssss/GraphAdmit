from __future__ import annotations

import json

from graphadmit.cli import main


def test_make_policy_cli_writes_live_admission_policy(tmp_path) -> None:
    output = tmp_path / "policy.json"
    rc = main([
        "make-policy",
        "--max-tokens",
        "1024",
        "--output",
        str(output),
    ])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    policy = payload["runtime_policy"]
    assert policy["kind"] == "graphadmit_online_exploration_policy"
    assert policy["correctness_required"] is True
    assert policy["live_admission"]["enabled"] is True
    assert policy["residual_capture"]["extra_capture_sizes"]
