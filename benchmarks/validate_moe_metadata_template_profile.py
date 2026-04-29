#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/moe_metadata_template_profile_validation.json")
    parser.add_argument("--workdir", default="results/moe_metadata_template_profile")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "moe.jsonl"
    rows = [
        {
            "component": "moe_routed_experts",
            "layer_id": 0,
            "num_tokens": 750,
            "num_experts_per_tok": 1,
            "configured_num_experts": 128,
            "route_hash": 11,
            "count_hash": 101,
            "data_ptr": 1000,
            "active_experts": 48,
            "max_expert_count": 31,
            "capacity_bucket": 32,
            "metadata_template_id": "moe:tokens=750:topk=1:experts=128:capacity=32",
            "address_template_id": "moe:topk_ptr=1000:shape=(750, 1)",
        },
        {
            "component": "moe_routed_experts",
            "layer_id": 0,
            "num_tokens": 750,
            "num_experts_per_tok": 1,
            "configured_num_experts": 128,
            "route_hash": 12,
            "count_hash": 102,
            "data_ptr": 2000,
            "active_experts": 50,
            "max_expert_count": 33,
            "capacity_bucket": 64,
            "metadata_template_id": "moe:tokens=750:topk=1:experts=128:capacity=64",
            "address_template_id": "moe:topk_ptr=2000:shape=(750, 1)",
        },
        {
            "component": "moe_routed_experts",
            "layer_id": 1,
            "num_tokens": 750,
            "num_experts_per_tok": 1,
            "configured_num_experts": 128,
            "route_hash": 21,
            "count_hash": 201,
            "data_ptr": 3000,
            "active_experts": 47,
            "max_expert_count": 29,
            "capacity_bucket": 32,
            "metadata_template_id": "moe:tokens=750:topk=1:experts=128:capacity=32",
            "address_template_id": "moe:topk_ptr=3000:shape=(750, 1)",
        },
    ]
    log_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    out = Path(args.output)
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("summarize_vllm_moe_profile.py")),
        "--moe-log",
        str(log_path),
        "--output",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    print(out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
