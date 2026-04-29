#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--default-action", default="cp")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    policy = data.get("runtime_policy", data)
    strict = dict(policy)
    strict["rules"] = [
        {**rule, "action": args.default_action}
        for rule in policy.get("rules", [])
    ]
    strict["default_action"] = args.default_action
    strict["single_engine_graph_actions"] = ["default", "cp"]
    strict["single_engine_fallback_actions"] = [
        "eager",
        "compile",
        "compiled",
        "fallback",
        "none",
        "ours",
        "ours_cp",
    ]
    strict["strict_online"] = True
    strict["strict_online_reason"] = (
        "disable extra graph templates for multi-request online batches "
        "until metadata/key-collapse correctness is proven"
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runtime_policy": strict}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "rules": len(strict.get("rules", [])),
        "default_action": strict["default_action"],
        "single_engine_graph_actions": strict["single_engine_graph_actions"],
    }, indent=2))


if __name__ == "__main__":
    main()
