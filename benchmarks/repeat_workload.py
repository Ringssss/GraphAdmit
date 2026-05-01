#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def length_summary(requests: list[dict]) -> dict[str, float | int]:
    lengths = [
        int(req.get("actual_input_length", req.get("target_input_length", 0)))
        for req in requests
    ]
    return {
        "min": min(lengths) if lengths else 0,
        "p50": percentile(lengths, 50.0),
        "p90": percentile(lengths, 90.0),
        "p95": percentile(lengths, 95.0),
        "p99": percentile(lengths, 99.0),
        "max": max(lengths) if lengths else 0,
        "gt512": sum(1 for length in lengths if length > 512),
        "gt1024": sum(1 for length in lengths if length > 1024),
        "gt2048": sum(1 for length in lengths if length > 2048),
    }


def repeat_requests(source: list[dict], count: int) -> list[dict]:
    if not source:
        raise ValueError("source workload has no requests")
    requests: list[dict] = []
    for idx in range(count):
        original = source[idx % len(source)]
        req = copy.deepcopy(original)
        rounds = idx // len(source)
        req["idx"] = idx
        req["chat_id"] = int(req.get("chat_id", idx)) + rounds * 1_000_000
        req["timestamp"] = float(req.get("timestamp", 0.0)) + rounds * 10_000.0
        requests.append(req)
    return requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    requests = repeat_requests(payload.get("requests", []), args.count)
    payload["requests"] = requests
    payload["num_requests"] = len(requests)
    payload["length_summary"] = length_summary(requests)
    payload["repeat_source"] = str(Path(args.input))
    payload["repeat_count"] = int(args.count)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": args.output,
        "num_requests": len(requests),
        "length_summary": payload["length_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
