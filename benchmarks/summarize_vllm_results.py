#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def pct(delta):
    return f"{delta * 100:+.1f}%"


def summarize(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data["results"]
    by_name = {r["config"]: r for r in results}
    eager = results[0]
    ours = next((r for r in results if r["config"].startswith("3. Ours")), None)
    vllm = next((r for r in results if "vLLM graph max512 no-CP" in r["config"]), None)
    cp = next((r for r in results if "CP" in r["config"] and r["config"].startswith("4.")), None)
    print(f"\n# {path}")
    print(f"workload={data.get('workload')} stats={data.get('workload_stats')}")
    print(f"planner={data.get('planner', {}).get('mode')} note={data.get('planner', {}).get('note')}")
    for r in results:
        print(f"- {r['config']}: avg={r['avg_ms']:.2f} p50={r['p50_ms']:.2f} p95={r['p95_ms']:.2f} p99={r['p99_ms']:.2f} init={r['init_s']:.1f}")
    if ours and vllm:
        print("comparisons_vs_vllm_no_cp:")
        for key, label in [("avg_ms", "avg"), ("p95_ms", "p95"), ("p99_ms", "p99")]:
            print(f"  {label}: {pct((ours[key] - vllm[key]) / vllm[key])} ({ours[key]:.2f} vs {vllm[key]:.2f})")
    if ours and cp:
        print("comparisons_vs_vllm_cp:")
        for key, label in [("avg_ms", "avg"), ("p95_ms", "p95"), ("p99_ms", "p99")]:
            print(f"  {label}: {pct((ours[key] - cp[key]) / cp[key])} ({ours[key]:.2f} vs {cp[key]:.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for item in args.paths:
        summarize(Path(item))


if __name__ == "__main__":
    main()
