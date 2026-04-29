#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def summarize_ranges(result):
    import numpy as np

    lens = np.array([row["tok"] for row in result["per_req"]])
    ms = np.array([row["ms"] for row in result["per_req"]])
    out = {}
    for lo, hi in [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 32768)]:
        mask = (lens > lo) & (lens <= hi)
        if mask.any():
            vals = ms[mask]
            out[f"({lo},{hi}]"] = {
                "n": int(mask.sum()),
                "avg_ms": float(vals.mean()),
                "p95_ms": float(np.percentile(vals, 95)),
            }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    merged = None
    results = []
    seen = set()
    for path in args.inputs:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if merged is None:
            merged = {key: value for key, value in data.items() if key != "results"}
        for result in data.get("results", []):
            name = result["config"]
            if name in seen:
                continue
            seen.add(name)
            results.append(result)

    if not results:
        raise ValueError("no results to merge")

    base = results[0]["avg_ms"]
    reference_outputs = [row["output_token_ids"] for row in results[0]["per_req"]]
    for result in results:
        result["speedup_vs_first"] = base / result["avg_ms"] if result["avg_ms"] > 0 else None
        result["ranges"] = summarize_ranges(result)
        result["same_outputs_vs_first"] = [
            row["output_token_ids"] == reference
            for row, reference in zip(result["per_req"], reference_outputs)
        ]
        result["all_same_outputs_vs_first"] = all(result["same_outputs_vs_first"])

    merged["partial"] = False
    merged["results"] = results
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    for result in results:
        print(
            f"{result['config']}: avg={result['avg_ms']:.2f} "
            f"p95={result['p95_ms']:.2f} p99={result['p99_ms']:.2f} "
            f"speedup={result['speedup_vs_first']:.2f} "
            f"same={result['all_same_outputs_vs_first']}"
        )
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
