#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def desc_key(desc):
    if not isinstance(desc, dict):
        return str(desc)
    return (desc.get("num_tokens"), desc.get("num_reqs"), desc.get("uniform"),
            desc.get("has_lora"), desc.get("num_active_loras"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args()
    summaries = {}
    for file_name in args.files:
        path = Path(file_name)
        events = []
        if path.exists():
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        dispatch = [e for e in events if e.get("kind") == "dispatch"]
        captures = [e for e in events if e.get("kind") == "capture_done"]
        replays = [e for e in events if e.get("kind") == "replay"]
        mode_counts = Counter(e.get("mode") for e in dispatch)
        reason_counts = Counter(e.get("reason") for e in dispatch)
        family_by_mode = defaultdict(set)
        for e in dispatch:
            family_by_mode[e.get("mode")].add(desc_key(e.get("batch_descriptor")))
        replay_family_by_mode = defaultdict(set)
        for e in replays:
            replay_family_by_mode[e.get("mode")].add(desc_key(e.get("batch_descriptor")))
        requested = [e.get("requested_num_tokens") for e in dispatch if isinstance(e.get("requested_num_tokens"), int)]
        padded = []
        for e in dispatch:
            desc = e.get("batch_descriptor")
            if isinstance(desc, dict) and isinstance(desc.get("num_tokens"), int):
                padded.append(desc["num_tokens"])
        summary = {
            "events": len(events),
            "dispatches": len(dispatch),
            "captures": len(captures),
            "replays": len(replays),
            "mode_counts": dict(mode_counts),
            "reason_counts": dict(reason_counts),
            "dispatch_families_by_mode": {k: len(v) for k, v in family_by_mode.items()},
            "replay_families_by_mode": {k: len(v) for k, v in replay_family_by_mode.items()},
            "requested_min_max": [min(requested), max(requested)] if requested else None,
            "padded_min_max": [min(padded), max(padded)] if padded else None,
            "top_dispatch_keys": [
                {"key": list(k) if isinstance(k, tuple) else k, "count": v}
                for k, v in Counter(desc_key(e.get("batch_descriptor")) for e in dispatch).most_common(12)
            ],
        }
        summaries[str(path)] = summary
        print(f"\n{path}")
        print(json.dumps(summary, indent=2, sort_keys=True))
    if args.out:
        Path(args.out).parent.mkdir(exist_ok=True, parents=True)
        json.dump(summaries, open(args.out, "w"), indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
