#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from shutil import copy2


PATCHED_FILES = [
    "vllm/v1/cudagraph_dispatcher.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/worker/gpu_model_runner.py",
    "vllm/model_executor/layers/fused_moe/routed_experts_capturer.py",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="surrogateescape")


def has_staticity_markers(path: Path) -> bool:
    text = read_text(path)
    markers = [
        "STATICITY_VLLM_RUNTIME_POLICY",
        "STATICITY_VLLM_FIXED_METADATA_ARENA",
        "STATICITY_VLLM_LIVE_ADMISSION",
        "STATICITY_VLLM_TEMPLATE_SCHEDULER",
        "STATICITY_VLLM_MOE_PROFILE",
        "staticity_template_tokens",
        "_staticity_runtime_allows_graph",
        "metadata_template_id",
    ]
    return any(marker in text for marker in markers)


def make_patch(source_root: Path, installed_root: Path, output: Path) -> dict[str, object]:
    chunks: list[str] = []
    manifest = {"source_root": str(source_root), "installed_root": str(installed_root), "files": []}
    for rel in PATCHED_FILES:
        src = source_root / rel
        dst = installed_root / rel
        if not dst.exists():
            raise FileNotFoundError(dst)
        before = read_text(src).splitlines(keepends=True) if src.exists() else []
        after = read_text(dst).splitlines(keepends=True)
        diff = list(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                n=3,
            )
        )
        chunks.extend(diff)
        manifest["files"].append({
            "path": rel,
            "source_exists": src.exists(),
            "source_has_staticity": src.exists() and has_staticity_markers(src),
            "installed_has_staticity": has_staticity_markers(dst),
            "diff_lines": len(diff),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(chunks), encoding="utf-8")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"patch": str(output), "manifest": str(manifest_path), **manifest}


def apply_from_installed(source_root: Path, installed_root: Path) -> dict[str, object]:
    copied = []
    for rel in PATCHED_FILES:
        src = installed_root / rel
        dst = source_root / rel
        if not src.exists():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        copy2(src, dst)
        copied.append({"path": rel, "staticity_markers": has_staticity_markers(dst)})
    return {"source_root": str(source_root), "installed_root": str(installed_root), "copied": copied}


def verify(source_root: Path) -> dict[str, object]:
    files = []
    ok = True
    for rel in PATCHED_FILES:
        path = source_root / rel
        exists = path.exists()
        markers = exists and has_staticity_markers(path)
        ok = ok and exists and markers
        files.append({"path": rel, "exists": exists, "staticity_markers": markers})
    return {"source_root": str(source_root), "ok": ok, "files": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default="external/FlowPrefill",
        help="vLLM source tree to materialize/check",
    )
    parser.add_argument(
        "--installed-root",
        default="/home/zhujianian/miniconda3/envs/crossstage/lib/python3.10/site-packages",
        help="site-packages root containing patched vLLM",
    )
    parser.add_argument(
        "--output",
        default="results/vllm_staticity_sitepackages.patch",
        help="patch output path for --make-patch",
    )
    parser.add_argument(
        "--mode",
        choices=["make-patch", "apply-from-installed", "verify"],
        default="make-patch",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    installed_root = Path(args.installed_root)
    if args.mode == "make-patch":
        result = make_patch(source_root, installed_root, Path(args.output))
    elif args.mode == "apply-from-installed":
        result = apply_from_installed(source_root, installed_root)
    else:
        result = verify(source_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
