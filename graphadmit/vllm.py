from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .policy import load_policy


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TREE_PATCH_PATH = PACKAGE_ROOT / "patches" / "vllm_staticity.patch"
RESOURCE_PATCH_PATH = Path(__file__).resolve().parent / "resources" / "vllm_staticity.patch"
PATCH_PATH = SOURCE_TREE_PATCH_PATH if SOURCE_TREE_PATCH_PATH.exists() else RESOURCE_PATCH_PATH


@dataclass(frozen=True)
class PatchStatus:
    patch: Path
    target: Path
    status: str
    detail: str

    def to_json(self) -> dict[str, str]:
        return {
            "patch": str(self.patch),
            "target": str(self.target),
            "status": self.status,
            "detail": self.detail,
        }


def _run_git_apply(
    target: Path,
    patch: Path,
    *args: str,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "apply", *args, str(patch)],
        cwd=str(target),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def check_patch(target: str | Path, patch: str | Path | None = None) -> PatchStatus:
    target_path = Path(target).resolve()
    patch_path = Path(patch).resolve() if patch is not None else PATCH_PATH
    if not target_path.exists():
        return PatchStatus(patch_path, target_path, "missing-target", "target path does not exist")
    if not patch_path.exists():
        return PatchStatus(patch_path, target_path, "missing-patch", "patch file does not exist")

    forward = _run_git_apply(target_path, patch_path, "--check")
    if forward.returncode == 0:
        return PatchStatus(patch_path, target_path, "clean", "patch can be applied")

    reverse = _run_git_apply(target_path, patch_path, "--reverse", "--check")
    if reverse.returncode == 0:
        return PatchStatus(patch_path, target_path, "applied", "patch is already present")

    detail = (forward.stderr or reverse.stderr or "patch does not apply").strip()
    return PatchStatus(patch_path, target_path, "conflict", detail)


def apply_patch(target: str | Path, patch: str | Path | None = None) -> PatchStatus:
    status = check_patch(target, patch)
    if status.status == "applied":
        return status
    if status.status != "clean":
        return status
    proc = _run_git_apply(status.target, status.patch)
    if proc.returncode != 0:
        return PatchStatus(
            status.patch,
            status.target,
            "failed",
            (proc.stderr or proc.stdout or "git apply failed").strip(),
        )
    return check_patch(status.target, status.patch)


def env_for_policy(
    policy: str | Path,
    *,
    observations: str | Path | None = None,
    active: bool = True,
    live_admission: bool | None = None,
    shadow_baseline: bool = True,
    fixed_metadata_arena: bool = True,
    template_scheduler: bool = False,
    live_capture: bool | None = None,
) -> dict[str, str]:
    policy_path = Path(policy).resolve()
    data = load_policy(policy_path)
    runtime_policy: dict[str, Any] = data.get("runtime_policy", data)
    live = runtime_policy.get("live_admission", {})
    capture = runtime_policy.get("live_capture", {})
    live_enabled = bool(live.get("enabled", False)) if live_admission is None else bool(live_admission)
    live_capture_enabled = (
        bool(capture.get("enabled", False))
        if live_capture is None else bool(live_capture)
    )

    env = {
        "STATICITY_VLLM_RUNTIME_POLICY": str(policy_path),
        "STATICITY_VLLM_RUNTIME_ACTIVE": "1" if active else "0",
        "STATICITY_VLLM_LIVE_ADMISSION": "1" if live_enabled else "0",
        "STATICITY_VLLM_LIVE_EXPLORE": "1" if live_enabled else "0",
        "STATICITY_VLLM_LIVE_MIN_SAMPLES": str(int(live.get("min_samples", 2))),
        "STATICITY_VLLM_LIVE_MIN_USEFUL_RATE": str(float(live.get("min_useful_rate", 0.67))),
        "STATICITY_VLLM_LIVE_MIN_SAVING_MS": str(float(live.get("min_saving_ms", 0.5))),
        "STATICITY_VLLM_FIXED_METADATA_ARENA": "1" if fixed_metadata_arena else "0",
        "STATICITY_VLLM_TEMPLATE_SCHEDULER": "1" if template_scheduler else "0",
        "STATICITY_VLLM_LIVE_CAPTURE": "1" if live_capture_enabled else "0",
        "STATICITY_VLLM_ALLOW_RUNTIME_CUDAGRAPH_CAPTURE": (
            "1" if live_capture_enabled else "0"
        ),
        "STATICITY_VLLM_LIVE_SHADOW_BASELINE": "1" if shadow_baseline else "0",
    }
    if live.get("max_p95_regression_ms") is not None:
        env["STATICITY_VLLM_LIVE_MAX_P95_REGRESSION_MS"] = str(
            float(live["max_p95_regression_ms"])
        )
    if observations is not None:
        env["STATICITY_VLLM_LIVE_OBSERVATIONS"] = str(Path(observations).resolve())
    return env


def shell_exports(env: dict[str, str]) -> str:
    lines = []
    for key in sorted(env):
        value = env[key].replace("'", "'\"'\"'")
        lines.append(f"export {key}='{value}'")
    return "\n".join(lines)


def exec_vllm(args: list[str], env: dict[str, str]) -> int:
    merged = os.environ.copy()
    merged.update(env)
    return subprocess.call(["vllm", *args], env=merged)


def require_vllm() -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import vllm; print(getattr(vllm, '__version__', 'unknown'))"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        return False, str(exc)
    if proc.returncode != 0:
        return False, proc.stderr.strip() or "vLLM import failed"
    return True, proc.stdout.strip()
