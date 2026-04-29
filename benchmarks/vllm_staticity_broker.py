#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prefill_graph.planner.dp_solver import VLLM_DEFAULT_SIZES
from prefill_graph.runtime import (
    DynamicityProfiler,
    RequestContext,
    RuntimePlanner,
    RuntimePolicy,
    TemplateAdmissionController,
    TemplateAwareScheduler,
)


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"avg_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    arr = np.array(values, dtype=np.float64)
    return {
        "avg_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def ceil_bucket(value: int, buckets: list[int]) -> int:
    for bucket in buckets:
        if value <= bucket:
            return bucket
    return buckets[-1] if buckets else value


def request_time_ms(req: dict[str, Any], idx: int) -> float:
    if "timestamp" in req:
        return float(req["timestamp"]) * 1000.0
    return float(idx)


def choose_action(planner: RuntimePlanner, idx: int, tokens: int, allowed_actions: set[str]) -> str:
    action = planner.choose(RequestContext(idx=idx, tokens=tokens, mode="prefill")).action
    if action in allowed_actions:
        return action
    if action == "ours" and "ours_cp" in allowed_actions:
        return "ours_cp"
    if action == "default" and "cp" in allowed_actions:
        return "cp"
    return sorted(allowed_actions)[0]


def expected_wait_budget_ms(policy: RuntimePolicy, action: str, fallback_action: str, max_wait_ms: float) -> float:
    if max_wait_ms <= 0:
        return 0.0
    action_stats = policy.action_stats.get(action)
    fallback_stats = policy.action_stats.get(fallback_action)
    if action_stats is None or fallback_stats is None:
        return 0.0
    if action_stats.avg is None or fallback_stats.avg is None:
        return 0.0
    saving_ms = float(fallback_stats.avg) - float(action_stats.avg)
    if saving_ms <= 0:
        return 0.0
    return max(0.0, min(max_wait_ms, saving_ms))


def load_reference_outputs(path: str | None, config_name: str) -> list[list[int]] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "results" in data:
        for result in data["results"]:
            if result.get("config") == config_name:
                return [
                    [int(value) for value in row.get("output_token_ids", [])]
                    for row in result.get("per_req", [])
                ]
        raise KeyError(f"reference config {config_name!r} not found in {path}")
    if "rows" in data:
        return [
            [int(value) for value in row.get("output_token_ids", [])]
            for row in data["rows"]
        ]
    if "per_req" in data:
        return [
            [int(value) for value in row.get("output_token_ids", [])]
            for row in data["per_req"]
        ]
    raise KeyError(f"could not find reference outputs in {path}")


def worker_loop(args: argparse.Namespace) -> None:
    import torch
    from vllm import LLM, SamplingParams

    worker_dir = Path(args.worker_dir)
    worker_dir.mkdir(parents=True, exist_ok=True)
    capture_sizes = parse_int_list(args.capture_sizes)
    compilation_config = {"cudagraph_mode": "FULL_AND_PIECEWISE"}
    if capture_sizes:
        compilation_config["cudagraph_capture_sizes"] = capture_sizes
        compilation_config["max_cudagraph_capture_size"] = max(capture_sizes)
    profile = worker_dir / f"{args.action}_dispatcher.jsonl"
    os.environ["VLLM_CG_TRACE_FILE"] = str(profile)
    init_start = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        disable_log_stats=True,
        enable_chunked_prefill=True,
        compilation_config=compilation_config,
    )
    os.environ.pop("VLLM_CG_TRACE_FILE", None)
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    warmup_s = 0.0
    if args.worker_warmup_iters > 0:
        warmup_start = time.perf_counter()
        for _ in range(args.worker_warmup_iters):
            llm.generate([args.worker_warmup_prompt], sampling, use_tqdm=False)
            torch.cuda.synchronize()
        warmup_s = time.perf_counter() - warmup_start
    ready = {
        "action": args.action,
        "init_s": time.perf_counter() - init_start,
        "warmup_s": warmup_s,
        "worker_warmup_iters": args.worker_warmup_iters,
        "capture_sizes": capture_sizes,
        "profile": str(profile),
    }
    (worker_dir / "ready.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")

    processed: set[str] = set()
    while True:
        if (worker_dir / "stop.json").exists():
            break
        requests = sorted(worker_dir.glob("request_*.json"))
        did_work = False
        for request_path in requests:
            request_id = request_path.stem.replace("request_", "")
            if request_id in processed:
                continue
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            did_work = True
            items = request.get("items")
            if items is None:
                items = [
                    {
                        "idx": request["idx"],
                        "tok": request["tok"],
                        "prompt": request["prompt"],
                    }
                ]
            prompts = [item["prompt"] for item in items]
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - start) * 1000.0
            rows = []
            for item, request_output in zip(items, outputs):
                output = request_output.outputs[0] if request_output.outputs else None
                rows.append(
                    {
                        "idx": item["idx"],
                        "tok": item["tok"],
                        "action": args.action,
                        "ms": ms,
                        "service_ms": ms,
                        "output_token_ids": [
                            int(value) for value in (output.token_ids if output is not None else [])
                        ],
                        "output_text": output.text if output is not None else "",
                    }
                )
            response = {
                "action": args.action,
                "batch_size": len(rows),
                "service_ms": ms,
                "rows": rows,
            }
            if len(rows) == 1:
                response.update(rows[0])
            response_path = worker_dir / f"response_{request_id}.json"
            response_path.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
            processed.add(request_id)
        if not did_work:
            time.sleep(0.01)
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def spawn_worker(
    *,
    action: str,
    worker_dir: Path,
    devices: str,
    port: int,
    args: argparse.Namespace,
    capture_sizes: list[int],
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-dir",
        str(worker_dir),
        "--action",
        action,
        "--model",
        args.model,
        "--tp-size",
        str(args.tp_size),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-tokens",
        str(args.max_tokens),
        "--capture-sizes",
        ",".join(str(size) for size in capture_sizes),
        "--worker-warmup-iters",
        str(args.worker_warmup_iters),
        "--worker-warmup-prompt",
        args.worker_warmup_prompt,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices
    env["VLLM_PORT"] = str(port)
    return subprocess.Popen(cmd, env=env)


def wait_ready(worker_dir: Path, proc: subprocess.Popen, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    ready_path = worker_dir / "ready.json"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"worker {worker_dir} exited with code {proc.returncode}")
        if ready_path.exists():
            return json.loads(ready_path.read_text(encoding="utf-8"))
        time.sleep(0.5)
    raise TimeoutError(f"worker {worker_dir} did not become ready in {timeout_s}s")


def send_request(worker_dir: Path, request_id: int, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request_path = worker_dir / f"request_{request_id:06d}.json"
    response_path = worker_dir / f"response_{request_id:06d}.json"
    request_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if response_path.exists():
            return json.loads(response_path.read_text(encoding="utf-8"))
        time.sleep(0.01)
    raise TimeoutError(f"request {request_id} timed out for worker {worker_dir}")


def build_request_item(
    req: dict[str, Any],
    idx: int,
    planner: RuntimePlanner,
    actions: set[str],
    buckets: list[int],
    profiler: DynamicityProfiler,
    admission: TemplateAdmissionController,
    fallback_action: str,
    max_wait_ms: float = 0.0,
) -> dict[str, Any]:
    tokens = int(req["actual_input_length"])
    action = choose_action(planner, idx, tokens, actions)
    bucket = ceil_bucket(tokens, buckets)
    template_id = f"{action}:tok<={bucket}"
    can_admit, reject_reason = admission.can_capture(template_id)
    if can_admit:
        admission.state(template_id).metadata.update({"action": action, "bucket": bucket})
    if not can_admit and fallback_action in actions:
        action = fallback_action
        bucket = ceil_bucket(tokens, buckets)
        template_id = f"{action}:tok<={bucket}"
        fallback_admit, fallback_reject_reason = admission.can_capture(template_id)
        if fallback_admit:
            admission.state(template_id).metadata.update({"action": action, "bucket": bucket})
        reject_reason = reject_reason or fallback_reject_reason
    arrival_ms = request_time_ms(req, idx)
    wait_budget_ms = expected_wait_budget_ms(
        planner.policy,
        action,
        fallback_action,
        max_wait_ms,
    )
    profiler.observe("num_tokens", tokens, in_graph_key=True, semantic=True, component="vllm_broker")
    profiler.observe("runtime_action", action, in_graph_key=True, semantic=False, component="vllm_broker")
    profiler.observe("template_id", template_id, in_graph_key=True, semantic=False, component="vllm_broker")
    profiler.observe("arrival_ms_bucket", int(arrival_ms // 100), in_graph_key=False, semantic=False, component="vllm_broker")
    return {
        "idx": idx,
        "tok": tokens,
        "prompt": req["prompt"],
        "action": action,
        "bucket": bucket,
        "template_id": template_id,
        "arrival_ms": arrival_ms,
        "scheduler_wait_budget_ms": wait_budget_ms,
        "admission_reject_reason": reject_reason,
    }


def make_scheduler(args: argparse.Namespace) -> TemplateAwareScheduler:
    return TemplateAwareScheduler(
        lambda request: str(request["template_id"]),
        max_wait_ms=args.scheduler_max_wait_ms,
        max_batch_size=args.scheduler_max_batch_size,
        adaptive_wait=args.scheduler_adaptive_wait,
        adaptive_min_samples=args.scheduler_adaptive_min_samples,
        adaptive_min_hit_rate=args.scheduler_adaptive_min_hit_rate,
    )


def build_scheduled_batches(
    reqs: list[dict[str, Any]],
    planner: RuntimePlanner,
    actions: set[str],
    buckets: list[int],
    max_wait_ms: float,
    max_batch_size: int,
    profiler: DynamicityProfiler,
    admission: TemplateAdmissionController,
    fallback_action: str,
    adaptive_wait: bool = False,
    adaptive_min_samples: int = 4,
    adaptive_min_hit_rate: float = 0.5,
) -> tuple[list[Any], TemplateAwareScheduler]:
    scheduler_args = argparse.Namespace(
        scheduler_max_wait_ms=max_wait_ms,
        scheduler_max_batch_size=max_batch_size,
        scheduler_adaptive_wait=adaptive_wait,
        scheduler_adaptive_min_samples=adaptive_min_samples,
        scheduler_adaptive_min_hit_rate=adaptive_min_hit_rate,
    )
    scheduler = make_scheduler(scheduler_args)
    batches = []
    for idx, req in enumerate(reqs):
        item = build_request_item(
            req,
            idx,
            planner,
            actions,
            buckets,
            profiler,
            admission,
            fallback_action,
            max_wait_ms,
        )
        batches.extend(scheduler.add(item, item["arrival_ms"]))
    batches.extend(scheduler.finish())
    return batches, scheduler


def execute_batch(
    *,
    batch: Any,
    batch_idx: int,
    worker_dirs: dict[str, Path],
    procs: dict[str, subprocess.Popen],
    args: argparse.Namespace,
    reference_outputs: list[list[int]] | None,
    admission: TemplateAdmissionController,
) -> list[dict[str, Any]]:
    action = str(batch.requests[0]["action"])
    if action not in procs:
        action = sorted(procs)[0]
    response = send_request(
        worker_dirs[action],
        batch_idx,
        {
            "batch_idx": batch_idx,
            "template_id": batch.template_id,
            "items": batch.requests,
        },
        args.request_timeout_s,
    )
    service_ms = float(response["service_ms"])
    response_rows = response.get("rows", [response])
    wait_by_idx = {
        int(item["idx"]): float(wait_ms)
        for item, wait_ms in zip(batch.requests, batch.wait_ms)
    }
    request_by_idx = {int(item["idx"]): item for item in batch.requests}
    final_rows = []
    for row in response_rows:
        idx = int(row["idx"])
        item = request_by_idx[idx]
        wait_ms = wait_by_idx[idx]
        final_row = {
            **row,
            "batch_idx": batch_idx,
            "batch_size": len(batch.requests),
            "template_id": batch.template_id,
            "bucket": item["bucket"],
            "arrival_ms": item["arrival_ms"],
            "flush_time_ms": batch.flush_time_ms,
            "wait_ms": wait_ms,
            "service_ms": service_ms,
            "e2e_ms": wait_ms + service_ms,
            "admission_reject_reason": item.get("admission_reject_reason"),
        }
        reference_ok = None
        if reference_outputs is not None and idx < len(reference_outputs):
            reference_ok = final_row["output_token_ids"] == reference_outputs[idx]
            final_row["same_output_vs_reference"] = reference_ok
            admission.record_validation(batch.template_id, reference_ok)
        else:
            admission.record_validation(batch.template_id, True)
        admission.record_replay(batch.template_id, service_ms / 1000.0)
        if (
            reference_ok is False
            and args.retry_on_mismatch
            and args.fallback_action in procs
            and action != args.fallback_action
        ):
            retry = send_request(
                worker_dirs[args.fallback_action],
                1_000_000 + idx,
                {
                    "batch_idx": batch_idx,
                    "template_id": f"{args.fallback_action}:retry",
                    "items": [item],
                },
                args.request_timeout_s,
            )
            retry_row = retry.get("rows", [retry])[0]
            final_row.update(
                {
                    **retry_row,
                    "action": args.fallback_action,
                    "retried_after_mismatch": True,
                    "retry_service_ms": float(retry["service_ms"]),
                    "service_ms": float(retry["service_ms"]),
                    "e2e_ms": wait_ms + float(retry["service_ms"]),
                }
            )
            if idx < len(reference_outputs):
                final_row["same_output_vs_reference"] = (
                    final_row["output_token_ids"] == reference_outputs[idx]
                )
        final_rows.append(final_row)
    print(
        f"[batch {batch_idx + 1}] "
        f"template={batch.template_id} action={action} "
        f"n={len(batch.requests)} service={service_ms:.2f} ms "
        f"wait_max={max(batch.wait_ms) if batch.wait_ms else 0.0:.2f} ms"
    )
    return final_rows


def broker_main(args: argparse.Namespace) -> None:
    workload = json.loads(Path(args.workload).read_text(encoding="utf-8"))
    reqs = workload["requests"][: args.limit]
    policy = RuntimePolicy.from_json_file(args.policy)
    planner = RuntimePlanner(policy)
    profiler = DynamicityProfiler(args.profile_jsonl)
    admission = TemplateAdmissionController(max_templates=args.admission_max_templates)
    reference_outputs = load_reference_outputs(args.reference_results, args.reference_config)
    actions = {item.strip() for item in args.actions.split(",") if item.strip()}
    default_small = [size for size in VLLM_DEFAULT_SIZES if size <= 512]
    ours_sizes = sorted(set(default_small + parse_int_list(args.extra_capture_sizes)))
    scheduler_buckets = parse_int_list(args.scheduler_buckets)
    if not scheduler_buckets:
        scheduler_buckets = sorted(set(default_small + parse_int_list(args.extra_capture_sizes) + [1024, 1536, 2048, 3072, 4096]))
    if args.dry_run_scheduler:
        batches, scheduler = build_scheduled_batches(
            reqs,
            planner,
            actions,
            scheduler_buckets,
            args.scheduler_max_wait_ms,
            args.scheduler_max_batch_size,
            profiler,
            admission,
            args.fallback_action,
            args.scheduler_adaptive_wait,
            args.scheduler_adaptive_min_samples,
            args.scheduler_adaptive_min_hit_rate,
        )
        result = {
            "kind": "vllm_staticity_broker_scheduler_dry_run",
            "model": args.model,
            "workload": args.workload,
            "policy": args.policy,
            "actions": sorted(actions),
            "scheduler": scheduler.summary(),
            "dynamicity": profiler.summary(),
            "admission": admission.summary(),
            "batches": [
                {
                    "batch_idx": idx,
                    "template_id": batch.template_id,
                    "flush_time_ms": batch.flush_time_ms,
                    "wait_ms": batch.wait_ms,
                    "idx": [item["idx"] for item in batch.requests],
                    "tokens": [item["tok"] for item in batch.requests],
                    "actions": [item["action"] for item in batch.requests],
                }
                for idx, batch in enumerate(batches)
            ],
            "planner": planner.summary(),
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "batches"}, indent=2, ensure_ascii=False))
        return

    with tempfile.TemporaryDirectory(prefix="staticity_broker_") as tmp:
        root = Path(args.worker_root) if args.worker_root else Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        procs: dict[str, subprocess.Popen] = {}
        worker_dirs: dict[str, Path] = {}
        try:
            if "cp" in actions:
                worker_dirs["cp"] = root / "cp"
                procs["cp"] = spawn_worker(
                    action="cp",
                    worker_dir=worker_dirs["cp"],
                    devices=args.cp_devices,
                    port=args.cp_port,
                    args=args,
                    capture_sizes=[],
                )
            if "ours_cp" in actions:
                worker_dirs["ours_cp"] = root / "ours_cp"
                procs["ours_cp"] = spawn_worker(
                    action="ours_cp",
                    worker_dir=worker_dirs["ours_cp"],
                    devices=args.ours_cp_devices,
                    port=args.ours_cp_port,
                    args=args,
                    capture_sizes=ours_sizes,
                )
            ready = {
                action: wait_ready(worker_dirs[action], proc, args.worker_ready_timeout_s)
                for action, proc in procs.items()
            }
            rows: list[dict[str, Any]] = []
            scheduler = make_scheduler(args)
            batch_idx = 0
            for idx, req in enumerate(reqs):
                item = build_request_item(
                    req,
                    idx,
                    planner,
                    set(procs),
                    scheduler_buckets,
                    profiler,
                    admission,
                    args.fallback_action,
                    args.scheduler_max_wait_ms,
                )
                for batch in scheduler.add(item, item["arrival_ms"]):
                    rows.extend(
                        execute_batch(
                            batch=batch,
                            batch_idx=batch_idx,
                            worker_dirs=worker_dirs,
                            procs=procs,
                            args=args,
                            reference_outputs=reference_outputs,
                            admission=admission,
                        )
                    )
                    batch_idx += 1
            for batch in scheduler.finish():
                rows.extend(
                    execute_batch(
                        batch=batch,
                        batch_idx=batch_idx,
                        worker_dirs=worker_dirs,
                        procs=procs,
                        args=args,
                        reference_outputs=reference_outputs,
                        admission=admission,
                    )
                )
                batch_idx += 1
            rows.sort(key=lambda row: int(row["idx"]))
            service_latencies = [float(row["service_ms"]) for row in rows]
            e2e_latencies = [float(row["e2e_ms"]) for row in rows]
            result = {
                "kind": "vllm_staticity_broker",
                "model": args.model,
                "workload": args.workload,
                "policy": args.policy,
                "actions": sorted(procs),
                "worker_ready": ready,
                "stats": stats(e2e_latencies),
                "service_stats": stats(service_latencies),
                "scheduler": scheduler.summary(),
                "dynamicity": profiler.summary(),
                "admission": admission.summary(),
                "all_same_outputs_vs_reference": (
                    all(bool(row.get("same_output_vs_reference")) for row in rows)
                    if reference_outputs is not None
                    else None
                ),
                "planner": planner.summary(),
                "rows": rows,
                "boundary": (
                    "This is a real online multi-process broker when workers use disjoint "
                    "CUDA_VISIBLE_DEVICES. It is intended for Qwen3-32B TP4 on an 8-GPU node. "
                    "Qwen3-235B TP8 cannot run two such workers on the same 8-GPU node."
                ),
            }
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, ensure_ascii=False))
        finally:
            for worker_dir in worker_dirs.values():
                worker_dir.mkdir(parents=True, exist_ok=True)
                (worker_dir / "stop.json").write_text("{}", encoding="utf-8")
            for proc in procs.values():
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-dir")
    parser.add_argument("--worker-root")
    parser.add_argument("--action", default="")
    parser.add_argument("--model", default="/mnt/models/Qwen3-32B")
    parser.add_argument("--workload", default="results/qwentrace_morspec_qwen_64_4096.json")
    parser.add_argument("--policy", default="results/runtime_policy_vllm_qwen3_32b_64_broker2.json")
    parser.add_argument("--output", default="results/vllm_qwen3_32b_staticity_broker.json")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--actions", default="cp,ours_cp")
    parser.add_argument("--cp-devices", default="0,1,2,3")
    parser.add_argument("--ours-cp-devices", default="4,5,6,7")
    parser.add_argument("--cp-port", type=int, default=46831)
    parser.add_argument("--ours-cp-port", type=int, default=46861)
    parser.add_argument("--extra-capture-sizes", default="768,832,896,960,1024")
    parser.add_argument("--capture-sizes", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--worker-ready-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--worker-warmup-iters", type=int, default=0)
    parser.add_argument(
        "--worker-warmup-prompt",
        default=("Warm up the model with a deterministic one-token response. " * 64).strip(),
    )
    parser.add_argument("--scheduler-max-wait-ms", type=float, default=0.0)
    parser.add_argument("--scheduler-max-batch-size", type=int, default=1)
    parser.add_argument("--scheduler-buckets", default="")
    parser.add_argument("--scheduler-adaptive-wait", action="store_true")
    parser.add_argument("--scheduler-adaptive-min-samples", type=int, default=4)
    parser.add_argument("--scheduler-adaptive-min-hit-rate", type=float, default=0.5)
    parser.add_argument("--profile-jsonl")
    parser.add_argument("--admission-max-templates", type=int, default=0)
    parser.add_argument("--fallback-action", default="cp")
    parser.add_argument("--reference-results")
    parser.add_argument("--reference-config", default="2. vLLM graph max512 no-CP")
    parser.add_argument("--retry-on-mismatch", action="store_true")
    parser.add_argument("--dry-run-scheduler", action="store_true")
    args = parser.parse_args()

    if args.worker:
        if not args.worker_dir:
            raise ValueError("--worker-dir is required in worker mode")
        worker_loop(args)
        return
    broker_main(args)


if __name__ == "__main__":
    main()
