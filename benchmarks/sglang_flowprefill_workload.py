#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import inspect
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch


def parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def extract_output(row: Any) -> dict[str, Any]:
    if isinstance(row, list):
        row = row[0] if row else {}
    if not isinstance(row, dict):
        return {"token_ids": [], "text": str(row)}
    text = row.get("text") or row.get("output_text") or ""
    token_ids = (
        row.get("output_token_ids")
        or row.get("token_ids")
        or row.get("output_ids")
        or []
    )
    meta = row.get("meta_info") or {}
    if not token_ids and isinstance(meta, dict):
        token_ids = meta.get("output_token_ids") or meta.get("token_ids") or []
    return {
        "token_ids": [int(x) for x in token_ids],
        "text": text,
    }


def run_config(
    *,
    config_name: str,
    model_path: str,
    prompts: list[str],
    lens: list[int],
    tp_size: int,
    context_length: int,
    mem_fraction_static: float,
    max_new_tokens: int,
    temperature: float,
    disable_cuda_graph: bool,
    cuda_graph_bs: list[int] | None,
    cuda_graph_max_bs: int | None,
    enable_piecewise_cuda_graph: bool,
    piecewise_cuda_graph_tokens: list[int] | None,
    piecewise_cuda_graph_max_tokens: int | None,
    chunked_prefill_size: int | None,
    max_prefill_tokens: int | None,
    attention_backend: str | None,
    prefill_attention_backend: str | None,
    decode_attention_backend: str | None,
    moe_runner_backend: str | None,
    moe_a2a_backend: str | None,
    warmup_count: int,
) -> dict[str, Any]:
    import sglang as sgl
    from sglang.srt.server_args import ServerArgs

    server_arg_names = set(inspect.signature(ServerArgs).parameters)

    engine_kwargs: dict[str, Any] = {
        "model_path": model_path,
        "tp_size": tp_size,
        "context_length": context_length,
        "mem_fraction_static": mem_fraction_static,
        "disable_cuda_graph": disable_cuda_graph,
        "log_level": "warning",
    }
    optional = {
        "cuda_graph_bs": cuda_graph_bs,
        "cuda_graph_max_bs": cuda_graph_max_bs,
        "piecewise_cuda_graph_tokens": piecewise_cuda_graph_tokens,
        "piecewise_cuda_graph_max_tokens": piecewise_cuda_graph_max_tokens,
        "chunked_prefill_size": chunked_prefill_size,
        "max_prefill_tokens": max_prefill_tokens,
        "attention_backend": attention_backend,
        "prefill_attention_backend": prefill_attention_backend,
        "decode_attention_backend": decode_attention_backend,
        "moe_runner_backend": moe_runner_backend,
        "moe_a2a_backend": moe_a2a_backend,
    }
    if "enable_piecewise_cuda_graph" in server_arg_names:
        optional["enable_piecewise_cuda_graph"] = enable_piecewise_cuda_graph
    if "disable_piecewise_cuda_graph" in server_arg_names:
        optional["disable_piecewise_cuda_graph"] = not enable_piecewise_cuda_graph
    engine_kwargs.update({k: v for k, v in optional.items() if v is not None})
    engine_kwargs = {k: v for k, v in engine_kwargs.items() if k in server_arg_names}
    print(f"\n=== {config_name} ===")
    print(json.dumps(engine_kwargs, sort_keys=True))
    t0 = time.monotonic()
    llm = sgl.Engine(**engine_kwargs)
    init_s = time.monotonic() - t0
    sampling_params = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "ignore_eos": True,
    }
    for prompt in prompts[: max(0, warmup_count)]:
        _ = llm.generate(prompt=prompt, sampling_params=sampling_params)
        torch.cuda.synchronize()
    per_req: list[dict[str, Any]] = []
    latencies: list[float] = []
    for i, prompt in enumerate(prompts):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = llm.generate(prompt=prompt, sampling_params=sampling_params)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000.0
        parsed = extract_output(output)
        row = {
            "tok": int(lens[i]),
            "ms": float(ms),
            "output_token_ids": parsed["token_ids"],
            "output_text": parsed["text"],
        }
        per_req.append(row)
        latencies.append(float(ms))
        print(f"  [{i + 1}/{len(prompts)}] len={lens[i]} {ms:.2f} ms")
    if hasattr(llm, "shutdown"):
        llm.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "config": config_name,
        "init_s": float(init_s),
        "avg_ms": float(statistics.mean(latencies)),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "per_req": per_req,
        "engine_kwargs": engine_kwargs,
    }


def add_correctness(results: list[dict[str, Any]]) -> None:
    if not results:
        return
    ref = results[0]["per_req"]
    for result in results:
        flags = []
        for a, b in zip(ref, result["per_req"]):
            flags.append(a.get("output_token_ids") == b.get("output_token_ids"))
        result["same_outputs_vs_first"] = flags
        result["all_same_outputs_vs_first"] = all(flags)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--configs", default="cg",
                        help="comma-separated: eager,cg,piecewise")
    parser.add_argument("--cuda-graph-bs", default=None)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=None)
    parser.add_argument("--piecewise-cuda-graph-tokens", default=None)
    parser.add_argument("--piecewise-cuda-graph-max-tokens", type=int, default=None)
    parser.add_argument("--chunked-prefill-size", type=int, default=None)
    parser.add_argument("--max-prefill-tokens", type=int, default=None)
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--prefill-attention-backend", default=None)
    parser.add_argument("--decode-attention-backend", default=None)
    parser.add_argument("--moe-runner-backend", default=None)
    parser.add_argument("--moe-a2a-backend", default=None)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--warmup-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workload = json.loads(Path(args.workload).read_text(encoding="utf-8"))
    reqs = workload["requests"]
    if args.limit:
        reqs = reqs[: args.limit]
    prompts = [row["prompt"] for row in reqs]
    lens = [int(row["actual_input_length"]) for row in reqs]
    print(
        f"workload={args.workload} n={len(prompts)} "
        f"p50={percentile(lens, 50):.0f} p95={percentile(lens, 95):.0f}"
    )
    requested = {item.strip() for item in args.configs.split(",") if item.strip()}
    results: list[dict[str, Any]] = []
    common = {
        "model_path": args.model,
        "prompts": prompts,
        "lens": lens,
        "tp_size": args.tp_size,
        "context_length": args.context_length,
        "mem_fraction_static": args.mem_fraction_static,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "chunked_prefill_size": args.chunked_prefill_size,
        "max_prefill_tokens": args.max_prefill_tokens,
        "attention_backend": args.attention_backend,
        "prefill_attention_backend": args.prefill_attention_backend,
        "decode_attention_backend": args.decode_attention_backend,
        "moe_runner_backend": args.moe_runner_backend,
        "moe_a2a_backend": args.moe_a2a_backend,
        "warmup_count": 0 if args.no_warmup else args.warmup_count,
    }
    if "eager" in requested:
        results.append(run_config(
            config_name="SGLang eager/no-CG",
            disable_cuda_graph=True,
            cuda_graph_bs=None,
            cuda_graph_max_bs=None,
            enable_piecewise_cuda_graph=False,
            piecewise_cuda_graph_tokens=None,
            piecewise_cuda_graph_max_tokens=None,
            **common,
        ))
    if "cg" in requested:
        results.append(run_config(
            config_name="SGLang default CUDA Graph",
            disable_cuda_graph=False,
            cuda_graph_bs=parse_int_list(args.cuda_graph_bs),
            cuda_graph_max_bs=args.cuda_graph_max_bs,
            enable_piecewise_cuda_graph=False,
            piecewise_cuda_graph_tokens=None,
            piecewise_cuda_graph_max_tokens=None,
            **common,
        ))
    if "piecewise" in requested:
        results.append(run_config(
            config_name="SGLang piecewise CUDA Graph",
            disable_cuda_graph=False,
            cuda_graph_bs=parse_int_list(args.cuda_graph_bs),
            cuda_graph_max_bs=args.cuda_graph_max_bs,
            enable_piecewise_cuda_graph=True,
            piecewise_cuda_graph_tokens=parse_int_list(args.piecewise_cuda_graph_tokens),
            piecewise_cuda_graph_max_tokens=args.piecewise_cuda_graph_max_tokens,
            **common,
        ))
    add_correctness(results)
    first_avg = results[0]["avg_ms"] if results else 0.0
    for result in results:
        result["speedup_vs_first"] = first_avg / result["avg_ms"] if result["avg_ms"] else None
        print(
            f"{result['config']}: avg={result['avg_ms']:.2f} "
            f"p95={result['p95_ms']:.2f} p99={result['p99_ms']:.2f} "
            f"speedup={result['speedup_vs_first']:.2f} "
            f"init={result['init_s']:.1f} correct={result['all_same_outputs_vs_first']}"
        )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "workload": args.workload,
        "model": args.model,
        "lengths": lens,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
