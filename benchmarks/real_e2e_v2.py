"""
Real End-to-End Benchmark v2: Per-Request Prefill Latency
=========================================================
Sends requests ONE BY ONE to isolate per-request prefill latency.
This measures the real CUDA graph benefit for each request individually.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_azure_trace(trace_name, num_samples, max_tokens, seed=42):
    trace_map = {
        "conv": "/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week.csv",
        "code": "/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_code_1week.csv",
    }
    df = pd.read_csv(trace_map.get(trace_name, trace_name), nrows=200000)
    ctx = df["ContextTokens"].values
    ctx = ctx[(ctx <= max_tokens) & (ctx >= 4)]
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ctx), size=min(num_samples, len(ctx)), replace=False)
    return ctx[indices].astype(int).tolist()


def compute_dp_capture_sizes(token_lengths, max_size):
    import bisect
    counts = np.array(token_lengths)
    counts = counts[counts <= max_size]
    candidates = set()
    for s in range(1, min(257, max_size + 1)):
        candidates.add(s)
    for s in range(256, min(1025, max_size + 1), 8):
        candidates.add(s)
    for s in range(1024, min(2049, max_size + 1), 16):
        candidates.add(s)
    for s in range(2048, max_size + 1, 32):
        candidates.add(s)
    candidates.add(max_size)
    candidates = sorted(candidates)
    percentiles = list(range(0, 101, 2))
    bucket_targets = set()
    for p in percentiles:
        if len(counts) == 0:
            break
        val = int(np.percentile(counts, p))
        idx = bisect.bisect_left(candidates, val)
        if idx < len(candidates):
            bucket_targets.add(candidates[idx])
    p25 = int(np.percentile(counts, 25)) if len(counts) > 0 else 0
    p75 = int(np.percentile(counts, 75)) if len(counts) > 0 else max_size
    for c in candidates:
        if p25 <= c <= p75:
            bucket_targets.add(c)
    for s in [1, 2, 4, 8, 16, 32]:
        if s <= max_size:
            bucket_targets.add(s)
    return sorted(bucket_targets)


def run_single_config(
    model_path, prompts, actual_lens, config_name,
    tp_size, max_model_len, gpu_mem_util,
    enforce_eager=False, cudagraph_mode="FULL_AND_PIECEWISE",
    capture_sizes=None, max_capture_size=None,
):
    """Run per-request benchmarks sending ONE prompt at a time."""
    from vllm import LLM, SamplingParams

    compilation_config = {}
    if not enforce_eager:
        compilation_config["cudagraph_mode"] = cudagraph_mode
        if capture_sizes is not None:
            compilation_config["cudagraph_capture_sizes"] = capture_sizes
        if max_capture_size is not None:
            compilation_config["max_cudagraph_capture_size"] = max_capture_size

    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    if not enforce_eager and capture_sizes:
        print(f"  {len(capture_sizes)} capture sizes, max={max(capture_sizes)}")
    print(f"{'='*60}")

    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=enforce_eager,
        disable_log_stats=True,
    )
    if not enforce_eager:
        llm_kwargs["compilation_config"] = compilation_config

    t0 = time.monotonic()
    llm = LLM(**llm_kwargs)
    init_time = time.monotonic() - t0

    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)

    # Warmup
    _ = llm.generate([prompts[0]], sampling_params)

    # Per-request timing: send each prompt individually
    ttfts = []
    for i, prompt in enumerate(prompts):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        out = llm.generate([prompt], sampling_params)
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        ttfts.append((t_end - t_start) * 1000)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(prompts)}] avg so far: {np.mean(ttfts):.2f} ms")

    ttfts = np.array(ttfts)
    lens = np.array(actual_lens)

    result = {
        "config": config_name,
        "num_prompts": len(prompts),
        "init_time_s": init_time,
        "avg_ttft_ms": float(ttfts.mean()),
        "p50_ttft_ms": float(np.percentile(ttfts, 50)),
        "p95_ttft_ms": float(np.percentile(ttfts, 95)),
        "p99_ttft_ms": float(np.percentile(ttfts, 99)),
        "min_ttft_ms": float(ttfts.min()),
        "max_ttft_ms": float(ttfts.max()),
        "per_request": [
            {"tokens": int(l), "ttft_ms": float(t)}
            for l, t in zip(lens, ttfts)
        ],
    }

    # Breakdown by token range
    print(f"\n  Results for {config_name}:")
    print(f"  {'Range':>15s}  {'Count':>5s}  {'Avg ms':>8s}  {'P50 ms':>8s}  {'P95 ms':>8s}")
    ranges = [(0, 128), (128, 512), (512, 1024), (1024, 2048), (2048, 4096)]
    for lo, hi in ranges:
        mask = (lens >= lo) & (lens < hi)
        if mask.sum() > 0:
            sub = ttfts[mask]
            print(f"  {f'[{lo},{hi})':>15s}  {mask.sum():5d}  {sub.mean():8.2f}  "
                  f"{np.percentile(sub, 50):8.2f}  {np.percentile(sub, 95):8.2f}")

    print(f"\n  Overall: avg={ttfts.mean():.2f} ms, p50={np.percentile(ttfts, 50):.2f}, "
          f"p95={np.percentile(ttfts, 95):.2f}, p99={np.percentile(ttfts, 99):.2f}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/models/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--trace", default="conv", choices=["conv", "code"])
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--our-max-capture", type=int, default=2048)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    # Load trace
    token_lengths = load_azure_trace(args.trace, args.num_prompts, args.max_model_len)
    token_lengths_arr = np.array(token_lengths)
    print(f"Model: {args.model.split('/')[-1]}")
    print(f"Trace: {args.trace} (n={len(token_lengths)})")
    for p in [25, 50, 75, 90, 95]:
        print(f"  p{p}: {int(np.percentile(token_lengths_arr, p))} tokens")
    print(f"  >512: {(token_lengths_arr > 512).mean()*100:.1f}%")

    # Generate prompts
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = ("The quick brown fox jumps over the lazy dog. "
            "Artificial intelligence transforms industries rapidly. ") * 200
    base_ids = tokenizer.encode(base)
    prompts = []
    actual_lens = []
    for tgt in token_lengths:
        ids = base_ids[:tgt]
        prompts.append(tokenizer.decode(ids, skip_special_tokens=True))
        actual_lens.append(len(ids))
    del tokenizer

    # DP-planned sizes
    our_max = min(args.our_max_capture, args.max_model_len)
    dp_sizes = compute_dp_capture_sizes(token_lengths, our_max)
    print(f"\nDP-planned: {len(dp_sizes)} sizes, max={max(dp_sizes)}")

    all_results = []

    # 1. Eager
    r = run_single_config(
        args.model, prompts, actual_lens, "Eager",
        args.tp_size, args.max_model_len, args.gpu_memory_utilization,
        enforce_eager=True)
    all_results.append(r)

    # 2. vLLM Default FULL_AND_PIECEWISE (max=512)
    r = run_single_config(
        args.model, prompts, actual_lens, "vLLM F&P (max=512)",
        args.tp_size, args.max_model_len, args.gpu_memory_utilization,
        cudagraph_mode="FULL_AND_PIECEWISE")
    all_results.append(r)

    # 3. Ours: DP + extended
    r = run_single_config(
        args.model, prompts, actual_lens, f"Ours (DP, max={our_max})",
        args.tp_size, args.max_model_len, args.gpu_memory_utilization,
        cudagraph_mode="FULL_AND_PIECEWISE",
        capture_sizes=dp_sizes, max_capture_size=our_max)
    all_results.append(r)

    # Final comparison
    print(f"\n{'='*90}")
    print(f"FINAL COMPARISON: {args.model.split('/')[-1]}, trace={args.trace}")
    print(f"{'='*90}")
    print(f"{'Config':<25s} {'Avg':>8s} {'P50':>8s} {'P95':>8s} {'P99':>8s} {'Init':>6s}")
    print(f"{'':25s} {'(ms)':>8s} {'(ms)':>8s} {'(ms)':>8s} {'(ms)':>8s} {'(s)':>6s}")
    print(f"{'-'*90}")
    eager_avg = all_results[0]["avg_ttft_ms"]
    for r in all_results:
        sp = eager_avg / r["avg_ttft_ms"] if r["avg_ttft_ms"] > 0 else 0
        print(f"{r['config']:<25s} {r['avg_ttft_ms']:8.2f} {r['p50_ttft_ms']:8.2f} "
              f"{r['p95_ttft_ms']:8.2f} {r['p99_ttft_ms']:8.2f} {r['init_time_s']:6.1f}"
              f"  ({sp:.2f}x vs eager)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    model_name = args.model.split("/")[-1]
    out_file = out_dir / f"real_e2e_v2_{model_name}_{args.trace}_{len(prompts)}.json"
    with open(out_file, "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
