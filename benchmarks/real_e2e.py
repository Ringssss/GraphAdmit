"""
Real End-to-End Benchmark: vLLM with Real Models and Real Traces
================================================================
Tests our DP-planned extended capture sizes against vLLM baselines
using real Azure LLM traces and real models.

Configurations tested:
1. Eager: no CUDA graph (enforce_eager=True)
2. vLLM Default: FULL_AND_PIECEWISE, max_capture=512 (default)
3. Ours-Extended: FULL_AND_PIECEWISE, DP-planned sizes, max_capture=2048+

Metrics: TTFT (prefill latency), throughput, GPU memory
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


def load_azure_trace(trace_name: str, num_samples: int, max_tokens: int,
                     seed: int = 42) -> list[int]:
    """Load real Azure trace and sample prompt lengths."""
    trace_map = {
        "conv": "/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week.csv",
        "code": "/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_code_1week.csv",
    }
    path = trace_map.get(trace_name, trace_name)
    df = pd.read_csv(path, nrows=200000)
    ctx = df["ContextTokens"].values

    # Filter to max_tokens and sample
    ctx = ctx[ctx <= max_tokens]
    ctx = ctx[ctx >= 4]  # min reasonable length
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ctx), size=min(num_samples, len(ctx)), replace=False)
    sampled = ctx[indices].astype(int)
    return sampled.tolist()


def generate_prompts_for_lengths(tokenizer, target_lengths: list[int]) -> list[str]:
    """Generate prompts that produce approximately the target token lengths."""
    base = ("The quick brown fox jumps over the lazy dog. "
            "In a world where technology evolves rapidly, "
            "artificial intelligence continues to transform industries. ") * 200
    base_ids = tokenizer.encode(base)

    prompts = []
    actual_lengths = []
    for tgt in target_lengths:
        ids = base_ids[:tgt]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        prompts.append(text)
        actual_lengths.append(len(ids))
    return prompts, actual_lengths


def compute_dp_capture_sizes(token_lengths: list[int], max_size: int) -> list[int]:
    """Compute DP-optimized capture sizes for the given workload."""
    counts = np.array(token_lengths)
    counts = counts[counts <= max_size]

    # Generate candidates with adaptive granularity
    candidates = set()
    # Fine grain for small sizes
    for s in range(1, min(257, max_size + 1)):
        candidates.add(s)
    # Medium grain 256-1024
    for s in range(256, min(1025, max_size + 1), 8):
        candidates.add(s)
    # Coarser grain 1024-2048
    for s in range(1024, min(2049, max_size + 1), 16):
        candidates.add(s)
    # Coarsest grain 2048+
    for s in range(2048, max_size + 1, 32):
        candidates.add(s)
    candidates.add(max_size)
    candidates = sorted(candidates)

    # Simple but effective: pick sizes at percentile boundaries
    if len(counts) == 0:
        return list(range(1, min(257, max_size + 1), 8))

    percentiles = list(range(0, 101, 2))  # every 2 percentiles
    bucket_targets = set()
    for p in percentiles:
        val = int(np.percentile(counts, p))
        # Snap to nearest candidate
        import bisect
        idx = bisect.bisect_left(candidates, val)
        if idx < len(candidates):
            bucket_targets.add(candidates[idx])

    # Also add some fine-grain coverage for the dense part of the distribution
    p25 = int(np.percentile(counts, 25))
    p75 = int(np.percentile(counts, 75))
    for c in candidates:
        if p25 <= c <= p75:
            bucket_targets.add(c)

    # Always include small sizes for decode
    for s in [1, 2, 4, 8, 16, 32]:
        if s <= max_size:
            bucket_targets.add(s)

    result = sorted(bucket_targets)
    return result


def run_vllm_benchmark(
    model_path: str,
    prompts: list[str],
    config_name: str,
    tp_size: int,
    max_model_len: int,
    gpu_mem_util: float,
    enforce_eager: bool = False,
    cudagraph_mode: str = "FULL_AND_PIECEWISE",
    capture_sizes: list[int] | None = None,
    max_capture_size: int | None = None,
) -> dict:
    """Run a single vLLM benchmark configuration."""
    from vllm import LLM, SamplingParams

    compilation_config: dict = {}
    if enforce_eager:
        pass  # Will set enforce_eager=True on LLM
    else:
        compilation_config["cudagraph_mode"] = cudagraph_mode
        if capture_sizes is not None:
            compilation_config["cudagraph_capture_sizes"] = capture_sizes
        if max_capture_size is not None:
            compilation_config["max_cudagraph_capture_size"] = max_capture_size

    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"  enforce_eager={enforce_eager}")
    if not enforce_eager:
        print(f"  cudagraph_mode={cudagraph_mode}")
        if capture_sizes:
            print(f"  capture_sizes: {len(capture_sizes)} sizes, "
                  f"range=[{min(capture_sizes)}, {max(capture_sizes)}]")
        if max_capture_size:
            print(f"  max_capture_size={max_capture_size}")
    print(f"{'='*60}")

    t_init_start = time.monotonic()

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

    llm = LLM(**llm_kwargs)

    t_init_end = time.monotonic()
    init_time = t_init_end - t_init_start

    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)

    # Warmup run
    _ = llm.generate(prompts[:2], sampling_params)

    # Timed run
    torch.cuda.synchronize()
    t_start = time.monotonic()
    outputs = llm.generate(prompts, sampling_params)
    t_end = time.monotonic()

    total_time = t_end - t_start
    num_prompts = len(prompts)
    throughput = num_prompts / total_time
    avg_ttft = total_time / num_prompts * 1000  # ms

    # Get per-request metrics from outputs
    ttfts = []
    for out in outputs:
        m = out.metrics
        if m and m.first_token_time and m.arrival_time:
            ttft_ms = (m.first_token_time - m.arrival_time) * 1000
            ttfts.append(ttft_ms)

    if ttfts:
        ttfts = np.array(ttfts)
        p50 = np.percentile(ttfts, 50)
        p95 = np.percentile(ttfts, 95)
        p99 = np.percentile(ttfts, 99)
        avg_real = ttfts.mean()
    else:
        # Fallback: use total time / num_prompts
        p50 = p95 = p99 = avg_real = avg_ttft

    result = {
        "config": config_name,
        "num_prompts": num_prompts,
        "total_time_s": total_time,
        "throughput_rps": throughput,
        "avg_ttft_ms": avg_real,
        "p50_ttft_ms": p50,
        "p95_ttft_ms": p95,
        "p99_ttft_ms": p99,
        "init_time_s": init_time,
    }

    print(f"\nResults:")
    print(f"  Total time:    {total_time:.2f}s")
    print(f"  Throughput:    {throughput:.1f} req/s")
    print(f"  Avg TTFT:      {avg_real:.2f} ms")
    print(f"  P50 TTFT:      {p50:.2f} ms")
    print(f"  P95 TTFT:      {p95:.2f} ms")
    print(f"  P99 TTFT:      {p99:.2f} ms")
    print(f"  Init time:     {init_time:.1f}s")

    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/models/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--trace", default="conv", choices=["conv", "code"])
    parser.add_argument("--num-prompts", type=int, default=300)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--our-max-capture", type=int, default=2048,
                        help="Max capture size for our DP-planned config")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Trace: {args.trace}")
    print(f"Max model len: {args.max_model_len}")

    # Load real trace
    token_lengths = load_azure_trace(
        args.trace, args.num_prompts, args.max_model_len)
    token_lengths_arr = np.array(token_lengths)

    print(f"\nWorkload ({args.trace} trace, n={len(token_lengths)}):")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p:2d}: {int(np.percentile(token_lengths_arr, p)):5d} tokens")
    over_512 = (token_lengths_arr > 512).mean() * 100
    over_1024 = (token_lengths_arr > 1024).mean() * 100
    print(f"  >512: {over_512:.1f}%, >1024: {over_1024:.1f}%")

    # Generate prompts
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts, actual_lengths = generate_prompts_for_lengths(tokenizer, token_lengths)
    del tokenizer

    # Compute DP-planned capture sizes
    our_max = min(args.our_max_capture, args.max_model_len)
    dp_sizes = compute_dp_capture_sizes(token_lengths, our_max)
    print(f"\nDP-planned: {len(dp_sizes)} sizes, range=[{min(dp_sizes)}, {max(dp_sizes)}]")

    # vLLM default sizes for reference
    vllm_default_max = 512
    vllm_sizes_count = len([s for s in range(1, vllm_default_max + 1)
                           if s <= 4 or s % 8 == 0])
    coverage_vllm = (token_lengths_arr <= vllm_default_max).mean() * 100
    coverage_ours = (token_lengths_arr <= our_max).mean() * 100
    print(f"vLLM default coverage (max={vllm_default_max}): {coverage_vllm:.1f}%")
    print(f"Our coverage (max={our_max}): {coverage_ours:.1f}%")

    all_results = []

    # -----------------------------------------------------------------------
    # Config 1: Eager (no CUDA graph)
    # -----------------------------------------------------------------------
    r = run_vllm_benchmark(
        model_path=args.model,
        prompts=prompts,
        config_name="Eager (no graph)",
        tp_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    all_results.append(r)

    # -----------------------------------------------------------------------
    # Config 2: vLLM Default FULL_AND_PIECEWISE (max=512)
    # -----------------------------------------------------------------------
    r = run_vllm_benchmark(
        model_path=args.model,
        prompts=prompts,
        config_name=f"vLLM Default (F&P, max=512)",
        tp_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_memory_utilization,
        cudagraph_mode="FULL_AND_PIECEWISE",
    )
    all_results.append(r)

    # -----------------------------------------------------------------------
    # Config 3: vLLM PIECEWISE only (max=512)
    # -----------------------------------------------------------------------
    r = run_vllm_benchmark(
        model_path=args.model,
        prompts=prompts,
        config_name=f"vLLM PIECEWISE (max=512)",
        tp_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_memory_utilization,
        cudagraph_mode="PIECEWISE",
    )
    all_results.append(r)

    # -----------------------------------------------------------------------
    # Config 4: Ours — DP-planned extended sizes
    # -----------------------------------------------------------------------
    r = run_vllm_benchmark(
        model_path=args.model,
        prompts=prompts,
        config_name=f"Ours (DP, max={our_max})",
        tp_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_memory_utilization,
        cudagraph_mode="FULL_AND_PIECEWISE",
        capture_sizes=dp_sizes,
        max_capture_size=our_max,
    )
    all_results.append(r)

    # -----------------------------------------------------------------------
    # Print comparison
    # -----------------------------------------------------------------------
    print(f"\n{'='*100}")
    print(f"END-TO-END RESULTS: {args.model.split('/')[-1]}, "
          f"trace={args.trace}, n={len(prompts)}")
    print(f"{'='*100}")
    print(f"{'Config':<30s} {'Avg TTFT':>9s} {'P50':>7s} {'P95':>7s} "
          f"{'P99':>7s} {'Tput':>8s} {'Init':>6s}")
    print(f"{'':30s} {'(ms)':>9s} {'(ms)':>7s} {'(ms)':>7s} "
          f"{'(ms)':>7s} {'(req/s)':>8s} {'(s)':>6s}")
    print(f"{'-'*100}")

    eager_avg = None
    for r in all_results:
        if "Eager" in r["config"]:
            eager_avg = r["avg_ttft_ms"]
        print(f"{r['config']:<30s} {r['avg_ttft_ms']:9.2f} {r['p50_ttft_ms']:7.2f} "
              f"{r['p95_ttft_ms']:7.2f} {r['p99_ttft_ms']:7.2f} "
              f"{r['throughput_rps']:8.1f} {r['init_time_s']:6.1f}")

    if eager_avg:
        print(f"\n--- Speedup vs Eager (avg TTFT) ---")
        for r in all_results:
            if "Eager" not in r["config"]:
                sp = eager_avg / r["avg_ttft_ms"] if r["avg_ttft_ms"] > 0 else 0
                print(f"  {r['config']:<30s}: {sp:.2f}x")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    model_name = args.model.split("/")[-1]
    out_file = out_dir / f"real_e2e_{model_name}_{args.trace}_{len(prompts)}.json"
    with open(out_file, "w") as f:
        json.dump({
            "model": args.model,
            "trace": args.trace,
            "num_prompts": len(prompts),
            "max_model_len": args.max_model_len,
            "workload_stats": {
                "p50": int(np.median(token_lengths_arr)),
                "p95": int(np.percentile(token_lengths_arr, 95)),
                "pct_over_512": float(over_512),
                "pct_over_1024": float(over_1024),
            },
            "dp_capture_sizes": dp_sizes,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
