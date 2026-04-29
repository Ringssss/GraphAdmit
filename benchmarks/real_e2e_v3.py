"""
Real E2E Benchmark v3: Isolate CUDA Graph benefit
==================================================
Tests with torch.compile disabled to isolate the pure CUDA graph vs eager gap.

Also tests throughput (batch all at once) to see impact under load.
"""

import argparse, gc, json, os, time
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
    idx = rng.choice(len(ctx), size=min(num_samples, len(ctx)), replace=False)
    return ctx[idx].astype(int).tolist()


def compute_dp_sizes(token_lengths, max_size):
    import bisect
    counts = np.array([t for t in token_lengths if t <= max_size])
    candidates = set()
    for s in range(1, min(257, max_size+1)):
        candidates.add(s)
    for s in range(256, min(1025, max_size+1), 8):
        candidates.add(s)
    for s in range(1024, min(2049, max_size+1), 16):
        candidates.add(s)
    for s in range(2048, max_size+1, 32):
        candidates.add(s)
    candidates.add(max_size)
    candidates = sorted(candidates)
    targets = set()
    if len(counts) > 0:
        for p in range(0, 101, 2):
            val = int(np.percentile(counts, p))
            i = bisect.bisect_left(candidates, val)
            if i < len(candidates):
                targets.add(candidates[i])
        p25, p75 = int(np.percentile(counts, 25)), int(np.percentile(counts, 75))
        for c in candidates:
            if p25 <= c <= p75:
                targets.add(c)
    for s in [1, 2, 4, 8, 16, 32]:
        if s <= max_size:
            targets.add(s)
    return sorted(targets)


def run_config(model_path, prompts, config_name, tp_size, max_model_len,
               gpu_mem_util, enforce_eager=False, compilation_config=None):
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=model_path, tensor_parallel_size=tp_size,
        max_model_len=max_model_len, gpu_memory_utilization=gpu_mem_util,
        enforce_eager=enforce_eager, disable_log_stats=True,
    )
    if compilation_config and not enforce_eager:
        llm_kwargs["compilation_config"] = compilation_config

    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"{'='*60}")

    t0 = time.monotonic()
    llm = LLM(**llm_kwargs)
    init_time = time.monotonic() - t0

    sp = SamplingParams(max_tokens=1, temperature=0.0)

    # Warmup
    _ = llm.generate([prompts[0]], sp)

    # Per-request timing
    ttfts = []
    for i, p in enumerate(prompts):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        _ = llm.generate([p], sp)
        torch.cuda.synchronize()
        ttfts.append((time.perf_counter() - t_start) * 1000)
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(prompts)}] avg={np.mean(ttfts):.2f} ms")

    ttfts = np.array(ttfts)

    # Also batch throughput test
    torch.cuda.synchronize()
    t_batch_start = time.perf_counter()
    _ = llm.generate(prompts, sp)
    torch.cuda.synchronize()
    batch_time = time.perf_counter() - t_batch_start
    batch_throughput = len(prompts) / batch_time

    result = {
        "config": config_name,
        "n": len(prompts),
        "avg_ms": float(ttfts.mean()),
        "p50_ms": float(np.percentile(ttfts, 50)),
        "p95_ms": float(np.percentile(ttfts, 95)),
        "p99_ms": float(np.percentile(ttfts, 99)),
        "batch_throughput_rps": float(batch_throughput),
        "batch_time_s": float(batch_time),
        "init_s": float(init_time),
    }

    print(f"  Per-req: avg={ttfts.mean():.2f}, p50={np.percentile(ttfts,50):.2f}, "
          f"p95={np.percentile(ttfts,95):.2f} ms")
    print(f"  Batch throughput: {batch_throughput:.1f} req/s ({batch_time:.2f}s for {len(prompts)})")

    del llm; gc.collect(); torch.cuda.empty_cache(); time.sleep(2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/models/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--trace", default="conv")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--our-max-capture", type=int, default=2048)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    token_lengths = load_azure_trace(args.trace, args.num_prompts, args.max_model_len)
    arr = np.array(token_lengths)
    print(f"Model: {args.model.split('/')[-1]}, trace={args.trace}, n={len(token_lengths)}")
    print(f"Tokens: p50={int(np.median(arr))}, p95={int(np.percentile(arr,95))}, "
          f">512: {(arr>512).mean()*100:.0f}%")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    base = ("The quick brown fox jumps over the lazy dog. "
            "Artificial intelligence transforms industries rapidly. ") * 200
    base_ids = tok.encode(base)
    prompts = [tok.decode(base_ids[:t], skip_special_tokens=True) for t in token_lengths]
    del tok

    dp_sizes = compute_dp_sizes(token_lengths, min(args.our_max_capture, args.max_model_len))
    print(f"DP sizes: {len(dp_sizes)} (max={max(dp_sizes)})")

    results = []

    # 1. Pure Eager (no compile, no graph)
    r = run_config(args.model, prompts, "1. Pure Eager",
                   args.tp_size, args.max_model_len, args.gpu_memory_utilization,
                   enforce_eager=True)
    results.append(r)

    # 2. vLLM Default (compile + FULL_AND_PIECEWISE, max=512)
    r = run_config(args.model, prompts, "2. vLLM Default (compile+F&P, max=512)",
                   args.tp_size, args.max_model_len, args.gpu_memory_utilization,
                   compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"})
    results.append(r)

    # 3. Compile only, no graph
    r = run_config(args.model, prompts, "3. Compile Only (no graph)",
                   args.tp_size, args.max_model_len, args.gpu_memory_utilization,
                   compilation_config={"cudagraph_mode": "NONE"})
    results.append(r)

    # 4. Ours: compile + F&P + DP sizes (max=2048)
    r = run_config(args.model, prompts, f"4. Ours (compile+F&P, DP max={max(dp_sizes)})",
                   args.tp_size, args.max_model_len, args.gpu_memory_utilization,
                   compilation_config={
                       "cudagraph_mode": "FULL_AND_PIECEWISE",
                       "cudagraph_capture_sizes": dp_sizes,
                       "max_cudagraph_capture_size": max(dp_sizes),
                   })
    results.append(r)

    # 5. Ours: compile + PIECEWISE + DP sizes (max=2048) — wider capture
    r = run_config(args.model, prompts, f"5. Ours (compile+PCW, DP max={max(dp_sizes)})",
                   args.tp_size, args.max_model_len, args.gpu_memory_utilization,
                   compilation_config={
                       "cudagraph_mode": "PIECEWISE",
                       "cudagraph_capture_sizes": dp_sizes,
                       "max_cudagraph_capture_size": max(dp_sizes),
                   })
    results.append(r)

    # Print final table
    print(f"\n{'='*100}")
    print(f"FINAL: {args.model.split('/')[-1]}, trace={args.trace}, n={len(prompts)}")
    print(f"{'='*100}")
    print(f"{'Config':<45s} {'Avg':>7s} {'P50':>7s} {'P95':>7s} {'Batch':>8s} {'Init':>5s}")
    print(f"{'':45s} {'(ms)':>7s} {'(ms)':>7s} {'(ms)':>7s} {'(rps)':>8s} {'(s)':>5s}")
    print('-'*100)
    eager_avg = results[0]["avg_ms"]
    for r in results:
        sp = eager_avg / r["avg_ms"] if r["avg_ms"] > 0 else 0
        print(f"{r['config']:<45s} {r['avg_ms']:7.2f} {r['p50_ms']:7.2f} "
              f"{r['p95_ms']:7.2f} {r['batch_throughput_rps']:8.1f} {r['init_s']:5.1f}"
              f"  ({sp:.2f}x)")

    out = Path(args.output_dir); out.mkdir(exist_ok=True)
    fn = out / f"real_v3_{args.model.split('/')[-1]}_{args.trace}_{len(prompts)}.json"
    json.dump(results, open(fn, "w"), indent=2)
    print(f"\nSaved to {fn}")


if __name__ == "__main__":
    main()
