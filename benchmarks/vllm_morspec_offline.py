#!/usr/bin/env python3
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.morspec_loader import load_morspec_prompts
from prefill_graph.planner.dp_solver import CostModel, VLLM_DEFAULT_SIZES, generate_candidates, solve_bucket_dp


def exact_dp_sizes(lengths, max_s, max_buckets=64):
    arr = np.array(lengths, dtype=np.int64)
    candidates = generate_candidates(max_size=max_s, fine_grain_up_to=min(256, max_s), fine_step=8, coarse_step=64)
    candidates = sorted(set(candidates) | {s for s in VLLM_DEFAULT_SIZES if s <= max_s} | {max_s})
    plan = solve_bucket_dp(
        token_counts=arr,
        max_buckets=max_buckets,
        candidate_sizes=candidates,
        cost_model=CostModel(),
        memory_budget_mb=2048.0,
        warmup_budget_s=30.0,
        lambda_mem=0.1,
        lambda_warmup=100.0,
        lambda_fallback=1.0,
    )
    return plan.bucket_sizes, plan


def run_config(model, prompts, lens, name, tp, max_model_len, gmu, eager=False, chunked=False, cap_sizes=None, max_cap=None, graph_mode='FULL_AND_PIECEWISE', batch_mode=False):
    from vllm import LLM, SamplingParams

    cc = {}
    if not eager:
        cc['cudagraph_mode'] = graph_mode
        if cap_sizes:
            cc['cudagraph_capture_sizes'] = cap_sizes
        if max_cap:
            cc['max_cudagraph_capture_size'] = max_cap

    kw = dict(
        model=model,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        gpu_memory_utilization=gmu,
        enforce_eager=eager,
        disable_log_stats=True,
        enable_chunked_prefill=chunked,
    )
    if not chunked:
        kw['max_num_batched_tokens'] = max_model_len
    if not eager:
        kw['compilation_config'] = cc

    print(f"\n=== {name} === chunked={chunked} eager={eager} batch={batch_mode}")
    if cap_sizes:
        print(f'capture sizes={len(cap_sizes)} max={max(cap_sizes)}')
    t0 = time.monotonic()
    llm = LLM(**kw)
    init_s = time.monotonic() - t0
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    _ = llm.generate([prompts[0]], sp)

    ttfts = []
    if batch_mode:
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = llm.generate(prompts, sp)
        torch.cuda.synchronize()
        total_s = time.perf_counter() - start
        ttfts = [total_s * 1000 / len(prompts)] * len(prompts)
    else:
        for i, prompt in enumerate(prompts):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = llm.generate([prompt], sp)
            torch.cuda.synchronize()
            ttfts.append((time.perf_counter() - start) * 1000)
            print(f"  [{i+1}/{len(prompts)}] {ttfts[-1]:.2f} ms")
    del llm
    gc.collect(); torch.cuda.empty_cache(); time.sleep(2)

    arr = np.array(ttfts)
    return {
        'config': name,
        'n': len(prompts),
        'init_s': init_s,
        'chunked': chunked,
        'batch_mode': batch_mode,
        'avg_ms': float(arr.mean()),
        'p50_ms': float(np.percentile(arr, 50)),
        'p95_ms': float(np.percentile(arr, 95)),
        'p99_ms': float(np.percentile(arr, 99)),
        'per_req': [{'tok': int(l), 'ms': float(t)} for l, t in zip(lens, ttfts)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/mnt/models/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--dataset', default='/home/zhujianian/morspec/data/gsm8k.jsonl')
    parser.add_argument('--num-samples', type=int, default=16)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--tp-size', type=int, default=1)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--our-max', type=int, default=2048)
    parser.add_argument('--max-buckets', type=int, default=64)
    parser.add_argument('--batch-mode', action='store_true')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    raw_prompts, _ = load_morspec_prompts(args.dataset, args.num_samples, args.offset)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    lens = []
    for prompt in raw_prompts:
        ids = tokenizer(prompt)['input_ids']
        if len(ids) >= args.max_model_len:
            ids = ids[:args.max_model_len - 1]
            prompt = tokenizer.decode(ids, skip_special_tokens=True)
        prompts.append(prompt)
        lens.append(len(ids))
    print(f'dataset={args.dataset} n={len(prompts)} lens p50={np.percentile(lens,50):.0f} p95={np.percentile(lens,95):.0f}')
    del tokenizer

    our_sizes, plan = exact_dp_sizes(lens, min(args.our_max, args.max_model_len), args.max_buckets)
    print(f'ours sizes={len(our_sizes)} max={max(our_sizes)} hit={plan.expected_hit_rate*100:.1f}% waste={plan.expected_padding_waste_pct:.2f}% fallback={plan.expected_fallback_count}')

    results = []
    results.append(run_config(args.model, prompts, lens, '1. Eager no-CP', args.tp_size, args.max_model_len, args.gpu_memory_utilization, eager=True, chunked=False, batch_mode=args.batch_mode))
    results.append(run_config(args.model, prompts, lens, '2. vLLM graph max512 no-CP', args.tp_size, args.max_model_len, args.gpu_memory_utilization, chunked=False, batch_mode=args.batch_mode))
    results.append(run_config(args.model, prompts, lens, f'3. Ours DP max{args.our_max} no-CP', args.tp_size, args.max_model_len, args.gpu_memory_utilization, chunked=False, cap_sizes=our_sizes, max_cap=max(our_sizes), batch_mode=args.batch_mode))
    results.append(run_config(args.model, prompts, lens, '4. vLLM graph max512 CP', args.tp_size, args.max_model_len, args.gpu_memory_utilization, chunked=True, batch_mode=args.batch_mode))

    eager_avg = results[0]['avg_ms']
    for r in results:
        r['speedup_vs_eager'] = eager_avg / r['avg_ms'] if r['avg_ms'] > 0 else None
        print(f"{r['config']}: avg={r['avg_ms']:.2f} p95={r['p95_ms']:.2f} p99={r['p99_ms']:.2f} speedup={r['speedup_vs_eager']:.2f}x")

    result = {
        'model': args.model,
        'dataset': args.dataset,
        'num_samples': len(prompts),
        'lens': lens,
        'planner': {
            'bucket_sizes': [int(x) for x in our_sizes],
            'expected_hit_rate': plan.expected_hit_rate,
            'expected_padding_waste_pct': plan.expected_padding_waste_pct,
            'expected_fallback_count': int(plan.expected_fallback_count),
            'total_graph_memory_mb': plan.total_graph_memory_mb,
            'total_warmup_time_s': plan.total_warmup_time_s,
        },
        'results': results,
    }
    out = Path(args.output) if args.output else Path('results') / f"vllm_morspec_{Path(args.dataset).stem}_{len(prompts)}{'_batch' if args.batch_mode else '_offline'}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
