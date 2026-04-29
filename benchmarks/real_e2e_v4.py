"""
Real E2E Benchmark v4: Chunked Prefill DISABLED
================================================
With CP disabled, long prefill runs as a single iteration.
This is where extended CUDA graph capture sizes should show clear benefit.

Configs:
1. Eager (no graph, no CP)
2. vLLM Default graph (max=512, no CP) — long prefills fallback to eager
3. Ours: DP-planned extended sizes (max=2048, no CP) — long prefills hit graph
4. vLLM Default graph (max=512, WITH CP) — baseline with chunking
"""

import argparse, gc, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, torch

from prefill_graph.planner.dp_solver import (
    CostModel,
    VLLM_DEFAULT_SIZES,
    generate_candidates,
    solve_bucket_dp,
)


def load_trace(name, n, max_tok, seed=42):
    paths = {
        "conv": "/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week.csv",
        "code": "/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_code_1week.csv",
    }
    df = pd.read_csv(paths[name], nrows=200000)
    ctx = df["ContextTokens"].values
    ctx = ctx[(ctx <= max_tok) & (ctx >= 4)]
    rng = np.random.RandomState(seed)
    return ctx[rng.choice(len(ctx), min(n, len(ctx)), replace=False)].astype(int).tolist()


def heuristic_sizes(lengths, max_s):
    import bisect
    arr = np.array([t for t in lengths if t <= max_s])
    cands = sorted(set(
        list(range(1, min(257, max_s+1))) +
        list(range(256, min(1025, max_s+1), 8)) +
        list(range(1024, min(2049, max_s+1), 16)) +
        list(range(2048, max_s+1, 32)) +
        [max_s]
    ))
    tgts = set()
    if len(arr) > 0:
        for p in range(0, 101, 2):
            v = int(np.percentile(arr, p))
            i = bisect.bisect_left(cands, v)
            if i < len(cands): tgts.add(cands[i])
        lo, hi = int(np.percentile(arr, 25)), int(np.percentile(arr, 75))
        for c in cands:
            if lo <= c <= hi: tgts.add(c)
    for s in [1,2,4,8,16,32]:
        if s <= max_s: tgts.add(s)
    return sorted(tgts)


def exact_dp_sizes(lengths, max_s, max_buckets, memory_budget_mb,
                   warmup_budget_s, lambda_mem, lambda_warmup,
                   lambda_fallback):
    arr = np.array(lengths, dtype=np.int64)
    candidates = generate_candidates(
        max_size=max_s,
        fine_grain_up_to=min(256, max_s),
        fine_step=8,
        coarse_step=64,
    )
    candidates = sorted(set(candidates) | {s for s in VLLM_DEFAULT_SIZES if s <= max_s} | {max_s})
    plan = solve_bucket_dp(
        token_counts=arr,
        max_buckets=max_buckets,
        candidate_sizes=candidates,
        cost_model=CostModel(),
        memory_budget_mb=memory_budget_mb,
        warmup_budget_s=warmup_budget_s,
        lambda_mem=lambda_mem,
        lambda_warmup=lambda_warmup,
        lambda_fallback=lambda_fallback,
    )
    return plan.bucket_sizes, plan


def run(model, prompts, lens_arr, name, tp, mml, gmu,
        eager=False, cg_mode="FULL_AND_PIECEWISE",
        cap_sizes=None, max_cap=None, chunked=False, trace_file=None):
    from vllm import LLM, SamplingParams

    old_trace_file = os.environ.get("VLLM_CG_TRACE_FILE")
    if trace_file:
        os.environ["VLLM_CG_TRACE_FILE"] = trace_file
        Path(trace_file).parent.mkdir(exist_ok=True, parents=True)
        Path(trace_file).write_text("")

    cc = {}
    if not eager:
        cc["cudagraph_mode"] = cg_mode
        if cap_sizes: cc["cudagraph_capture_sizes"] = cap_sizes
        if max_cap:   cc["max_cudagraph_capture_size"] = max_cap

    kw = dict(model=model, tensor_parallel_size=tp, max_model_len=mml,
              gpu_memory_utilization=gmu, enforce_eager=eager,
              disable_log_stats=True, enable_chunked_prefill=chunked)
    if not chunked:
        kw["max_num_batched_tokens"] = mml
    if not eager:
        kw["compilation_config"] = cc

    print(f"\n{'='*60}")
    print(f" {name}")
    print(f" chunked_prefill={chunked}, eager={eager}")
    if cap_sizes: print(f" capture_sizes: {len(cap_sizes)}, max={max(cap_sizes)}")
    print(f"{'='*60}")

    t0 = time.monotonic()
    try:
        llm = LLM(**kw)
        init_s = time.monotonic() - t0

        sp = SamplingParams(max_tokens=1, temperature=0.0)
        _ = llm.generate([prompts[0]], sp)  # warmup

        ttfts = []
        for i, p in enumerate(prompts):
            torch.cuda.synchronize()
            ts = time.perf_counter()
            _ = llm.generate([p], sp)
            torch.cuda.synchronize()
            ttfts.append((time.perf_counter() - ts) * 1000)
            if (i+1) % 25 == 0:
                print(f"  [{i+1}/{len(prompts)}] avg={np.mean(ttfts):.2f} ms")

        del llm
    finally:
        if trace_file:
            if old_trace_file is None:
                os.environ.pop("VLLM_CG_TRACE_FILE", None)
            else:
                os.environ["VLLM_CG_TRACE_FILE"] = old_trace_file

    ttfts = np.array(ttfts)
    lens = np.array(lens_arr)

    # Per-range breakdown
    print(f"\n  {'Range':>15s}  {'N':>4s}  {'Avg':>8s}  {'P50':>8s}  {'P95':>8s}")
    for lo, hi in [(0,256),(256,512),(512,1024),(1024,2048),(2048,4096)]:
        m = (lens >= lo) & (lens < hi)
        if m.sum() > 0:
            s = ttfts[m]
            print(f"  {f'[{lo},{hi})':>15s}  {m.sum():4d}  {s.mean():8.2f}  "
                  f"{np.percentile(s,50):8.2f}  {np.percentile(s,95):8.2f}")

    res = dict(config=name, n=len(prompts), init_s=init_s, chunked=chunked,
               avg=float(ttfts.mean()), p50=float(np.percentile(ttfts,50)),
               p95=float(np.percentile(ttfts,95)), p99=float(np.percentile(ttfts,99)),
               per_req=[dict(tok=int(l), ms=float(t)) for l,t in zip(lens, ttfts)])

    print(f"  Overall: avg={ttfts.mean():.2f}, p50={np.percentile(ttfts,50):.2f}, "
          f"p95={np.percentile(ttfts,95):.2f}")

    gc.collect(); torch.cuda.empty_cache(); time.sleep(2)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/models/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--trace", default="conv")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--our-max", type=int, default=2048)
    parser.add_argument("--planner", choices=["exact", "heuristic"], default="exact")
    parser.add_argument("--max-buckets", type=int, default=64)
    parser.add_argument("--memory-budget-mb", type=float, default=2048.0)
    parser.add_argument("--warmup-budget-s", type=float, default=30.0)
    parser.add_argument("--lambda-mem", type=float, default=0.1)
    parser.add_argument("--lambda-warmup", type=float, default=100.0)
    parser.add_argument("--lambda-fallback", type=float, default=1.0)
    parser.add_argument("--trace-cudagraph", action="store_true")
    args = parser.parse_args()

    toks = load_trace(args.trace, args.num_prompts, args.max_model_len)
    arr = np.array(toks)
    print(f"Model: {args.model.split('/')[-1]}, trace={args.trace}, n={len(toks)}")
    for p in [25,50,75,90,95]:
        print(f"  p{p}: {int(np.percentile(arr,p))} tokens")
    print(f"  >512: {(arr>512).mean()*100:.0f}%, >1024: {(arr>1024).mean()*100:.0f}%")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    base = ("The quick brown fox jumps over the lazy dog. "
            "In a world of artificial intelligence, technology transforms everything. ") * 300
    bids = tok.encode(base)
    prompts = [tok.decode(bids[:t], skip_special_tokens=True) for t in toks]
    actual = [min(len(bids[:t]), t) for t in toks]
    del tok

    plan = None
    if args.planner == "exact":
        our_sizes, plan = exact_dp_sizes(
            toks, min(args.our_max, args.max_model_len), args.max_buckets,
            args.memory_budget_mb, args.warmup_budget_s, args.lambda_mem,
            args.lambda_warmup, args.lambda_fallback)
        print(f"Exact DP sizes: {len(our_sizes)}, max={max(our_sizes) if our_sizes else 0}")
        print(f"  hit={plan.expected_hit_rate*100:.1f}%, waste={plan.expected_padding_waste_pct:.2f}%, "
              f"fallback={plan.expected_fallback_count}, mem={plan.total_graph_memory_mb:.1f}MB, "
              f"warmup={plan.total_warmup_time_s:.1f}s")
    else:
        our_sizes = heuristic_sizes(toks, min(args.our_max, args.max_model_len))
        print(f"Heuristic sizes: {len(our_sizes)}, max={max(our_sizes)}")

    results = []
    M = args.model

    # 1. Eager, no CP
    trace_prefix = f"traces/v4_{args.model.split('/')[-1]}_{args.trace}_{len(prompts)}"

    results.append(run(M, prompts, actual, "1. Eager (no CP)", args.tp_size,
                       args.max_model_len, args.gpu_memory_utilization,
                       eager=True, chunked=False))

    # 2. vLLM default graph, no CP (max=512)
    results.append(run(M, prompts, actual, "2. vLLM graph (max=512, no CP)", args.tp_size,
                       args.max_model_len, args.gpu_memory_utilization,
                       cg_mode="FULL_AND_PIECEWISE", chunked=False,
                       trace_file=f"{trace_prefix}_vllm_no_cp.jsonl" if args.trace_cudagraph else None))

    # 3. Ours: extended graph, no CP (max=our_max)
    results.append(run(M, prompts, actual, f"3. Ours ({args.planner}, max={args.our_max}, no CP)", args.tp_size,
                       args.max_model_len, args.gpu_memory_utilization,
                       cg_mode="FULL_AND_PIECEWISE",
                       cap_sizes=our_sizes, max_cap=max(our_sizes), chunked=False,
                       trace_file=f"{trace_prefix}_ours_no_cp.jsonl" if args.trace_cudagraph else None))

    # 4. vLLM default WITH chunked prefill (production baseline)
    results.append(run(M, prompts, actual, "4. vLLM graph (max=512, WITH CP)", args.tp_size,
                       args.max_model_len, args.gpu_memory_utilization,
                       cg_mode="FULL_AND_PIECEWISE", chunked=True,
                       trace_file=f"{trace_prefix}_vllm_cp.jsonl" if args.trace_cudagraph else None))

    # Final table
    print(f"\n{'='*95}")
    print(f"FINAL: {args.model.split('/')[-1]}, trace={args.trace}, n={len(prompts)}")
    print(f"{'='*95}")
    print(f"{'Config':<40s} {'Avg':>7s} {'P50':>7s} {'P95':>7s} {'P99':>7s} {'Init':>5s} {'vs E':>6s}")
    print('-'*95)
    ea = results[0]["avg"]
    for r in results:
        sp = ea / r["avg"] if r["avg"] > 0 else 0
        print(f"{r['config']:<40s} {r['avg']:7.2f} {r['p50']:7.2f} "
              f"{r['p95']:7.2f} {r['p99']:7.2f} {r['init_s']:5.0f} {sp:6.2f}x")

    Path("results").mkdir(exist_ok=True)
    fn = f"results/v4_{args.model.split('/')[-1]}_{args.trace}_{len(prompts)}.json"
    json.dump(results, open(fn, "w"), indent=2)
    print(f"\nSaved to {fn}")
    if plan is not None:
        plan_fn = f"results/v4_plan_{args.model.split('/')[-1]}_{args.trace}_{len(prompts)}.json"
        json.dump(dict(
            planner=args.planner,
            bucket_sizes=[int(x) for x in plan.bucket_sizes],
            expected_hit_rate=plan.expected_hit_rate,
            expected_padding_waste_pct=plan.expected_padding_waste_pct,
            expected_total_padding_tokens=int(plan.expected_total_padding_tokens),
            expected_fallback_count=int(plan.expected_fallback_count),
            total_graph_memory_mb=plan.total_graph_memory_mb,
            total_warmup_time_s=plan.total_warmup_time_s,
            cost=plan.cost,
        ), open(plan_fn, "w"), indent=2)
        print(f"Saved plan to {plan_fn}")


if __name__ == "__main__":
    main()
