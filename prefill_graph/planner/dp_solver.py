"""
Exact DP Bucket Planner for CUDA Graph Capture Sizes
=====================================================
Given a trace of token counts and resource budgets, finds the optimal
set of bucket sizes minimizing:

    E[padding_waste] + λ_mem * graph_memory(B) + λ_warmup * warmup(B) + λ_fall * fallback_miss(B)

Uses exact 1D dynamic programming: O(k * n) where k = max_buckets, n = unique token counts.

Key advantage over vLLM's fixed [1,2,4,8,16,24,...,512] schedule:
- Trace-driven: adapts to actual workload distribution
- Budget-aware: respects memory and warmup constraints
- Extended range: can allocate buckets beyond 512 where the workload needs them
- Fewer wasted buckets: doesn't pre-capture sizes the workload never hits
"""

import json
import bisect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class BucketPlan:
    """Result of the DP planner."""
    bucket_sizes: list[int]
    expected_hit_rate: float
    expected_padding_waste_pct: float
    expected_total_padding_tokens: int
    expected_fallback_count: int
    total_graph_memory_mb: float
    total_warmup_time_s: float
    cost: float  # total objective value
    assignment: dict[int, int]  # token_count -> assigned bucket


@dataclass
class CostModel:
    """Empirical cost model calibrated from vLLM measurements."""
    # Memory: graph memory grows roughly linearly with bucket size
    # Measured: 0.64 GiB for 51 capture sizes up to 512
    # ≈ 12.5 MB per capture size on average, but larger sizes cost more
    memory_per_token_mb: float = 0.02  # ~10 MB for bucket_size=512
    memory_base_mb: float = 2.0  # fixed overhead per graph

    # Warmup: capture time per bucket
    # Measured: 6 seconds for 102 captures (51 PIECEWISE + 51 FULL)
    # ≈ 60ms per capture
    warmup_per_bucket_s: float = 0.06

    # Compute cost of padding: wasted FLOPs proportional to padded tokens
    # For prefill, each padded token costs ~2 * num_params FLOPs
    # For 8B model: ~16 GFLOPs per token
    # At H100 ~1000 TFLOPS: ~16 μs per padded token
    compute_per_padded_token_us: float = 16.0

    # Fallback cost: eager execution overhead vs graph replay
    # CPU launch overhead per kernel: ~5-10 μs
    # For a 32-layer model with ~15 kernels/layer: ~480 kernels
    # Graph replay saves: 480 * 7 μs = 3.36 ms
    fallback_overhead_us: float = 3360.0

    def graph_memory_mb(self, bucket_size: int) -> float:
        return self.memory_base_mb + self.memory_per_token_mb * bucket_size

    def warmup_time_s(self, bucket_size: int) -> float:
        return self.warmup_per_bucket_s

    def padding_cost(self, actual: int, bucket: int) -> float:
        """Cost of padding from actual to bucket size (in μs equivalent)."""
        return (bucket - actual) * self.compute_per_padded_token_us

    def fallback_cost(self, actual: int) -> float:
        """Cost of running eager fallback (in μs equivalent)."""
        return self.fallback_overhead_us


# ---------------------------------------------------------------------------
# DP Solver
# ---------------------------------------------------------------------------
def solve_bucket_dp(
    token_counts: np.ndarray,
    max_buckets: int,
    candidate_sizes: list[int],
    cost_model: CostModel,
    memory_budget_mb: float = 2048.0,
    warmup_budget_s: float = 30.0,
    lambda_mem: float = 0.1,
    lambda_warmup: float = 100.0,
    lambda_fallback: float = 1.0,
) -> BucketPlan:
    """
    Find optimal bucket set B of size ≤ max_buckets from candidate_sizes
    minimizing total expected cost.

    Args:
        token_counts: array of observed token counts (one per request)
        max_buckets: maximum number of buckets to select
        candidate_sizes: sorted list of candidate bucket sizes
        cost_model: empirical cost model
        memory_budget_mb: max GPU memory for all graphs
        warmup_budget_s: max total capture time
        lambda_mem: weight for memory cost
        lambda_warmup: weight for warmup cost
        lambda_fallback: weight for fallback penalty

    Returns:
        BucketPlan with optimal bucket sizes
    """
    # Build frequency table of unique token counts
    unique, counts = np.unique(token_counts, return_counts=True)
    n_unique = len(unique)
    total_requests = len(token_counts)

    # Ensure candidates are sorted
    candidates = sorted(candidate_sizes)
    n_candidates = len(candidates)

    # Precompute: for each candidate c, what's the total padding cost
    # of assigning all tokens in range (prev_candidate, c] to bucket c
    # We'll compute this during DP transitions

    # DP table: dp[k][j] = min cost using k buckets, where the k-th bucket
    # is candidates[j], covering all tokens ≤ candidates[j]
    INF = float('inf')

    # For efficiency, map each unique token count to the smallest
    # candidate that can cover it
    token_to_candidate_idx = {}
    for i, t in enumerate(unique):
        idx = bisect.bisect_left(candidates, t)
        if idx < n_candidates:
            token_to_candidate_idx[i] = idx
        else:
            token_to_candidate_idx[i] = -1  # no candidate covers this

    # Precompute prefix sums for efficient range queries
    # For range of unique token indices [lo, hi), if assigned to bucket candidates[j]:
    # cost = sum over i in [lo, hi) of counts[i] * (candidates[j] - unique[i])
    # = candidates[j] * sum(counts[lo:hi]) - sum(counts[lo:hi] * unique[lo:hi])

    prefix_count = np.zeros(n_unique + 1, dtype=np.float64)
    prefix_weighted = np.zeros(n_unique + 1, dtype=np.float64)
    for i in range(n_unique):
        prefix_count[i + 1] = prefix_count[i] + counts[i]
        prefix_weighted[i + 1] = prefix_weighted[i] + counts[i] * unique[i]

    def range_padding_cost(lo: int, hi: int, bucket: int) -> float:
        """Total padding cost for tokens unique[lo:hi] assigned to bucket."""
        if lo >= hi:
            return 0.0
        n_req = prefix_count[hi] - prefix_count[lo]
        sum_tokens = prefix_weighted[hi] - prefix_weighted[lo]
        total_padding = bucket * n_req - sum_tokens
        return total_padding * cost_model.compute_per_padded_token_us

    def range_fallback_cost(lo: int, hi: int) -> float:
        """Total fallback cost for tokens unique[lo:hi] not covered by any bucket."""
        if lo >= hi:
            return 0.0
        n_req = prefix_count[hi] - prefix_count[lo]
        return n_req * cost_model.fallback_overhead_us * lambda_fallback

    # Find the range of unique indices each candidate covers
    # unique_idx_for_candidate[j] = first index i where unique[i] <= candidates[j]
    # All tokens with unique index in [0, unique_upper[j]) can fit in candidates[j]
    unique_upper = np.zeros(n_candidates, dtype=int)
    for j in range(n_candidates):
        unique_upper[j] = bisect.bisect_right(unique, candidates[j])

    # DP
    # dp[j] = (cost, memory, warmup) for current number of buckets, last bucket = candidates[j]
    # We iterate over k (number of buckets) and j (index into candidates)

    # Initialize: k=1 (one bucket at candidates[j])
    dp_prev = np.full(n_candidates, INF)
    dp_mem = np.full(n_candidates, INF)
    dp_warmup = np.full(n_candidates, INF)
    dp_parent = np.full((max_buckets, n_candidates), -1, dtype=int)

    for j in range(n_candidates):
        hi = int(unique_upper[j])
        mem = cost_model.graph_memory_mb(candidates[j])
        warmup = cost_model.warmup_time_s(candidates[j])

        if mem > memory_budget_mb or warmup > warmup_budget_s:
            continue

        # Padding cost for all tokens covered by this single bucket
        pad_cost = range_padding_cost(0, hi, candidates[j])
        # Fallback cost for tokens NOT covered (> candidates[j])
        fall_cost = range_fallback_cost(hi, n_unique)
        # Fixed cost of this bucket
        fixed_cost = lambda_mem * mem + lambda_warmup * warmup

        dp_prev[j] = pad_cost + fall_cost + fixed_cost
        dp_mem[j] = mem
        dp_warmup[j] = warmup

    # Iterate k = 2..max_buckets
    for k in range(2, max_buckets + 1):
        dp_curr = np.full(n_candidates, INF)
        dp_mem_curr = np.full(n_candidates, INF)
        dp_warmup_curr = np.full(n_candidates, INF)

        for j in range(k - 1, n_candidates):
            hi_j = int(unique_upper[j])
            mem_j = cost_model.graph_memory_mb(candidates[j])
            warmup_j = cost_model.warmup_time_s(candidates[j])

            # Try each previous bucket candidates[p] where p < j
            for p in range(k - 2, j):
                if dp_prev[p] >= INF:
                    continue

                cum_mem = dp_mem[p] + mem_j
                cum_warmup = dp_warmup[p] + warmup_j
                if cum_mem > memory_budget_mb or cum_warmup > warmup_budget_s:
                    continue

                hi_p = int(unique_upper[p])
                # Tokens in (candidates[p], candidates[j]] assigned to candidates[j]
                pad_cost = range_padding_cost(hi_p, hi_j, candidates[j])
                # Fallback cost for tokens > candidates[j]
                fall_cost = range_fallback_cost(hi_j, n_unique)
                fixed_cost = lambda_mem * mem_j + lambda_warmup * warmup_j

                total = dp_prev[p] - range_fallback_cost(hi_p, n_unique) + pad_cost + fall_cost + fixed_cost

                if total < dp_curr[j]:
                    dp_curr[j] = total
                    dp_mem_curr[j] = cum_mem
                    dp_warmup_curr[j] = cum_warmup
                    dp_parent[k - 1][j] = p

        dp_prev = dp_curr
        dp_mem = dp_mem_curr
        dp_warmup = dp_warmup_curr

    # Find best solution across all k and j
    best_cost = INF
    best_j = -1
    best_k = -1

    # Check all k values (we stored results for each k in dp iterations)
    # Re-run to find best across all k — actually we need to keep track per k
    # Let me fix: store dp per k

    # Simpler approach: just re-run and keep all dp tables
    all_dp = [np.full(n_candidates, INF) for _ in range(max_buckets)]
    all_mem = [np.full(n_candidates, INF) for _ in range(max_buckets)]
    all_warmup = [np.full(n_candidates, INF) for _ in range(max_buckets)]
    all_parent = [[(-1) for _ in range(n_candidates)] for _ in range(max_buckets)]

    # k=0 (1 bucket)
    for j in range(n_candidates):
        hi = int(unique_upper[j])
        mem = cost_model.graph_memory_mb(candidates[j])
        warmup = cost_model.warmup_time_s(candidates[j])
        if mem > memory_budget_mb or warmup > warmup_budget_s:
            continue
        pad_cost = range_padding_cost(0, hi, candidates[j])
        fall_cost = range_fallback_cost(hi, n_unique)
        fixed_cost = lambda_mem * mem + lambda_warmup * warmup
        all_dp[0][j] = pad_cost + fall_cost + fixed_cost
        all_mem[0][j] = mem
        all_warmup[0][j] = warmup

    for k in range(1, max_buckets):
        for j in range(k, n_candidates):
            hi_j = int(unique_upper[j])
            mem_j = cost_model.graph_memory_mb(candidates[j])
            warmup_j = cost_model.warmup_time_s(candidates[j])

            for p in range(k - 1, j):
                if all_dp[k - 1][p] >= INF:
                    continue
                cum_mem = all_mem[k - 1][p] + mem_j
                cum_warmup = all_warmup[k - 1][p] + warmup_j
                if cum_mem > memory_budget_mb or cum_warmup > warmup_budget_s:
                    continue

                hi_p = int(unique_upper[p])
                # Remove old fallback cost from dp[k-1][p], add new segments
                old_fall = range_fallback_cost(hi_p, n_unique)
                new_pad = range_padding_cost(hi_p, hi_j, candidates[j])
                new_fall = range_fallback_cost(hi_j, n_unique)
                fixed = lambda_mem * mem_j + lambda_warmup * warmup_j

                total = all_dp[k - 1][p] - old_fall + new_pad + new_fall + fixed

                if total < all_dp[k][j]:
                    all_dp[k][j] = total
                    all_mem[k][j] = cum_mem
                    all_warmup[k][j] = cum_warmup
                    all_parent[k][j] = p

    # Find global best
    for k in range(max_buckets):
        for j in range(n_candidates):
            if all_dp[k][j] < best_cost:
                best_cost = all_dp[k][j]
                best_j = j
                best_k = k

    if best_j < 0:
        # No valid solution — return empty
        return BucketPlan(
            bucket_sizes=[], expected_hit_rate=0.0,
            expected_padding_waste_pct=0.0, expected_total_padding_tokens=0,
            expected_fallback_count=total_requests,
            total_graph_memory_mb=0.0, total_warmup_time_s=0.0,
            cost=INF, assignment={},
        )

    # Backtrack to find selected buckets
    selected = []
    j = best_j
    for k_idx in range(best_k, -1, -1):
        selected.append(candidates[j])
        j = all_parent[k_idx][j]
    selected.reverse()

    # Compute assignment and metrics
    assignment = {}
    total_padding = 0
    hit_count = 0
    for t in token_counts:
        idx = bisect.bisect_left(selected, t)
        if idx < len(selected):
            bucket = selected[idx]
            assignment[int(t)] = bucket
            total_padding += bucket - t
            hit_count += 1
        # else: fallback

    total_padded = sum(
        assignment.get(int(t), int(t)) for t in token_counts
    )
    waste_pct = total_padding / total_padded * 100 if total_padded > 0 else 0.0

    return BucketPlan(
        bucket_sizes=selected,
        expected_hit_rate=hit_count / total_requests,
        expected_padding_waste_pct=waste_pct,
        expected_total_padding_tokens=total_padding,
        expected_fallback_count=total_requests - hit_count,
        total_graph_memory_mb=float(all_mem[best_k][best_j]),
        total_warmup_time_s=float(all_warmup[best_k][best_j]),
        cost=best_cost,
        assignment=assignment,
    )


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
def generate_candidates(
    max_size: int = 4096,
    fine_grain_up_to: int = 256,
    fine_step: int = 8,
    coarse_step: int = 64,
) -> list[int]:
    """Generate candidate bucket sizes with fine grain for small and coarse for large."""
    candidates = set()
    # Powers of 2
    s = 1
    while s <= max_size:
        candidates.add(s)
        s *= 2
    # Fine grain
    for s in range(fine_step, min(fine_grain_up_to, max_size) + 1, fine_step):
        candidates.add(s)
    # Coarse grain
    for s in range(fine_grain_up_to, max_size + 1, coarse_step):
        candidates.add(s)
    return sorted(candidates)


# ---------------------------------------------------------------------------
# vLLM baseline
# ---------------------------------------------------------------------------
VLLM_DEFAULT_SIZES = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88,
                      96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176,
                      184, 192, 200, 208, 216, 224, 232, 240, 248, 256,
                      272, 288, 304, 320, 336, 352, 368, 384, 400, 416,
                      432, 448, 464, 480, 496, 512]


def evaluate_plan(
    token_counts: np.ndarray,
    bucket_sizes: list[int],
    cost_model: CostModel,
) -> dict:
    """Evaluate a bucket plan against a workload trace."""
    total = len(token_counts)
    hit = 0
    total_padding = 0
    total_actual = 0

    for t in token_counts:
        idx = bisect.bisect_left(bucket_sizes, t)
        if idx < len(bucket_sizes):
            hit += 1
            total_padding += bucket_sizes[idx] - t
            total_actual += t
        else:
            total_actual += t

    total_padded = total_actual + total_padding
    mem = sum(cost_model.graph_memory_mb(s) for s in bucket_sizes)
    warmup = sum(cost_model.warmup_time_s(s) for s in bucket_sizes)

    return {
        "num_buckets": len(bucket_sizes),
        "max_bucket": max(bucket_sizes) if bucket_sizes else 0,
        "hit_rate": hit / total,
        "fallback_rate": 1 - hit / total,
        "padding_waste_pct": total_padding / total_padded * 100 if total_padded > 0 else 0,
        "total_padding_tokens": total_padding,
        "graph_memory_mb": mem,
        "warmup_time_s": warmup,
    }


# ---------------------------------------------------------------------------
# Main: compare DP planner vs vLLM baseline
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", default="traces")
    parser.add_argument("--max-buckets", type=int, default=64)
    parser.add_argument("--max-candidate-size", type=int, default=4096)
    parser.add_argument("--memory-budget-mb", type=float, default=2048.0)
    parser.add_argument("--warmup-budget-s", type=float, default=30.0)
    args = parser.parse_args()

    cost_model = CostModel()
    candidates = generate_candidates(max_size=args.max_candidate_size)
    print(f"Candidate bucket sizes: {len(candidates)} sizes, range [{candidates[0]}, {candidates[-1]}]")

    trace_dir = Path(args.trace_dir)
    trace_files = sorted(trace_dir.glob("exp1_*.json"))

    for tf in trace_files:
        with open(tf) as f:
            data = json.load(f)

        dist = data["config"]["distribution"]
        token_counts = np.array(data["token_distribution"]["actual_lens"])
        print(f"\n{'='*70}")
        print(f"Distribution: {dist} (n={len(token_counts)})")
        print(f"Token stats: min={token_counts.min()}, median={int(np.median(token_counts))}, "
              f"max={token_counts.max()}, p95={int(np.percentile(token_counts, 95))}")

        # Evaluate vLLM baseline
        vllm_eval = evaluate_plan(token_counts, VLLM_DEFAULT_SIZES, cost_model)
        print(f"\n--- vLLM Baseline (51 fixed sizes, max=512) ---")
        print(f"  Hit rate:       {vllm_eval['hit_rate']*100:.1f}%")
        print(f"  Fallback rate:  {vllm_eval['fallback_rate']*100:.1f}%")
        print(f"  Padding waste:  {vllm_eval['padding_waste_pct']:.1f}%")
        print(f"  Graph memory:   {vllm_eval['graph_memory_mb']:.0f} MB")
        print(f"  Warmup time:    {vllm_eval['warmup_time_s']:.1f}s")

        # Run DP planner
        print(f"\n  Running DP planner (max_buckets={args.max_buckets})...", end=" ")
        import time
        t0 = time.monotonic()
        plan = solve_bucket_dp(
            token_counts=token_counts,
            max_buckets=args.max_buckets,
            candidate_sizes=candidates,
            cost_model=cost_model,
            memory_budget_mb=args.memory_budget_mb,
            warmup_budget_s=args.warmup_budget_s,
        )
        t1 = time.monotonic()
        print(f"done in {t1-t0:.2f}s")

        dp_eval = evaluate_plan(token_counts, plan.bucket_sizes, cost_model)
        print(f"\n--- DP Planner (k={len(plan.bucket_sizes)} buckets, max={max(plan.bucket_sizes) if plan.bucket_sizes else 0}) ---")
        print(f"  Bucket sizes:   {plan.bucket_sizes}")
        print(f"  Hit rate:       {dp_eval['hit_rate']*100:.1f}%")
        print(f"  Fallback rate:  {dp_eval['fallback_rate']*100:.1f}%")
        print(f"  Padding waste:  {dp_eval['padding_waste_pct']:.1f}%")
        print(f"  Graph memory:   {dp_eval['graph_memory_mb']:.0f} MB")
        print(f"  Warmup time:    {dp_eval['warmup_time_s']:.1f}s")

        # Compare
        print(f"\n--- Improvement over vLLM Baseline ---")
        hit_delta = dp_eval['hit_rate'] - vllm_eval['hit_rate']
        mem_delta = dp_eval['graph_memory_mb'] - vllm_eval['graph_memory_mb']
        warmup_delta = dp_eval['warmup_time_s'] - vllm_eval['warmup_time_s']
        print(f"  Hit rate:     {'+' if hit_delta>=0 else ''}{hit_delta*100:.1f}pp")
        print(f"  Graph memory: {'+' if mem_delta>=0 else ''}{mem_delta:.0f} MB")
        print(f"  Warmup time:  {'+' if warmup_delta>=0 else ''}{warmup_delta:.1f}s")
        print(f"  Fewer buckets: {len(VLLM_DEFAULT_SIZES)} → {len(plan.bucket_sizes)} "
              f"({len(VLLM_DEFAULT_SIZES) - len(plan.bucket_sizes)} fewer)")


if __name__ == "__main__":
    main()
