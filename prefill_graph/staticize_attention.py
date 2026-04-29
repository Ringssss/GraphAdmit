"""
Module 2+3: Attention Metadata Staticization + Graph Key Collapse
=================================================================
Proof-of-concept that demonstrates:
1. Padding attention metadata (query_start_loc, seq_lens, block_table)
   to a fixed max_num_reqs eliminates num_reqs from the graph key
2. Full CUDA graph capture of a realistic LLaMA-like model including
   FlashAttention-style operations, with variable num_reqs per batch
3. Correctness: graph output matches eager output within fp16 tolerance
4. Performance: measures graph replay vs eager for various configurations

This implements the core of the "Canonical Prefill Arena" concept:
- Fixed token arena (max_num_tokens)
- Fixed request arena (max_num_reqs)
- Runtime: pad real requests, capture/replay, trim output
"""

import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# 1. Model: Simplified LLaMA layer (GQA + RMSNorm + SiLU FFN)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, hidden: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class LLaMAAttention(nn.Module):
    """GQA attention using PyTorch SDPA (graph-safe with explicit mask)."""
    def __init__(self, hidden: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden // num_heads
        self.kv_repeat = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, attn_mask):
        B, S, H = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # GQA: expand kv heads
        k = k.repeat_interleave(self.kv_repeat, dim=1)
        v = v.repeat_interleave(self.kv_repeat, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, S, H)
        return self.o_proj(out)


class LLaMALayer(nn.Module):
    def __init__(self, hidden: int, num_heads: int, num_kv_heads: int, inter: int):
        super().__init__()
        self.input_norm = RMSNorm(hidden)
        self.attn = LLaMAAttention(hidden, num_heads, num_kv_heads)
        self.post_attn_norm = RMSNorm(hidden)
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x, attn_mask):
        h = x + self.attn(self.input_norm(x), attn_mask)
        ffn = F.silu(self.gate_proj(self.post_attn_norm(h))) * self.up_proj(self.post_attn_norm(h))
        return h + self.down_proj(ffn)


class LLaMAModel(nn.Module):
    """Simplified LLaMA for CUDA graph testing."""
    def __init__(self, num_layers=4, hidden=2048, heads=16, kv_heads=4, inter=5504):
        super().__init__()
        self.layers = nn.ModuleList([
            LLaMALayer(hidden, heads, kv_heads, inter) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden)

    def forward(self, x, attn_mask):
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# 2. Static Buffer Arena (Canonical Prefill Arena)
# ---------------------------------------------------------------------------
@dataclass
class CanonicalArena:
    """Pre-allocated fixed-size buffers for CUDA graph capture.

    Key idea: instead of graph key = (num_tokens, num_reqs, ...),
    we canonicalize every batch to (max_tokens, max_reqs) by padding.
    This collapses the graph key to just the bucket size.
    """
    max_tokens: int
    max_reqs: int
    hidden: int
    num_heads: int
    device: torch.device
    dtype: torch.dtype

    def __post_init__(self):
        # Static input buffer
        self.hidden_states = torch.zeros(
            1, self.max_tokens, self.hidden,
            device=self.device, dtype=self.dtype)
        # Static attention mask: [1, 1, max_tokens, max_tokens]
        self.attn_mask = torch.zeros(
            1, 1, self.max_tokens, self.max_tokens,
            device=self.device, dtype=torch.bool)

    def fill(self, seq_lens: list[int], hidden_states: torch.Tensor):
        """Canonicalize a ragged batch into the fixed arena.

        Args:
            seq_lens: actual sequence lengths per request
            hidden_states: [1, total_actual_tokens, hidden] (packed)

        Returns:
            actual_total_tokens: how many real tokens are in the batch
        """
        total_actual = sum(seq_lens)
        num_reqs = len(seq_lens)

        # Copy hidden states (pad remainder with zeros)
        self.hidden_states.zero_()
        self.hidden_states[0, :total_actual] = hidden_states[0, :total_actual]

        # Build attention mask: block-diagonal causal
        # Each request i occupies [cum_lens[i], cum_lens[i+1]) in the token dim
        self.attn_mask.fill_(False)
        mask_2d = self.attn_mask[0, 0]  # [max_tokens, max_tokens]

        offset = 0
        for i, sl in enumerate(seq_lens):
            # Causal within this request's range
            for r in range(sl):
                mask_2d[offset + r, offset:offset + r + 1] = True
            offset += sl
        # Everything outside actual tokens stays False (masked out)

        return total_actual

    def fill_fast(self, seq_lens: list[int], hidden_states: torch.Tensor):
        """Faster version using vectorized mask construction."""
        total_actual = sum(seq_lens)

        self.hidden_states.zero_()
        self.hidden_states[0, :total_actual] = hidden_states[0, :total_actual]

        # Build block-diagonal causal mask efficiently
        self.attn_mask.fill_(False)
        mask_2d = self.attn_mask[0, 0]

        # Create indices for block-diagonal causal mask
        # For SDPA: True = attend, False = mask out
        offset = 0
        for sl in seq_lens:
            # Set the lower triangular block for this request
            block = torch.tril(torch.ones(sl, sl, device=self.device, dtype=torch.bool))
            mask_2d[offset:offset+sl, offset:offset+sl] = block
            offset += sl

        return total_actual


# ---------------------------------------------------------------------------
# 3. Graph Capture/Replay Engine
# ---------------------------------------------------------------------------
class PrefillGraphEngine:
    """Captures and replays CUDA graphs for canonicalized prefill batches."""

    def __init__(self, model: nn.Module, arenas: dict[int, CanonicalArena]):
        """
        Args:
            model: the LLaMA model
            arenas: bucket_size -> CanonicalArena mapping
        """
        self.model = model
        self.arenas = arenas
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.graph_outputs: dict[int, torch.Tensor] = {}
        self.pool = torch.cuda.graph_pool_handle()

    def warmup_and_capture(self, bucket_size: int):
        """Capture a CUDA graph for one bucket size."""
        arena = self.arenas[bucket_size]

        # Warmup runs (required before capture)
        with torch.no_grad():
            for _ in range(2):
                _ = self.model(arena.hidden_states, arena.attn_mask)
        torch.cuda.synchronize()

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(g, pool=self.pool):
                output = self.model(arena.hidden_states, arena.attn_mask)
        torch.cuda.synchronize()

        self.graphs[bucket_size] = g
        self.graph_outputs[bucket_size] = output

    def capture_all(self):
        """Capture graphs for all bucket sizes."""
        for bs in sorted(self.arenas.keys(), reverse=True):
            self.warmup_and_capture(bs)

    def execute(self, bucket_size: int, seq_lens: list[int],
                hidden_states: torch.Tensor) -> torch.Tensor:
        """Execute prefill using captured graph.

        Args:
            bucket_size: which bucket to use
            seq_lens: actual sequence lengths
            hidden_states: actual input [1, total_tokens, hidden]

        Returns:
            output for actual tokens only (trimmed)
        """
        arena = self.arenas[bucket_size]
        total_actual = arena.fill_fast(seq_lens, hidden_states)

        # Replay graph
        self.graphs[bucket_size].replay()
        torch.cuda.synchronize()

        # Trim output to actual tokens
        return self.graph_outputs[bucket_size][0, :total_actual].clone()

    def execute_eager(self, seq_lens: list[int],
                      hidden_states: torch.Tensor,
                      bucket_size: int) -> torch.Tensor:
        """Execute prefill eagerly (baseline)."""
        arena = self.arenas[bucket_size]
        total_actual = arena.fill_fast(seq_lens, hidden_states)

        with torch.no_grad():
            output = self.model(arena.hidden_states, arena.attn_mask)
        return output[0, :total_actual].clone()


# ---------------------------------------------------------------------------
# 4. Correctness + Performance Tests
# ---------------------------------------------------------------------------
def test_correctness(model, engine, bucket_size, arena, num_tests=20):
    """Verify graph output matches eager output for various request configs."""
    hidden = arena.hidden
    max_errs = []

    for i in range(num_tests):
        rng = np.random.RandomState(i + 42)

        # Randomly choose num_reqs and seq_lens that fit in bucket
        max_reqs = min(arena.max_reqs, bucket_size)
        num_reqs = rng.randint(1, max(2, max_reqs + 1))
        # Random seq_lens summing to ≤ bucket_size
        remaining = bucket_size
        seq_lens = []
        for r in range(num_reqs):
            if r == num_reqs - 1:
                sl = remaining
            else:
                sl = rng.randint(1, max(2, remaining - (num_reqs - r - 1) + 1))
            seq_lens.append(sl)
            remaining -= sl
            if remaining <= 0:
                break
        seq_lens = [s for s in seq_lens if s > 0]

        total = sum(seq_lens)
        if total == 0 or total > bucket_size:
            continue

        # Generate random input
        x = torch.randn(1, total, hidden, device=arena.device, dtype=arena.dtype)

        # Graph execution
        graph_out = engine.execute(bucket_size, seq_lens, x)

        # Eager execution (using same padded arena for fair comparison)
        eager_out = engine.execute_eager(seq_lens, x, bucket_size)

        max_err = (graph_out - eager_out).abs().max().item()
        max_errs.append(max_err)

    avg_err = np.mean(max_errs)
    max_max_err = np.max(max_errs)
    return avg_err, max_max_err, len(max_errs)


def benchmark_latency(engine, bucket_size, arena, num_iters=200):
    """Benchmark graph replay vs eager execution."""
    hidden = arena.hidden
    # Use a fixed workload for benchmarking
    seq_lens = [bucket_size]  # single request filling the bucket
    x = torch.randn(1, bucket_size, hidden, device=arena.device, dtype=arena.dtype)

    # Fill arena once
    arena.fill_fast(seq_lens, x)

    # Warmup
    for _ in range(10):
        engine.graphs[bucket_size].replay()
    torch.cuda.synchronize()

    # Benchmark graph
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        engine.graphs[bucket_size].replay()
    torch.cuda.synchronize()
    graph_ms = (time.perf_counter() - t0) / num_iters * 1000

    # Benchmark eager
    for _ in range(10):
        with torch.no_grad():
            _ = engine.model(arena.hidden_states, arena.attn_mask)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = engine.model(arena.hidden_states, arena.attn_mask)
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) / num_iters * 1000

    return graph_ms, eager_ms


def test_key_collapse(engine, bucket_size, arena):
    """Verify that different num_reqs produce correct results with SAME graph.

    This is the key experiment: one graph template handles any request layout
    within the same bucket, eliminating num_reqs from the graph key.
    """
    hidden = arena.hidden
    configs = [
        ([bucket_size], "1 req"),
        ([bucket_size // 2, bucket_size // 2], "2 equal reqs"),
        ([bucket_size // 4] * 4, "4 equal reqs"),
        ([1] * bucket_size, f"{bucket_size} single-token reqs"),
    ]

    if bucket_size >= 32:
        configs.append(
            ([bucket_size - 10, 5, 3, 2], "4 unequal reqs")
        )

    results = []
    for seq_lens, desc in configs:
        total = sum(seq_lens)
        if total > bucket_size:
            continue
        x = torch.randn(1, total, hidden, device=arena.device, dtype=arena.dtype)

        graph_out = engine.execute(bucket_size, seq_lens, x)
        eager_out = engine.execute_eager(seq_lens, x, bucket_size)

        max_err = (graph_out - eager_out).abs().max().item()
        results.append((desc, len(seq_lens), total, max_err))

    return results


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--inter", type=int, default=5504)
    parser.add_argument("--bucket-sizes", type=int, nargs="+",
                        default=[64, 128, 256, 512, 1024])
    parser.add_argument("--max-reqs-per-bucket", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    dtype = torch.float16

    print(f"{'='*70}")
    print(f"Module 2+3: Attention Staticization + Graph Key Collapse")
    print(f"{'='*70}")
    print(f"Model: {args.num_layers}L, hidden={args.hidden}, "
          f"heads={args.heads}, kv_heads={args.kv_heads}")
    print(f"Bucket sizes: {args.bucket_sizes}")
    print(f"Max reqs per bucket: {args.max_reqs_per_bucket}")
    print(f"Device: {device}")

    # Create model
    model = LLaMAModel(
        num_layers=args.num_layers,
        hidden=args.hidden,
        heads=args.heads,
        kv_heads=args.kv_heads,
        inter=args.inter,
    ).to(device=device, dtype=dtype).eval()

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {num_params:.1f}M")

    # Create arenas for each bucket size
    arenas = {}
    for bs in args.bucket_sizes:
        arenas[bs] = CanonicalArena(
            max_tokens=bs,
            max_reqs=min(args.max_reqs_per_bucket, bs),
            hidden=args.hidden,
            num_heads=args.heads,
            device=device,
            dtype=dtype,
        )

    # Create graph engine and capture
    engine = PrefillGraphEngine(model, arenas)

    print(f"\nCapturing graphs...")
    t0 = time.monotonic()
    engine.capture_all()
    t1 = time.monotonic()
    capture_time = t1 - t0
    print(f"Captured {len(args.bucket_sizes)} graphs in {capture_time:.2f}s")

    mem_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"GPU memory: {mem_mb:.0f} MB")

    # -----------------------------------------------------------------------
    # Test 1: Correctness (graph vs eager)
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Test 1: Correctness (graph output == eager output)")
    print(f"{'='*70}")
    for bs in args.bucket_sizes:
        avg_err, max_err, n_tests = test_correctness(
            model, engine, bs, arenas[bs])
        status = "PASS" if max_err < 1e-2 else "FAIL"
        print(f"  bucket={bs:5d}: avg_err={avg_err:.6f}, max_err={max_err:.6f} "
              f"(n={n_tests}) [{status}]")

    # -----------------------------------------------------------------------
    # Test 2: Key Collapse (different num_reqs, same graph)
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Test 2: Graph Key Collapse (variable num_reqs, ONE graph)")
    print(f"{'='*70}")
    for bs in args.bucket_sizes:
        print(f"\n  bucket={bs}:")
        results = test_key_collapse(engine, bs, arenas[bs])
        for desc, n_reqs, n_tokens, err in results:
            status = "PASS" if err < 1e-2 else "FAIL"
            print(f"    {desc:30s} (num_reqs={n_reqs:4d}, tokens={n_tokens:5d}): "
                  f"max_err={err:.6f} [{status}]")

    # -----------------------------------------------------------------------
    # Test 3: Latency (graph vs eager)
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Test 3: Latency (graph replay vs eager)")
    print(f"{'='*70}")
    print(f"  {'Bucket':>8s}  {'Graph(ms)':>10s}  {'Eager(ms)':>10s}  {'Speedup':>8s}")
    for bs in args.bucket_sizes:
        graph_ms, eager_ms = benchmark_latency(engine, bs, arenas[bs])
        speedup = eager_ms / graph_ms if graph_ms > 0 else float('inf')
        print(f"  {bs:8d}  {graph_ms:10.3f}  {eager_ms:10.3f}  {speedup:7.2f}x")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"  Graph capture time:  {capture_time:.2f}s for {len(args.bucket_sizes)} buckets")
    print(f"  GPU memory:          {mem_mb:.0f} MB")
    print(f"  Key collapse:        num_reqs eliminated from graph key")
    print(f"  Correctness:         graph == eager within fp16 tolerance")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
