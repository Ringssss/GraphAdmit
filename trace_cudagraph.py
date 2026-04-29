"""
Experiment 1: CUDA Graph Trace Study for Dynamic Prefill
=========================================================
Instruments vLLM's CudagraphDispatcher to record every dispatch decision,
then runs real inference with various prompt lengths to characterize:
- Graph hit/miss rates by mode (FULL/PIECEWISE/NONE)
- Padding waste distribution
- Graph family (unique BatchDescriptor) counts
- Fallback frequency and reasons
- Coverage metrics

Usage:
    python trace_cudagraph.py --model /mnt/models/Meta-Llama-3-8B-Instruct \
                              --mode FULL_AND_PIECEWISE \
                              --num-prompts 200
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. Trace record
# ---------------------------------------------------------------------------
@dataclass
class DispatchRecord:
    """One record per dispatch() call."""
    timestamp_us: float
    num_tokens_raw: int          # before padding
    num_tokens_padded: int       # after padding
    num_reqs: Optional[int]
    uniform: bool
    has_lora: bool
    mode_dispatched: str         # FULL / PIECEWISE / NONE
    batch_descriptor_key: str    # str(BatchDescriptor)
    padding_waste: int           # padded - raw
    is_capture: bool             # True = first time (capture), False = replay


# Global trace buffer
_trace_records: list[DispatchRecord] = []
_captured_descriptors: dict[str, set] = defaultdict(set)  # mode -> set of keys


# ---------------------------------------------------------------------------
# 2. Monkey-patch the dispatcher
# ---------------------------------------------------------------------------
def install_dispatch_hook():
    """Monkey-patch CudagraphDispatcher.dispatch to record every call."""
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    from vllm.config.compilation import CUDAGraphMode

    _original_dispatch = CudagraphDispatcher.dispatch

    def patched_dispatch(self, num_tokens, **kwargs):
        t0 = time.monotonic()
        mode, batch_desc = _original_dispatch(self, num_tokens, **kwargs)
        t1 = time.monotonic()

        key = str(batch_desc)
        mode_name = mode.name

        # Check if this is a first-time capture
        is_capture = key not in _captured_descriptors[mode_name]
        if mode != CUDAGraphMode.NONE:
            _captured_descriptors[mode_name].add(key)

        record = DispatchRecord(
            timestamp_us=(t0 + t1) / 2 * 1e6,
            num_tokens_raw=num_tokens,
            num_tokens_padded=batch_desc.num_tokens,
            num_reqs=batch_desc.num_reqs,
            uniform=batch_desc.uniform,
            has_lora=batch_desc.has_lora,
            mode_dispatched=mode_name,
            batch_descriptor_key=key,
            padding_waste=batch_desc.num_tokens - num_tokens,
            is_capture=is_capture,
        )
        _trace_records.append(record)
        return mode, batch_desc

    CudagraphDispatcher.dispatch = patched_dispatch
    print(f"[trace] Installed dispatch hook on CudagraphDispatcher")


# ---------------------------------------------------------------------------
# 3. Also hook CUDAGraphWrapper to count captures vs replays
# ---------------------------------------------------------------------------
_graph_events: list[dict] = []


def install_cudagraph_wrapper_hook():
    """Hook CUDAGraphWrapper.__call__ to log capture vs replay events."""
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.config.compilation import CUDAGraphMode

    _original_call = CUDAGraphWrapper.__call__

    def patched_call(self, *args, **kwargs):
        from vllm.forward_context import get_forward_context, is_forward_context_available

        event = {
            "runtime_mode": self.runtime_mode.name,
            "action": "unknown",
            "batch_descriptor": None,
        }

        if is_forward_context_available():
            ctx = get_forward_context()
            bd = ctx.batch_descriptor
            cm = ctx.cudagraph_runtime_mode

            if cm == CUDAGraphMode.NONE or cm != self.runtime_mode:
                event["action"] = "passthrough"
            elif bd is not None and bd in self.concrete_cudagraph_entries:
                entry = self.concrete_cudagraph_entries[bd]
                event["action"] = "replay" if entry.cudagraph is not None else "capture"
            else:
                event["action"] = "capture"

            if bd is not None:
                event["batch_descriptor"] = str(bd)

        result = _original_call(self, *args, **kwargs)
        _graph_events.append(event)
        return result

    CUDAGraphWrapper.__call__ = patched_call
    print(f"[trace] Installed hook on CUDAGraphWrapper.__call__")


# ---------------------------------------------------------------------------
# 4. Generate diverse prompts
# ---------------------------------------------------------------------------
def generate_prompts(num_prompts: int, distribution: str = "lognormal"):
    """Generate prompts with various lengths matching real serving patterns."""
    rng = np.random.RandomState(42)

    if distribution == "lognormal":
        # Log-normal with median ~128 tokens, heavy tail to 2048
        raw_lens = rng.lognormal(mean=4.5, sigma=1.0, size=num_prompts)
        raw_lens = np.clip(raw_lens, 8, 2048).astype(int)
    elif distribution == "uniform":
        raw_lens = rng.randint(16, 1024, size=num_prompts)
    elif distribution == "bimodal":
        # Mix of short (32-64) and long (512-1024)
        short = rng.randint(32, 64, size=num_prompts // 2)
        long = rng.randint(512, 1024, size=num_prompts - num_prompts // 2)
        raw_lens = np.concatenate([short, long])
        rng.shuffle(raw_lens)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Create prompts: repeat a short sentence to reach target length
    base = "The quick brown fox jumps over the lazy dog. "
    prompts = []
    for length in raw_lens:
        # Approximate: each word ≈ 1.3 tokens
        num_words = int(length / 1.3)
        text = (base * (num_words // len(base.split()) + 1))[:num_words * 6]
        prompts.append(text)

    print(f"[trace] Generated {len(prompts)} prompts, "
          f"target token lengths: min={raw_lens.min()}, median={int(np.median(raw_lens))}, "
          f"max={raw_lens.max()}, mean={raw_lens.mean():.0f}")
    return prompts, raw_lens


# ---------------------------------------------------------------------------
# 5. Run inference and collect traces
# ---------------------------------------------------------------------------
def run_trace_collection(
    model_path: str,
    cudagraph_mode: str,
    num_prompts: int,
    distribution: str,
    tp_size: int,
    max_model_len: int,
    output_dir: str,
    gpu_memory_utilization: float,
):
    """Run vLLM inference and collect CUDA graph dispatch traces."""

    # Install hooks BEFORE importing LLM (which triggers model loading)
    install_dispatch_hook()
    install_cudagraph_wrapper_hook()

    from vllm import LLM, SamplingParams

    print(f"\n[trace] Loading model: {model_path}")
    print(f"[trace] CUDAGraph mode: {cudagraph_mode}")
    print(f"[trace] TP size: {tp_size}")

    # Map mode string to compilation config
    compilation_config = {
        "cudagraph_mode": cudagraph_mode,
    }

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        compilation_config=compilation_config,
        enforce_eager=False,
    )

    prompts, target_lens = generate_prompts(num_prompts, distribution)
    sampling_params = SamplingParams(
        max_tokens=1,  # We only care about prefill, generate 1 token
        temperature=0.0,
    )

    print(f"\n[trace] Starting inference ({num_prompts} prompts)...")
    t_start = time.monotonic()
    outputs = llm.generate(prompts, sampling_params)
    t_end = time.monotonic()

    elapsed = t_end - t_start
    print(f"[trace] Inference done in {elapsed:.2f}s "
          f"({num_prompts/elapsed:.1f} prompts/s)")

    # ---------------------------------------------------------------------------
    # 6. Analyze and save
    # ---------------------------------------------------------------------------
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save raw trace
    trace_data = {
        "config": {
            "model": model_path,
            "cudagraph_mode": cudagraph_mode,
            "num_prompts": num_prompts,
            "distribution": distribution,
            "tp_size": tp_size,
            "max_model_len": max_model_len,
            "elapsed_s": elapsed,
        },
        "dispatch_records": [asdict(r) for r in _trace_records],
        "graph_events": _graph_events,
    }

    trace_file = out_path / f"trace_{cudagraph_mode}_{distribution}_{num_prompts}.json"
    with open(trace_file, "w") as f:
        json.dump(trace_data, f, indent=2)
    print(f"[trace] Saved raw trace to {trace_file}")

    # Print summary
    print_trace_summary(cudagraph_mode)

    return trace_file


def print_trace_summary(mode_name: str):
    """Print analysis of collected trace."""
    print(f"\n{'='*70}")
    print(f"CUDA Graph Trace Summary (mode={mode_name})")
    print(f"{'='*70}")

    if not _trace_records:
        print("No dispatch records collected.")
        return

    total = len(_trace_records)
    mode_counts = Counter(r.mode_dispatched for r in _trace_records)
    capture_count = sum(1 for r in _trace_records if r.is_capture)
    replay_count = sum(1 for r in _trace_records
                       if not r.is_capture and r.mode_dispatched != "NONE")

    print(f"\n--- Dispatch Decisions ({total} total) ---")
    for mode, count in mode_counts.most_common():
        pct = count / total * 100
        print(f"  {mode:12s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n--- Capture vs Replay ---")
    print(f"  Captures:    {capture_count}")
    print(f"  Replays:     {replay_count}")
    print(f"  Passthrough: {mode_counts.get('NONE', 0)}")

    # Graph family analysis
    print(f"\n--- Graph Families (unique BatchDescriptors) ---")
    for m, keys in _captured_descriptors.items():
        print(f"  {m:12s}: {len(keys)} unique families")

    # Padding waste
    wastes = [r.padding_waste for r in _trace_records if r.mode_dispatched != "NONE"]
    if wastes:
        wastes = np.array(wastes)
        tokens_raw = np.array([r.num_tokens_raw for r in _trace_records
                               if r.mode_dispatched != "NONE"])
        total_waste = wastes.sum()
        total_raw = tokens_raw.sum()
        print(f"\n--- Padding Waste ---")
        print(f"  Total padded tokens:   {total_raw + total_waste}")
        print(f"  Total actual tokens:   {total_raw}")
        print(f"  Total wasted:          {total_waste} "
              f"({total_waste/(total_raw+total_waste)*100:.1f}%)")
        print(f"  Per-dispatch waste:    "
              f"min={wastes.min()}, median={int(np.median(wastes))}, "
              f"max={wastes.max()}, mean={wastes.mean():.1f}")

    # num_reqs distribution
    reqs = [r.num_reqs for r in _trace_records
            if r.num_reqs is not None and r.mode_dispatched != "NONE"]
    if reqs:
        reqs = np.array(reqs)
        print(f"\n--- num_reqs in dispatched batches ---")
        print(f"  min={reqs.min()}, median={int(np.median(reqs))}, "
              f"max={reqs.max()}, unique={len(set(reqs))}")

    # uniform distribution
    uniform_count = sum(1 for r in _trace_records if r.uniform)
    print(f"\n--- Batch Types ---")
    print(f"  Uniform:     {uniform_count}")
    print(f"  Non-uniform: {total - uniform_count}")

    # GraphWrapper events
    if _graph_events:
        action_counts = Counter(e["action"] for e in _graph_events)
        print(f"\n--- CUDAGraphWrapper Events ({len(_graph_events)} total) ---")
        for action, count in action_counts.most_common():
            print(f"  {action:12s}: {count}")

    print(f"\n{'='*70}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CUDA Graph Trace Study")
    parser.add_argument("--model", type=str,
                        default="/mnt/models/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--mode", type=str, default="FULL_AND_PIECEWISE",
                        choices=["NONE", "PIECEWISE", "FULL",
                                 "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"])
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--distribution", type=str, default="lognormal",
                        choices=["lognormal", "uniform", "bimodal"])
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--output-dir", type=str, default="traces")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    run_trace_collection(
        model_path=args.model,
        cudagraph_mode=args.mode,
        num_prompts=args.num_prompts,
        distribution=args.distribution,
        tp_size=args.tp_size,
        max_model_len=args.max_model_len,
        output_dir=args.output_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
