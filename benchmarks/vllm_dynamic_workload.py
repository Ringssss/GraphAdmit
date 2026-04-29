#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig


def make_compilation(mode: str, sizes: list[int]) -> CompilationConfig:
    return CompilationConfig(
        cudagraph_mode=mode,
        cudagraph_capture_sizes=sizes,
        max_cudagraph_capture_size=max(sizes) if sizes else 0,
    )


def build_prompts(lengths: list[int]) -> list[str]:
    prompts = []
    for length in lengths:
        base = "CUDA graph staticity "
        repeat = max(1, length // 4)
        prompts.append((base * repeat)[: max(1, length * 5)])
    return prompts


def run_case(args, mode: str, lengths: list[int]) -> dict:
    trace_file = None
    old_trace = os.environ.get("VLLM_CG_TRACE_FILE")
    if args.trace_dir:
        Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
        trace_file = str(Path(args.trace_dir) / f"vllm_dynamic_{mode.lower()}_{len(lengths)}.jsonl")
        Path(trace_file).write_text("")
        os.environ["VLLM_CG_TRACE_FILE"] = trace_file

    cc = make_compilation(mode, args.capture_sizes)
    speculative_config = None
    if args.spec_model and args.num_speculative_tokens > 0:
        speculative_config = {
            "model": args.spec_model,
            "method": args.spec_method,
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_tensor_parallel_size": args.draft_tp,
        }
    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        compilation_config=cc,
        enable_prefix_caching=False,
        speculative_config=speculative_config,
    )
    t1 = time.time()
    prompts = build_prompts(lengths)
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    warmup_prompts = build_prompts([min(lengths) if lengths else 64])
    llm.generate(warmup_prompts, sampling)
    start = time.time()
    outputs = llm.generate(prompts, sampling)
    end = time.time()
    result = {
        "mode": mode,
        "model": args.model,
        "tp": args.tp,
        "lengths": lengths,
        "num_prompts": len(prompts),
        "init_s": t1 - t0,
        "batch_s": end - start,
        "throughput_req_s": len(prompts) / max(end - start, 1e-9),
        "trace_file": trace_file,
        "spec_model": args.spec_model or None,
        "num_speculative_tokens": args.num_speculative_tokens,
        "first_text": outputs[0].outputs[0].text if outputs else "",
    }
    if old_trace is None:
        os.environ.pop("VLLM_CG_TRACE_FILE", None)
    else:
        os.environ["VLLM_CG_TRACE_FILE"] = old_trace
    return result


def parse_lengths(spec: str) -> list[int]:
    return [int(x) for x in spec.replace(";", ",").split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/models/Qwen3-32B")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--lengths", default="64,128,256,512,768")
    parser.add_argument("--modes", nargs="+", default=["PIECEWISE", "FULL_AND_PIECEWISE"])
    parser.add_argument("--capture-sizes", type=int, nargs="+", default=[1,2,4,8,16,32,64,128,256,512,768,1024])
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--spec-model", default="")
    parser.add_argument("--spec-method", default="eagle3")
    parser.add_argument("--num-speculative-tokens", type=int, default=0)
    parser.add_argument("--draft-tp", type=int, default=1)
    parser.add_argument("--trace-dir", default="traces")
    parser.add_argument("--output", default="results/vllm_dynamic_workload.json")
    args = parser.parse_args()
    lengths = parse_lengths(args.lengths)
    rows = [run_case(args, mode, lengths) for mode in args.modes]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
