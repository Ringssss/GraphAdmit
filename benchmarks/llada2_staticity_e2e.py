#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer
from vllm import distributed
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context

from dinfer.model import LLaDA2MoeModelLM


CURRENT_VLLM_CONFIG = None

def init_vllm_single(port: int):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(port))
    pc = ParallelConfig(tensor_parallel_size=1, enable_expert_parallel=False)
    global CURRENT_VLLM_CONFIG
    vc = VllmConfig(parallel_config=pc)
    CURRENT_VLLM_CONFIG = vc
    ctx = set_current_vllm_config(vc)
    ctx.__enter__()
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")
    return ctx


def make_prompt(tokenizer, prompt: str, device):
    text = '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>' + prompt + '<|role_end|><role>ASSISTANT</role>'
    return torch.tensor(tokenizer(text)["input_ids"], device=device).unsqueeze(0)


def build_static_inputs(input_ids, total_length, block_length, mask_id, device):
    num_blocks = (total_length + block_length - 1) // block_length
    padded_total = num_blocks * block_length
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    attn_mask = (block_mask.repeat_interleave(block_length, 0)
                 .repeat_interleave(block_length, 1)
                 .unsqueeze(0).unsqueeze(0).log().to(torch.bfloat16))
    position_ids = torch.arange(padded_total, device=device).unsqueeze(0)
    x = torch.full((1, padded_total), mask_id, dtype=torch.long, device=device)
    x[:, :input_ids.shape[1]].copy_(input_ids)
    return x, attn_mask, position_ids


@torch.no_grad()
def forward_loop(model, x, attn_mask, position_ids, windows, steps, graph=False):
    logits_ref = None
    graph_obj = None
    static_x = x.clone()
    timings = []
    if graph:
        window = max(windows)
        static_attn = attn_mask[:, :, :window, :window].contiguous()
        static_pos = position_ids[:, :window].contiguous()
        static_in = static_x[:, :window]
        for _ in range(2):
            with set_forward_context(None, CURRENT_VLLM_CONFIG, num_tokens=static_in.numel()):
                logits_ref = model(static_in, attention_mask=static_attn, position_ids=static_pos).logits
        torch.cuda.synchronize()
        graph_obj = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_obj):
            with set_forward_context(None, CURRENT_VLLM_CONFIG, num_tokens=static_in.numel()):
                logits_ref = model(static_in, attention_mask=static_attn, position_ids=static_pos).logits
        torch.cuda.synchronize()
        start = time.time()
        for _window in windows:
            for _ in range(steps):
                graph_obj.replay()
        torch.cuda.synchronize()
        timings.append(time.time() - start)
        return {"seconds": sum(timings), "last_shape": list(logits_ref.shape), "mode": "cudagraph_max_window"}
    start = time.time()
    for window in windows:
        cur_x = static_x[:, :window]
        cur_attn = attn_mask[:, :, :window, :window]
        cur_pos = position_ids[:, :window]
        for _ in range(steps):
            with set_forward_context(None, CURRENT_VLLM_CONFIG, num_tokens=cur_x.numel()):
                logits_ref = model(cur_x, attention_mask=cur_attn, position_ids=cur_pos).logits
    torch.cuda.synchronize()
    return {"seconds": time.time() - start, "last_shape": list(logits_ref.shape), "mode": "eager_dynamic_window"}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/models/LLaDA2.0-mini")
    parser.add_argument("--prompt", default="Say hi in five words.")
    parser.add_argument("--gen-length", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--port", type=int, default=46031)
    parser.add_argument("--output", default="results/llada2_staticity_e2e.json")
    args = parser.parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    ctx = init_vllm_single(args.port)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        model = LLaDA2MoeModelLM(config=config).eval()
        t0 = time.time()
        model.load_weights(args.model, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        torch.cuda.synchronize()
        load_s = time.time() - t0
        input_ids = make_prompt(tokenizer, args.prompt, device)
        total_length = input_ids.shape[1] + args.gen_length
        mask_id = 156895
        x, attn_mask, pos = build_static_inputs(input_ids, total_length, args.block_length, mask_id, device)
        prompt_blocks = input_ids.shape[1] // args.block_length
        total_blocks = x.shape[1] // args.block_length
        windows = [(b + 1) * args.block_length for b in range(prompt_blocks, total_blocks)]
        eager = forward_loop(model, x, attn_mask, pos, windows, args.steps, graph=False)
        graphed = forward_loop(model, x, attn_mask, pos, windows, args.steps, graph=True)
        result = {
            "model": args.model,
            "prompt_len": input_ids.shape[1],
            "total_static_len": x.shape[1],
            "windows": windows,
            "steps": args.steps,
            "num_forwards": len(windows) * args.steps,
            "load_s": load_s,
            "eager": eager,
            "cudagraph": graphed,
            "speedup": eager["seconds"] / max(graphed["seconds"], 1e-9),
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
