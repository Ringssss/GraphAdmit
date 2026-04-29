#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def make_prompt(tokenizer, prompt: str, device):
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    )
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    return encoded.to(device)


def build_static_inputs(input_ids, total_length, block_length, mask_id, device):
    num_blocks = (total_length + block_length - 1) // block_length
    padded_total = num_blocks * block_length
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    attn_mask = (block_mask.repeat_interleave(block_length, 0)
                 .repeat_interleave(block_length, 1)
                 .unsqueeze(0).unsqueeze(0).log().to(torch.bfloat16))
    position_ids = torch.arange(padded_total, device=device).unsqueeze(0)
    x = torch.full((input_ids.shape[0], padded_total), mask_id, dtype=torch.long, device=device)
    x[:, :input_ids.shape[1]].copy_(input_ids)
    return x, attn_mask, position_ids


@torch.no_grad()
def loop_forward(model, x, attn_mask, position_ids, windows, steps):
    logits = None
    torch.cuda.synchronize()
    start = time.time()
    for window in windows:
        for _ in range(steps):
            logits = model(
                x[:, :window],
                attention_mask=attn_mask[:, :, :window, :window],
                position_ids=position_ids[:, :window],
            ).logits
    torch.cuda.synchronize()
    return {"seconds": time.time() - start, "last_shape": list(logits.shape)}


@torch.no_grad()
def graph_forward(model, x, attn_mask, position_ids, windows, steps):
    window = max(windows)
    static_x = x[:, :window].contiguous()
    static_attn = attn_mask[:, :, :window, :window].contiguous()
    static_pos = position_ids[:, :window].contiguous()
    logits = None
    for _ in range(2):
        logits = model(static_x, attention_mask=static_attn, position_ids=static_pos).logits
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = model(static_x, attention_mask=static_attn, position_ids=static_pos).logits
    torch.cuda.synchronize()
    start = time.time()
    for _window in windows:
        for _ in range(steps):
            graph.replay()
    torch.cuda.synchronize()
    return {"seconds": time.time() - start, "last_shape": list(logits.shape)}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/models/LLaDA2.0-mini")
    parser.add_argument("--prompt", default="Say hi in five words.")
    parser.add_argument("--gen-length", type=int, default=32)
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", default="results/llada2_hf_staticity_e2e.json")
    args = parser.parse_args()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).eval().to(device)
    torch.cuda.synchronize()
    load_s = time.time() - t0
    input_ids = make_prompt(tokenizer, args.prompt, device)
    total_length = input_ids.shape[1] + args.gen_length
    x, attn_mask, pos = build_static_inputs(input_ids, total_length, args.block_length, 156895, device)
    prompt_blocks = input_ids.shape[1] // args.block_length
    total_blocks = x.shape[1] // args.block_length
    windows = [(b + 1) * args.block_length for b in range(prompt_blocks, total_blocks)]
    eager = loop_forward(model, x, attn_mask, pos, windows, args.steps)
    try:
        graph = graph_forward(model, x, attn_mask, pos, windows, args.steps)
        error = None
    except Exception as exc:
        graph = None
        error = repr(exc)
    result = {
        "model": args.model,
        "prompt_len": input_ids.shape[1],
        "total_static_len": x.shape[1],
        "windows": windows,
        "steps": args.steps,
        "num_forwards": len(windows) * args.steps,
        "load_s": load_s,
        "eager": eager,
        "cudagraph": graph,
        "cudagraph_error": error,
        "speedup": (eager["seconds"] / graph["seconds"]) if graph else None,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
