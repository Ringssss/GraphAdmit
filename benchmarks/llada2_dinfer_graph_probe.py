#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

DINFER_ROOT = Path('/home/zhujianian/eurosys/dInfer')
if str(DINFER_ROOT / 'python') not in sys.path:
    sys.path.insert(0, str(DINFER_ROOT / 'python'))

from dinfer.model import LLaDA2MoeModelLM
from vllm import distributed
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config


def init_vllm(rank: int, world_size: int, port: int, enable_ep: bool):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', str(port))
    parallel_config = ParallelConfig(enable_expert_parallel=enable_ep)
    ctx = set_current_vllm_config(VllmConfig(parallel_config=parallel_config))
    ctx.__enter__()
    distributed.init_distributed_environment(world_size, rank, 'env://', rank, 'nccl')
    distributed.initialize_model_parallel(world_size, backend='nccl')
    return ctx


def make_prompt(tokenizer, text: str, device):
    prompt = '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>' + text + '<|role_end|><role>ASSISTANT</role>'
    return torch.tensor(tokenizer(prompt)['input_ids'], dtype=torch.long, device=device).unsqueeze(0)


def build_bd_inputs(input_ids, gen_length: int, block_length: int, mask_id: int):
    device = input_ids.device
    prompt_length = input_ids.shape[1]
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length
    x = torch.full((input_ids.shape[0], total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length].copy_(input_ids)
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    attn_mask = (block_mask.repeat_interleave(block_length, dim=0)
                 .repeat_interleave(block_length, dim=1)
                 .unsqueeze(0).log().to(torch.bfloat16))
    pos_ids = torch.arange(total_length, dtype=torch.long, device=device).unsqueeze(0)
    prompt_blocks = prompt_length // block_length
    windows = [(block_id + 1) * block_length for block_id in range(prompt_blocks, num_blocks)]
    return x, attn_mask, pos_ids, windows, total_length


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def run_dynamic_eager(model, x, attn_mask, pos_ids, windows, steps, warmups):
    logits = None
    for _ in range(warmups):
        for window in windows:
            logits = model(
                x[:, :window].clone(memory_format=torch.contiguous_format),
                use_cache=False,
                attention_mask=attn_mask[:, :window, :window],
                position_ids=pos_ids[:, :window].clone(memory_format=torch.contiguous_format),
            ).logits
    torch.cuda.synchronize()
    start = sync_time()
    for window in windows:
        for _ in range(steps):
            logits = model(
                x[:, :window].clone(memory_format=torch.contiguous_format),
                use_cache=False,
                attention_mask=attn_mask[:, :window, :window],
                position_ids=pos_ids[:, :window].clone(memory_format=torch.contiguous_format),
            ).logits
    end = sync_time()
    return {'seconds': end - start, 'last_shape': list(logits.shape)}


@torch.no_grad()
def run_static_eager(model, static_x, static_attn, static_pos, repeats, warmups):
    logits = None
    for _ in range(warmups):
        logits = model(static_x, use_cache=False, attention_mask=static_attn, position_ids=static_pos).logits
    torch.cuda.synchronize()
    start = sync_time()
    for _ in range(repeats):
        logits = model(static_x, use_cache=False, attention_mask=static_attn, position_ids=static_pos).logits
    end = sync_time()
    return {'seconds': end - start, 'last_shape': list(logits.shape)}


@torch.no_grad()
def run_static_graph(model, static_x, static_attn, static_pos, repeats, warmups):
    logits = None
    for _ in range(warmups):
        logits = model(static_x, use_cache=False, attention_mask=static_attn, position_ids=static_pos).logits
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            logits = model(static_x, use_cache=False, attention_mask=static_attn, position_ids=static_pos).logits
        torch.cuda.synchronize()
        start = sync_time()
        for _ in range(repeats):
            graph.replay()
        end = sync_time()
        return {'seconds': end - start, 'last_shape': list(logits.shape), 'error': None}
    except Exception as exc:
        torch.cuda.synchronize()
        return {'seconds': None, 'last_shape': list(logits.shape) if logits is not None else None, 'error': repr(exc)}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/mnt/models/LLaDA2.0-mini')
    parser.add_argument('--prompt', default='Say hi in five words.')
    parser.add_argument('--gen-length', type=int, default=16)
    parser.add_argument('--block-length', type=int, default=16)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--port', type=int, default=46111)
    parser.add_argument('--output', default='results/llada2_dinfer_graph_probe.json')
    parser.add_argument('--enable-ep', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    ctx = init_vllm(rank=0, world_size=1, port=args.port, enable_ep=args.enable_ep)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        model = LLaDA2MoeModelLM(config=config).eval()
        load_start = time.perf_counter()
        model.load_weights(args.model, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        load_s = sync_time() - load_start
        input_ids = make_prompt(tokenizer, args.prompt, device)
        x, attn_mask, pos_ids, windows, total_length = build_bd_inputs(input_ids, args.gen_length, args.block_length, 156895)
        max_window = max(windows)
        static_x = x[:, :max_window].clone(memory_format=torch.contiguous_format)
        static_attn = attn_mask[:, :max_window, :max_window].contiguous()
        static_pos = pos_ids[:, :max_window].clone(memory_format=torch.contiguous_format)
        repeats = len(windows) * args.steps

        dynamic = run_dynamic_eager(model, x, attn_mask, pos_ids, windows, args.steps, args.warmups)
        static_eager = run_static_eager(model, static_x, static_attn, static_pos, repeats, args.warmups)
        static_graph = run_static_graph(model, static_x, static_attn, static_pos, repeats, args.warmups)
        result = {
            'model': args.model,
            'prompt_len': input_ids.shape[1],
            'gen_length': args.gen_length,
            'block_length': args.block_length,
            'total_length': total_length,
            'windows': windows,
            'steps': args.steps,
            'repeats': repeats,
            'load_s': load_s,
            'dynamic_eager': dynamic,
            'static_eager_max_window': static_eager,
            'static_cuda_graph_max_window': static_graph,
            'graph_vs_dynamic_speedup': (dynamic['seconds'] / static_graph['seconds']) if static_graph.get('seconds') else None,
            'graph_vs_static_eager_speedup': (static_eager['seconds'] / static_graph['seconds']) if static_graph.get('seconds') else None,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        ctx.__exit__(None, None, None)


if __name__ == '__main__':
    main()
