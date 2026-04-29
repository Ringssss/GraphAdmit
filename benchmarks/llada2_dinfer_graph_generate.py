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
from dinfer import BlockIteratorFactory, KVCacheFactory, ThresholdParallelDecoder, BlockDiffusionLLM
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


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


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


def transfer_schedule(block_length: int, steps: int, device):
    base = block_length // steps
    remainder = block_length % steps
    vals = [base + (1 if i < remainder else 0) for i in range(steps)]
    return torch.tensor(vals, dtype=torch.long, device=device)


@torch.no_grad()
def greedy_update_from_logits(x, logits, window, block_length, mask_id, schedule, step):
    block_view = x[:, window - block_length:window]
    active_mask = block_view == mask_id
    active_count = int(active_mask.sum().item())
    if active_count == 0:
        return 0
    active_logits = logits[:, window - block_length:window, :]
    probs = torch.softmax(active_logits.float(), dim=-1)
    x0_p, x0 = torch.max(probs, dim=-1)
    confidence = torch.where(active_mask, x0_p, torch.full_like(x0_p, -torch.inf))
    num_to_transfer = min(int(schedule[step].item()), active_count)
    if num_to_transfer <= 0:
        return 0
    _, idx = torch.topk(confidence[0], k=num_to_transfer)
    block_view[0, idx] = x0[0, idx]
    return num_to_transfer


@torch.no_grad()
def capture_graph(model, static_x, static_attn, static_pos, warmups):
    pool = torch.cuda.graph_pool_handle()
    logits = None
    for _ in range(warmups):
        logits = model(static_x, use_cache=False, attention_mask=static_attn, position_ids=static_pos).logits
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        logits = model(static_x, use_cache=False, attention_mask=static_attn, position_ids=static_pos).logits
    torch.cuda.synchronize()
    return graph, logits


@torch.no_grad()
def run_graph_generate(model, input_ids, gen_length, block_length, steps, mask_id, warmups, graph_mode):
    device = input_ids.device
    x, attn_mask, pos_ids, windows, total_length = build_bd_inputs(input_ids, gen_length, block_length, mask_id)
    max_window = max(windows)
    schedule = transfer_schedule(block_length, steps, device)
    graph_templates = {}
    if graph_mode == 'max_window':
        static_x = x[:, :max_window].clone(memory_format=torch.contiguous_format)
        static_attn = attn_mask[:, :max_window, :max_window].contiguous()
        static_pos = pos_ids[:, :max_window].clone(memory_format=torch.contiguous_format)
        graph, logits = capture_graph(model, static_x, static_attn, static_pos, warmups)
        graph_templates[max_window] = (static_x, graph, logits)
    elif graph_mode == 'per_window':
        for window in windows:
            static_x = x[:, :window].clone(memory_format=torch.contiguous_format)
            static_attn = attn_mask[:, :window, :window].contiguous()
            static_pos = pos_ids[:, :window].clone(memory_format=torch.contiguous_format)
            graph, logits = capture_graph(model, static_x, static_attn, static_pos, warmups)
            graph_templates[window] = (static_x, graph, logits)

    nfe = 0
    transfers = []
    capture_s = 0.0
    replay_s = 0.0
    start = sync_time()
    for window in windows:
        template_window = max_window if graph_mode == 'max_window' else window
        if graph_mode == 'lazy_window':
            static_x = x[:, :window].clone(memory_format=torch.contiguous_format)
            static_attn = attn_mask[:, :window, :window].contiguous()
            static_pos = pos_ids[:, :window].clone(memory_format=torch.contiguous_format)
            capture_start = sync_time()
            graph, logits = capture_graph(model, static_x, static_attn, static_pos, warmups)
            capture_s += sync_time() - capture_start
            graph_templates = {window: (static_x, graph, logits)}
        else:
            static_x, graph, logits = graph_templates[template_window]
        replay_start = sync_time()
        for step in range(steps):
            static_x.copy_(x[:, :template_window])
            graph.replay()
            moved = greedy_update_from_logits(x, logits, window, block_length, mask_id, schedule, step)
            transfers.append(moved)
            nfe += 1
            if moved == 0:
                break
        replay_s += sync_time() - replay_start
    end = sync_time()
    return {
        'seconds': replay_s if graph_mode == 'lazy_window' else end - start,
        'total_wall_seconds': end - start,
        'capture_seconds': capture_s,
        'nfe': nfe,
        'transfers': transfers,
        'output_ids': x[:, input_ids.shape[1]:input_ids.shape[1] + gen_length].clone(),
        'total_ids': x[:, :input_ids.shape[1] + gen_length].clone(),
        'windows': windows,
        'graph_mode': graph_mode,
        'num_templates': len(graph_templates),
    }


@torch.no_grad()
def run_eager_generate(model, input_ids, gen_length, block_length, steps, mask_id):
    x, attn_mask, pos_ids, windows, total_length = build_bd_inputs(input_ids, gen_length, block_length, mask_id)
    schedule = transfer_schedule(block_length, steps, input_ids.device)
    nfe = 0
    transfers = []
    start = sync_time()
    for window in windows:
        for step in range(steps):
            logits = model(
                x[:, :window].clone(memory_format=torch.contiguous_format),
                use_cache=False,
                attention_mask=attn_mask[:, :window, :window],
                position_ids=pos_ids[:, :window].clone(memory_format=torch.contiguous_format),
            ).logits
            moved = greedy_update_from_logits(x, logits, window, block_length, mask_id, schedule, step)
            transfers.append(moved)
            nfe += 1
            if moved == 0:
                break
    end = sync_time()
    return {
        'seconds': end - start,
        'nfe': nfe,
        'transfers': transfers,
        'output_ids': x[:, input_ids.shape[1]:input_ids.shape[1] + gen_length].clone(),
        'total_ids': x[:, :input_ids.shape[1] + gen_length].clone(),
        'windows': windows,
    }


@torch.no_grad()
def run_dinfer_generate(model, input_ids, gen_length, block_length, mask_id, eos_id):
    decoder = ThresholdParallelDecoder(temperature=0, threshold=0.9, mask_id=mask_id, eos_id=eos_id)
    cache_factory = KVCacheFactory('prefix', is_bd_model=True)
    dllm = BlockDiffusionLLM(
        model,
        decoder,
        BlockIteratorFactory(start_block_align=True, use_block_diffusion=True),
        cache_factory=cache_factory,
        early_stop=True,
    )
    dllm.generate(input_ids, gen_length=gen_length, block_length=block_length)
    prev = dllm.num_forwards
    start = sync_time()
    out = dllm.generate(input_ids, gen_length=gen_length, block_length=block_length)
    end = sync_time()
    return {'seconds': end - start, 'nfe': dllm.num_forwards - prev, 'output_ids': out.clone()}


def tensor_hash(tensor):
    flat = tensor.detach().to(torch.int64).flatten()
    if flat.numel() == 0:
        return 0
    weights = torch.arange(1, flat.numel() + 1, device=flat.device, dtype=torch.int64)
    return int(((flat * weights).sum() % 2147483647).item())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/mnt/models/LLaDA2.0-mini')
    parser.add_argument('--prompt', default='Say hi in five words.')
    parser.add_argument('--gen-length', type=int, default=64)
    parser.add_argument('--block-length', type=int, default=16)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--port', type=int, default=46201)
    parser.add_argument('--output', default='results/llada2_dinfer_graph_generate.json')
    parser.add_argument('--run-dinfer', action='store_true')
    parser.add_argument('--graph-mode', choices=['max_window','per_window','lazy_window'], default='per_window')
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
        mask_id = 156895
        eos_id = 156892

        eager = run_eager_generate(model, input_ids, args.gen_length, args.block_length, args.steps, mask_id)
        graph = run_graph_generate(model, input_ids, args.gen_length, args.block_length, args.steps, mask_id, args.warmups, args.graph_mode)
        same_tokens = torch.equal(eager['output_ids'], graph['output_ids'])
        dinfer = run_dinfer_generate(model, input_ids, args.gen_length, args.block_length, mask_id, eos_id) if args.run_dinfer else None

        result = {
            'model': args.model,
            'prompt_len': input_ids.shape[1],
            'gen_length': args.gen_length,
            'block_length': args.block_length,
            'steps': args.steps,
            'load_s': load_s,
            'eager_loop': {
                'seconds': eager['seconds'],
                'nfe': eager['nfe'],
                'transfers': eager['transfers'],
                'hash': tensor_hash(eager['output_ids']),
                'text': tokenizer.decode(eager['output_ids'][0], skip_special_tokens=False),
            },
            'graph_loop': {
                'seconds': graph['seconds'],
                'total_wall_seconds': graph.get('total_wall_seconds'),
                'capture_seconds': graph.get('capture_seconds'),
                'nfe': graph['nfe'],
                'transfers': graph['transfers'],
                'hash': tensor_hash(graph['output_ids']),
                'text': tokenizer.decode(graph['output_ids'][0], skip_special_tokens=False),
                'windows': graph['windows'],
                'graph_mode': graph['graph_mode'],
                'num_templates': graph['num_templates'],
            },
            'same_tokens_eager_vs_graph': bool(same_tokens),
            'graph_speedup_vs_eager': eager['seconds'] / graph['seconds'],
            'dinfer_generate': None if dinfer is None else {
                'seconds': dinfer['seconds'],
                'nfe': dinfer['nfe'],
                'hash': tensor_hash(dinfer['output_ids']),
                'text': tokenizer.decode(dinfer['output_ids'][0], skip_special_tokens=False),
                'graph_speedup_vs_dinfer': dinfer['seconds'] / graph['seconds'],
            },
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    finally:
        ctx.__exit__(None, None, None)


if __name__ == '__main__':
    main()
