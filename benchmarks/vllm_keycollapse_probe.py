#!/usr/bin/env python3
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def grouped_requests(workload):
    groups = defaultdict(list)
    for req in workload['requests']:
        groups[req.get('group_id', str(req.get('id')))].append(req)
    return list(groups.items())


def make_prompt(req, use_token_ids=True):
    token_ids = req.get('prompt_token_ids')
    if use_token_ids and token_ids:
        return {'prompt_token_ids': [int(x) for x in token_ids]}
    return req['prompt']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', default='results/keycollapse_workload.json')
    parser.add_argument('--model', default='/mnt/models/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--tp-size', type=int, default=1)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.8)
    parser.add_argument('--max-tokens', type=int, default=1)
    parser.add_argument('--cudagraph-mode', default='FULL_AND_PIECEWISE',
                        choices=['FULL_AND_PIECEWISE', 'FULL', 'PIECEWISE'])
    parser.add_argument('--capture-sizes', default='1,2,4,8,16,24,32,40,48,56,64,80,96,112,128,160,192,224,256,320,384,448,512,640,768,896,1024,1280,1536,1792,2048')
    parser.add_argument('--profile-prefix', default='results/vllm_keycollapse_profile')
    parser.add_argument('--limit-groups', type=int, default=None)
    parser.add_argument('--use-token-ids', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--enable-return-routed-experts', action='store_true',
                        help='enable vLLM MoE routed-expert capture so STATICITY_VLLM_MOE_PROFILE has events')
    parser.add_argument('--output', default='results/vllm_keycollapse_probe.json')
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    workload = json.loads(Path(args.workload).read_text(encoding='utf-8'))
    groups = grouped_requests(workload)
    if args.limit_groups:
        groups = groups[:args.limit_groups]

    dispatch_log = f'{args.profile_prefix}_dispatcher.jsonl'
    runner_log = f'{args.profile_prefix}_runner.jsonl'
    attn_log = f'{args.profile_prefix}_attn.jsonl'
    moe_log = f'{args.profile_prefix}_moe.jsonl'
    Path(dispatch_log).parent.mkdir(parents=True, exist_ok=True)
    Path(dispatch_log).write_text('')
    Path(runner_log).write_text('')
    Path(attn_log).write_text('')
    Path(moe_log).write_text('')
    os.environ['STATICITY_VLLM_CG_PROFILE'] = dispatch_log
    os.environ['VLLM_CG_TRACE_FILE'] = dispatch_log
    os.environ['STATICITY_VLLM_RUNNER_PROFILE'] = runner_log
    os.environ['STATICITY_VLLM_ATTN_PROFILE'] = attn_log
    os.environ['STATICITY_VLLM_MOE_PROFILE'] = moe_log

    capture_sizes = [int(x) for x in args.capture_sizes.split(',') if x]
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        disable_log_stats=True,
        enable_return_routed_experts=args.enable_return_routed_experts,
        enable_chunked_prefill=False,
        max_num_batched_tokens=args.max_model_len,
        compilation_config={
            'cudagraph_mode': args.cudagraph_mode,
            'cudagraph_capture_sizes': capture_sizes,
            'max_cudagraph_capture_size': max(capture_sizes),
        },
    )
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    rows = []
    try:
        warmup_prompts = [make_prompt(groups[0][1][0], args.use_token_ids)]
        _ = llm.generate(warmup_prompts, sp)
        for group_id, reqs in groups:
            prompts = [make_prompt(req, args.use_token_ids) for req in reqs]
            lengths = [int(req['actual_input_length']) for req in reqs]
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = llm.generate(prompts, sp)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - start) * 1000
            output_token_ids = [
                list(out.outputs[0].token_ids) if out.outputs else []
                for out in outputs
            ]
            output_texts = [
                out.outputs[0].text if out.outputs else ''
                for out in outputs
            ]
            rows.append({
                'group_id': group_id,
                'num_reqs': len(reqs),
                'target_total_tokens': sum(lengths),
                'lengths': lengths,
                'latency_ms': ms,
                'output_count': len(outputs),
                'output_token_ids': output_token_ids,
                'output_texts': output_texts,
            })
            print(f'{group_id}: num_reqs={len(reqs)} total={sum(lengths)} latency={ms:.2f} ms')
    finally:
        del llm
        os.environ.pop('STATICITY_VLLM_CG_PROFILE', None)
        os.environ.pop('VLLM_CG_TRACE_FILE', None)
        os.environ.pop('STATICITY_VLLM_RUNNER_PROFILE', None)
        os.environ.pop('STATICITY_VLLM_ATTN_PROFILE', None)
        os.environ.pop('STATICITY_VLLM_MOE_PROFILE', None)

    result = {
        'workload': args.workload,
        'model': args.model,
        'use_token_ids': args.use_token_ids,
        'cudagraph_mode': args.cudagraph_mode,
        'capture_sizes': capture_sizes,
        'dispatcher_profile': dispatch_log,
        'runner_profile': runner_log,
        'attention_profile': attn_log,
        'moe_profile': moe_log,
        'rows': rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
