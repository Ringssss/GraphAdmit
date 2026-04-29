#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.morspec_loader import load_morspec_prompts, wrap_llada_prompt
from benchmarks.llada2_dinfer_kvcache_graph_generate import init_vllm, run_generate, tensor_hash

DINFER_ROOT = Path('/home/zhujianian/eurosys/dInfer')
if str(DINFER_ROOT / 'python') not in sys.path:
    sys.path.insert(0, str(DINFER_ROOT / 'python'))
from dinfer.model import LLaDA2MoeModelLM


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/mnt/models/LLaDA2.0-mini')
    parser.add_argument('--dataset', default='/home/zhujianian/morspec/data/gsm8k.jsonl')
    parser.add_argument('--num-samples', type=int, default=8)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--gen-length', type=int, default=64)
    parser.add_argument('--block-length', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=0.99)
    parser.add_argument('--warmups', type=int, default=0)
    parser.add_argument('--validate-replay', action='store_true')
    parser.add_argument('--port', type=int, default=46301)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    prompts, _ = load_morspec_prompts(args.dataset, args.num_samples, args.offset)
    ctx = init_vllm(rank=0, world_size=1, port=args.port, enable_ep=False)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        model = LLaDA2MoeModelLM(config=config).eval()
        load_start = sync_time()
        model.load_weights(args.model, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        load_s = sync_time() - load_start
        mask_id = 156895
        eos_id = 156892
        rows = []
        for idx, prompt in enumerate(prompts):
            text = wrap_llada_prompt(prompt)
            input_ids = torch.tensor(tokenizer(text)['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
            eager = run_generate(model, input_ids, args.gen_length, args.block_length, args.threshold, mask_id, eos_id,
                                 False, False, args.warmups, False, 1, 15)
            graph = run_generate(model, input_ids, args.gen_length, args.block_length, args.threshold, mask_id, eos_id,
                                 False, True, args.warmups, False, 1, 15, args.validate_replay)
            same = torch.equal(eager['output_ids'], graph['output_ids'])
            row = {
                'idx': idx,
                'prompt_len': int(input_ids.shape[1]),
                'eager_s': eager['seconds'],
                'graph_s': graph['seconds'],
                'speedup': eager['seconds'] / graph['seconds'] if graph['seconds'] > 0 else None,
                'same_tokens': same,
                'eager_nfe': eager['nfe'],
                'graph_nfe': graph['nfe'],
                'eager_hash': tensor_hash(eager['output_ids']),
                'graph_hash': tensor_hash(graph['output_ids']),
                'graph_stats': graph['stats'],
            }
            rows.append(row)
            print(f"[{idx+1}/{len(prompts)}] len={row['prompt_len']} eager={row['eager_s']:.3f}s graph={row['graph_s']:.3f}s speedup={row['speedup']:.2f} same={same}")
        eager_total = sum(r['eager_s'] for r in rows)
        graph_total = sum(r['graph_s'] for r in rows)
        result = {
            'model': args.model,
            'dataset': args.dataset,
            'num_samples': len(rows),
            'gen_length': args.gen_length,
            'block_length': args.block_length,
            'threshold': args.threshold,
            'validate_replay': args.validate_replay,
            'load_s': load_s,
            'eager_total_s': eager_total,
            'graph_total_s': graph_total,
            'total_speedup': eager_total / graph_total if graph_total > 0 else None,
            'all_same_tokens': all(r['same_tokens'] for r in rows),
            'rows': rows,
        }
        out = Path(args.output) if args.output else Path('results') / f"dinfer_morspec_{Path(args.dataset).stem}_{len(rows)}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps({k: result[k] for k in result if k != 'rows'}, indent=2, ensure_ascii=False))
        print(f'Saved to {out}')
    finally:
        ctx.__exit__(None, None, None)


if __name__ == '__main__':
    main()
