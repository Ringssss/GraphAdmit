#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.morspec_loader import extract_prompt


def load_morspec_texts(data_dir):
    texts = []
    for path in sorted(Path(data_dir).glob('*.jsonl')):
        with path.open(encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    texts.append(extract_prompt(json.loads(line)))
    if not texts:
        raise ValueError(f'No jsonl prompts under {data_dir}')
    return texts


def load_flowprefill(trace_path, limit, offset, types, max_input_len):
    rows = [json.loads(line) for line in open(trace_path, encoding='utf-8') if line.strip()]
    block_size = 16
    for item in rows:
        father_id = item.get('parent_chat_id', -1)
        if father_id != -1 and item.get('type') == 'text' and 0 <= father_id < len(rows):
            father = rows[father_id]
            item['original_input_length'] = item['input_length']
            item['input_length'] = max(1, (len(item.get('hash_ids', [])) - len(father.get('hash_ids', []))) * block_size)
    if types:
        allow = set(types.split(','))
        rows = [r for r in rows if r.get('type') in allow]
    if max_input_len:
        rows = [r for r in rows if r['input_length'] <= max_input_len]
    rows = rows[offset: offset + limit]
    return rows


def build_prompt_to_length(tokenizer, texts, target_len, cursor):
    sep = '\n\n---\n\n'
    pieces = []
    idx = cursor
    while True:
        pieces.append(texts[idx % len(texts)])
        prompt = sep.join(pieces)
        ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        if len(ids) >= target_len:
            ids = ids[:target_len]
            return tokenizer.decode(ids, skip_special_tokens=True), idx + 1, len(ids)
        idx += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', default='external/FlowPrefill/trace_build/qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl')
    ap.add_argument('--morspec-dir', default='/home/zhujianian/morspec/data')
    ap.add_argument('--model', default='/mnt/models/Meta-Llama-3-8B-Instruct')
    ap.add_argument('--limit', type=int, default=32)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--types', default='text,search,file')
    ap.add_argument('--max-input-len', type=int, default=4096)
    ap.add_argument('--min-input-len', type=int, default=1)
    ap.add_argument('--output', default='results/flowprefill_morspec_workload.json')
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    texts = load_morspec_texts(args.morspec_dir)
    rows = load_flowprefill(args.trace, args.limit * 4, args.offset, args.types, args.max_input_len)
    rows = [r for r in rows if r['input_length'] >= args.min_input_len][:args.limit]
    cursor = 0
    out_rows = []
    lens = []
    for i, row in enumerate(rows):
        target = int(row['input_length'])
        prompt, cursor, actual = build_prompt_to_length(tokenizer, texts, target, cursor)
        lens.append(actual)
        out_rows.append({
            'idx': i,
            'chat_id': row.get('chat_id'),
            'parent_chat_id': row.get('parent_chat_id'),
            'timestamp': row.get('timestamp'),
            'type': row.get('type'),
            'turn': row.get('turn'),
            'target_input_length': target,
            'actual_input_length': actual,
            'output_length': int(row.get('output_length', 1)),
            'prompt': prompt,
        })
    result = {
        'source_trace': args.trace,
        'morspec_dir': args.morspec_dir,
        'model': args.model,
        'num_requests': len(out_rows),
        'types': args.types,
        'max_input_len': args.max_input_len,
        'length_summary': {
            'min': int(np.min(lens)) if lens else 0,
            'p50': float(np.percentile(lens, 50)) if lens else 0,
            'p90': float(np.percentile(lens, 90)) if lens else 0,
            'p95': float(np.percentile(lens, 95)) if lens else 0,
            'p99': float(np.percentile(lens, 99)) if lens else 0,
            'max': int(np.max(lens)) if lens else 0,
            'gt512': int(sum(x > 512 for x in lens)),
            'gt1024': int(sum(x > 1024 for x in lens)),
            'gt2048': int(sum(x > 2048 for x in lens)),
        },
        'requests': out_rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in result if k != 'requests'}, indent=2, ensure_ascii=False))
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
