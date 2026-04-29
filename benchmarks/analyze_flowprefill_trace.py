#!/usr/bin/env python3
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def load(path, adjust_child_text=True):
    rows=[json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]
    if adjust_child_text:
        block_size=16
        for item in rows:
            father_id=item.get('parent_chat_id', -1)
            if father_id != -1 and item.get('type') == 'text' and 0 <= father_id < len(rows):
                father=rows[father_id]
                item['original_input_length']=item['input_length']
                item['input_length']=max(0, (len(item.get('hash_ids', []))-len(father.get('hash_ids', []))) * block_size)
    return rows


def pct(arr, ps=(0,25,50,75,90,95,99,99.9,100)):
    if len(arr)==0: return {}
    a=np.asarray(arr, dtype=float)
    return {str(p): float(np.percentile(a,p)) for p in ps}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trace', default='external/FlowPrefill/trace_build/qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl')
    ap.add_argument('--output', default='results/flowprefill_traceA_analysis.json')
    args=ap.parse_args()
    rows=load(args.trace)
    inputs=[r['input_length'] for r in rows]
    outputs=[r['output_length'] for r in rows]
    timestamps=[r['timestamp'] for r in rows]
    types=Counter(r.get('type','unknown') for r in rows)
    turns=Counter(r.get('turn',0) for r in rows)
    parent_count=sum(1 for r in rows if r.get('parent_chat_id',-1)!=-1)
    by_type={}
    for typ in types:
        sub=[r for r in rows if r.get('type')==typ]
        by_type[typ]={
            'count': len(sub),
            'input_length': pct([r['input_length'] for r in sub]),
            'output_length': pct([r['output_length'] for r in sub]),
            'turn': pct([r.get('turn',0) for r in sub]),
        }
    arrivals=np.diff(np.asarray(sorted(timestamps))) if len(timestamps)>1 else np.asarray([])
    buckets={
        '<=512': sum(x<=512 for x in inputs),
        '512-2k': sum(512<x<=2048 for x in inputs),
        '2k-4k': sum(2048<x<=4096 for x in inputs),
        '4k-8k': sum(4096<x<=8192 for x in inputs),
        '>8k': sum(x>8192 for x in inputs),
    }
    result={
        'trace': args.trace,
        'count': len(rows),
        'duration_s': max(timestamps)-min(timestamps) if timestamps else 0,
        'request_rate_per_s': len(rows)/(max(timestamps)-min(timestamps)) if len(timestamps)>1 else None,
        'type_counts': dict(types),
        'turn_counts_top': dict(turns.most_common(20)),
        'multi_turn_requests': parent_count,
        'multi_turn_fraction': parent_count/len(rows) if rows else 0,
        'input_length': pct(inputs),
        'output_length': pct(outputs),
        'interarrival_s': pct(arrivals.tolist()),
        'input_buckets': buckets,
        'by_type': by_type,
        'first_rows_compact': [{k:r.get(k) for k in ['chat_id','parent_chat_id','timestamp','input_length','output_length','type','turn']} for r in rows[:10]],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2)[:6000])
    print(f'Saved to {args.output}')

if __name__=='__main__': main()
