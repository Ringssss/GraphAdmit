#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter, deque
from pathlib import Path


def ceil_bucket(value, buckets):
    for bucket in buckets:
        if value <= bucket:
            return bucket
    return buckets[-1]


def request_tokens(req):
    for key in ('actual_input_length', 'target_input_length', 'prompt_len'):
        if key in req:
            return int(req[key])
    raise KeyError('request is missing an input length field')


def request_time(req, idx):
    if 'timestamp' in req:
        return float(req['timestamp']) * 1000.0
    return float(idx)


def simulate(requests, buckets, wait_ms):
    queue = deque()
    now = 0.0
    batches = []
    wait_times = []
    template_hits = Counter()

    def flush_template(template, flush_time):
        selected = []
        kept = deque()
        while queue:
            item = queue.popleft()
            if item['template'] == template:
                selected.append(item)
            else:
                kept.append(item)
        queue.extend(kept)
        if not selected:
            return
        wait_times.extend(flush_time - item['arrival_ms'] for item in selected)
        template_hits[template] += 1
        batches.append({
            'template': template,
            'num_reqs': len(selected),
            'tokens': [item['tokens'] for item in selected],
            'arrival_span_ms': max(item['arrival_ms'] for item in selected) - min(item['arrival_ms'] for item in selected),
            'flush_time_ms': flush_time,
        })

    for idx, req in enumerate(requests):
        arrival = request_time(req, idx)
        now = max(now, arrival)
        while queue and queue[0]['arrival_ms'] + wait_ms <= now:
            flush_template(queue[0]['template'], queue[0]['arrival_ms'] + wait_ms)
        tokens = request_tokens(req)
        template = ceil_bucket(tokens, buckets)
        queue.append({'idx': idx, 'arrival_ms': arrival, 'tokens': tokens, 'template': template})
    while queue:
        flush_template(queue[0]['template'], queue[0]['arrival_ms'] + wait_ms)

    waits_sorted = sorted(wait_times)
    def pct(p):
        if not waits_sorted:
            return 0.0
        rank = (len(waits_sorted) - 1) * p / 100.0
        lo = math.floor(rank)
        hi = min(lo + 1, len(waits_sorted) - 1)
        frac = rank - lo
        return waits_sorted[lo] * (1.0 - frac) + waits_sorted[hi] * frac

    return {
        'wait_ms': wait_ms,
        'num_requests': len(requests),
        'num_batches': len(batches),
        'avg_batch_size': sum(batch['num_reqs'] for batch in batches) / len(batches) if batches else 0.0,
        'max_batch_size': max((batch['num_reqs'] for batch in batches), default=0),
        'unique_templates': len(template_hits),
        'template_reuse': template_hits.most_common(),
        'wait_p50_ms': pct(50),
        'wait_p95_ms': pct(95),
        'wait_p99_ms': pct(99),
        'batches': batches[:200],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--buckets', default='128,256,384,512,768,1024,1536,2048,3072,4096,8192')
    parser.add_argument('--wait-ms', default='0,0.5,1,2')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    data = json.loads(Path(args.workload).read_text(encoding='utf-8'))
    requests = data['requests'][:args.limit]
    buckets = [int(value) for value in args.buckets.split(',') if value]
    waits = [float(value) for value in args.wait_ms.split(',') if value]
    runs = [simulate(requests, buckets, wait) for wait in waits]
    baseline_batches = runs[0]['num_batches'] if runs else None
    for run in runs:
        run['batch_reduction_vs_wait0'] = baseline_batches / run['num_batches'] if baseline_batches and run['num_batches'] else None
    result = {
        'workload': args.workload,
        'buckets': buckets,
        'runs': runs,
        'diagnosis': {
            'best_avg_batch_size': max((run['avg_batch_size'] for run in runs), default=0.0),
            'best_batch_reduction': max((run['batch_reduction_vs_wait0'] or 1.0 for run in runs), default=1.0),
            'scheduler_staticity_available': any(run['avg_batch_size'] > 1.0 for run in runs),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result['diagnosis'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
