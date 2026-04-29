#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RANGES = [(0,512),(512,1024),(1024,2048),(2048,4096),(4096,8192),(8192,32768)]


def rng(tok):
    for lo, hi in RANGES:
        if lo < tok <= hi:
            return f"({lo},{hi}]"
    return f">{RANGES[-1][1]}"


def action_name(config):
    if config.startswith('1.'):
        return 'eager'
    if config.startswith('2.'):
        return 'vllm_default_graph_or_compile'
    if config.startswith('3.'):
        return 'ours_candidate_graph'
    if config.startswith('4.'):
        return 'chunked_prefill_graph'
    if config.startswith('5.'):
        return 'ours_candidate_graph_plus_chunked_prefill'
    return config


def percentile(vals, p):
    return float(np.percentile(np.array(vals, dtype=np.float64), p)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--calib-n', type=int, default=16)
    ap.add_argument('--margin-pct', type=float, default=0.0)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding='utf-8'))
    results = data['results']
    n = len(results[0]['per_req'])
    calib_n = min(args.calib_n, max(1, n // 2))
    eval_indices = list(range(calib_n, n)) or list(range(n))

    per_action = {}
    for result in results:
        per_action[action_name(result['config'])] = result['per_req']

    calib = defaultdict(lambda: defaultdict(list))
    for action, rows in per_action.items():
        for i in range(calib_n):
            item = rows[i]
            calib[rng(item['tok'])][action].append(item['ms'])

    policy = {}
    for range_key, action_vals in calib.items():
        means = {a: float(np.mean(v)) for a, v in action_vals.items() if v}
        best_action = min(means, key=means.get)
        best_mean = means[best_action]
        baseline = means.get('vllm_default_graph_or_compile')
        if baseline is not None and best_mean > baseline * (1.0 - args.margin_pct / 100.0):
            best_action = 'vllm_default_graph_or_compile'
            best_mean = baseline
        policy[range_key] = {'action': best_action, 'calib_avg_ms': best_mean, 'all_calib_avg_ms': means}

    eval_rows = []
    for i in eval_indices:
        tok = per_action[next(iter(per_action))][i]['tok']
        range_key = rng(tok)
        chosen = policy.get(range_key, {'action': 'vllm_default_graph_or_compile'})['action']
        chosen_ms = per_action[chosen][i]['ms']
        row = {'idx': i, 'tok': tok, 'range': range_key, 'chosen_action': chosen, 'chosen_ms': chosen_ms}
        for action, rows in per_action.items():
            row[action] = rows[i]['ms']
        eval_rows.append(row)

    def stats_for(action):
        vals = [r[action] for r in eval_rows]
        return {'avg_ms': float(np.mean(vals)), 'p50_ms': percentile(vals, 50), 'p95_ms': percentile(vals, 95), 'p99_ms': percentile(vals, 99)}

    chosen_vals = [r['chosen_ms'] for r in eval_rows]
    summary = {
        'policy': policy,
        'source': args.input,
        'calib_n': calib_n,
        'eval_n': len(eval_rows),
        'policy_stats': {'avg_ms': float(np.mean(chosen_vals)), 'p50_ms': percentile(chosen_vals, 50), 'p95_ms': percentile(chosen_vals, 95), 'p99_ms': percentile(chosen_vals, 99)},
        'baseline_stats': {action: stats_for(action) for action in per_action},
        'speedup_vs_vllm_default_avg': stats_for('vllm_default_graph_or_compile')['avg_ms'] / float(np.mean(chosen_vals)),
        'speedup_vs_chunked_avg': stats_for('chunked_prefill_graph')['avg_ms'] / float(np.mean(chosen_vals)) if 'chunked_prefill_graph' in per_action else None,
        'eval_rows': eval_rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'eval_rows'}, indent=2))
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
