#!/usr/bin/env python3
import json
from pathlib import Path
files = [
    Path('results/dinfer_morspec_gsm8k_4.json'),
    Path('results/dinfer_morspec_gsm8k_4_validated.json'),
    Path('results/vllm_morspec_gsm8k_8_offline.json'),
    Path('results/vllm_morspec_gsm8k_8_server_default.json'),
]
for p in files:
    if not p.exists():
        print('MISSING', p); continue
    d=json.loads(p.read_text())
    print('\n##', p)
    if 'eager_total_s' in d:
        print('dInfer samples', d['num_samples'], 'eager_total_s', d['eager_total_s'], 'graph_total_s', d['graph_total_s'], 'speedup', d['total_speedup'], 'all_same', d['all_same_tokens'], 'validate', d.get('validate_replay'))
        for r in d['rows']:
            print(' ', r['idx'], 'len', r['prompt_len'], 'speedup', round(r['speedup'],3), 'same', r['same_tokens'], 'nfe', r['eager_nfe'], r['graph_nfe'])
    elif 'results' in d:
        print('vLLM offline samples', d['num_samples'], 'lens_p50', sorted(d['lens'])[len(d['lens'])//2])
        for r in d['results']:
            print(' ', r['config'], 'avg_ms', round(r['avg_ms'],2), 'p95', round(r['p95_ms'],2), 'p99', round(r['p99_ms'],2), 'speedup', round(r['speedup_vs_eager'],2), 'init_s', round(r['init_s'],1))
        print(' planner', d['planner'])
    elif 'avg_ms' in d:
        print('vLLM server samples', d['num_samples'], 'avg_ms', d['avg_ms'])
        print(' rows_ms', [round(r['seconds']*1000,2) for r in d['rows']])
