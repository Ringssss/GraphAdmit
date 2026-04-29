#!/usr/bin/env python3
import json
from pathlib import Path

paths = {
    'vllm_prefill_v4': Path('results/v4_Meta-Llama-3-8B-Instruct_conv_100.json'),
    'vllm_ragged_full': Path('results/ragged_full_e2e.json'),
    'llada_forward_probe': Path('results/llada2_dinfer_graph_probe_gen64.json'),
    'llada_kvcache_graph': Path('results/llada2_dinfer_kvcache_graph_gen64_t099_fullbuf.json'),
}

for name, path in paths.items():
    if not path.exists():
        print(f'MISSING {name}: {path}')
        continue
    data = json.loads(path.read_text())
    print(f'## {name} ({path})')
    if name == 'vllm_prefill_v4':
        rows = data if isinstance(data, list) else data.items()
        if isinstance(data, list):
            for row in data:
                label = row.get('config') or row.get('name') or row.get('label')
                print(label, 'avg', row.get('avg_ttft_ms', row.get('avg')), 'p95', row.get('p95_ttft_ms', row.get('p95')), 'p99', row.get('p99_ttft_ms', row.get('p99')))
        else:
            for key, val in data.items():
                if isinstance(val, dict) and 'avg_ttft_ms' in val:
                    print(key, 'avg', round(val['avg_ttft_ms'], 3), 'p95', round(val.get('p95_ttft_ms', 0), 3), 'p99', round(val.get('p99_ttft_ms', 0), 3))
    elif name == 'vllm_ragged_full':
        if isinstance(data, list):
            for row in data:
                rows = row.get('rows', [])
                avg_batch = sum(r['batch_ms'] for r in rows) / len(rows) if rows else None
                print(row.get('mode'), 'collapse', row.get('collapse'), 'ok', row.get('ok'), 'avg_batch_ms', avg_batch)
        else:
            print(json.dumps(data.get('summary', data), indent=2)[:4000])
    elif name == 'llada_forward_probe':
        graph_key = 'static_graph' if 'static_graph' in data else ('graph_replay' if 'graph_replay' in data else 'static_cuda_graph_max_window')
        speedups = data.get('speedups', {})
        speedup = speedups.get('graph_vs_dynamic_eager') or speedups.get('graph_replay_vs_dynamic_eager') or data.get('graph_vs_dynamic_speedup')
        print('dynamic_eager_s', data['dynamic_eager']['seconds'])
        print('graph_replay_s', data[graph_key]['seconds'])
        print('speedup', speedup)
        print('same_tokens', data.get('same_tokens'))
    elif name == 'llada_kvcache_graph':
        print('eager_s', data['eager_dinfer']['seconds'], 'nfe', data['eager_dinfer']['nfe'])
        print('graph_s', data['graph_dinfer']['seconds'], 'nfe', data['graph_dinfer']['nfe'])
        print('same_tokens', data['same_tokens'])
        print('total_speedup', data['speedup_total'])
        print('stats', data['graph_dinfer']['stats'])
