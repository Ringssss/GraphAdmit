#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


RANGES = [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 32768)]


def range_key(tokens):
    for lo, hi in RANGES:
        if lo < tokens <= hi:
            return f"({lo},{hi}]"
    return f">{RANGES[-1][1]}"


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def stats(values):
    if not values:
        return {'n': 0, 'avg': None, 'p50': None, 'p95': None, 'p99': None}
    return {
        'n': len(values),
        'avg': float(sum(values) / len(values)),
        'p50': percentile(values, 50),
        'p95': percentile(values, 95),
        'p99': percentile(values, 99),
    }


def action_name(config):
    if config.startswith('1.'):
        return 'eager'
    if config.startswith('2.'):
        return 'vllm_default_graph_or_compile'
    if config.startswith('3.'):
        return 'ours_candidate_graph'
    if config.startswith('4.'):
        return 'chunked_prefill_graph'
    return config


def profile_vllm(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    actions = {action_name(r['config']): r for r in data['results']}
    first = next(iter(actions.values()))
    token_ranges = Counter(range_key(row['tok']) for row in first['per_req'])
    per_range = {}
    for rng in token_ranges:
        per_range[rng] = {'n': token_ranges[rng], 'actions': {}}
        indices = [i for i, row in enumerate(first['per_req']) if range_key(row['tok']) == rng]
        for action, result in actions.items():
            values = [result['per_req'][i]['ms'] for i in indices]
            per_range[rng]['actions'][action] = stats(values)
        default_avg = per_range[rng]['actions'].get('vllm_default_graph_or_compile', {}).get('avg')
        if default_avg:
            for action, action_stats in per_range[rng]['actions'].items():
                avg = action_stats.get('avg')
                action_stats['speedup_vs_vllm_default'] = default_avg / avg if avg else None
        ranked = sorted(
            ((a, s['avg']) for a, s in per_range[rng]['actions'].items() if s['avg'] is not None),
            key=lambda item: item[1],
        )
        per_range[rng]['best_action'] = ranked[0][0] if ranked else None

    planner = data.get('planner', {})
    dp_count = len(planner.get('dp_candidate_bucket_sizes') or [])
    safe_count = len(planner.get('bucket_sizes') or [])
    workload_stats = data.get('workload_stats', {})
    long_frac = (workload_stats.get('gt512', 0) / workload_stats.get('n', 1)) if workload_stats.get('n') else None
    diagnoses = []
    if planner.get('mode') == 'safe' and safe_count == 51:
        diagnoses.append('safe mode is equivalent to vLLM default small graph capture sizes; it is a guardrail, not a differentiated staticity mechanism')
    if dp_count and planner.get('dp_expected_hit_rate') == 1.0:
        diagnoses.append('DP token-coverage planner predicts full hit rate, but measured latency shows long ranges are not always profitable')
    if long_frac and long_frac > 0.5:
        diagnoses.append('most requests exceed 512 tokens; capture-size-only changes cannot recover graph coverage without key/metadata canonicalization or CP')

    return {
        'kind': 'vllm',
        'source': str(path),
        'model': data.get('model'),
        'workload': data.get('workload'),
        'workload_stats': workload_stats,
        'planner_summary': {
            'mode': planner.get('mode'),
            'safe_bucket_count': safe_count,
            'dp_candidate_bucket_count': dp_count,
            'dp_expected_hit_rate': planner.get('dp_expected_hit_rate'),
            'dp_expected_padding_waste_pct': planner.get('dp_expected_padding_waste_pct'),
            'dp_total_graph_memory_mb': planner.get('dp_total_graph_memory_mb'),
            'dp_total_warmup_time_s': planner.get('dp_total_warmup_time_s'),
        },
        'per_range': per_range,
        'diagnoses': diagnoses,
        'missing_instrumentation': [
            'actual vLLM graph dispatch key fields',
            'graph hit/miss and fallback reason per request',
            'num_reqs/ragged layout/cu_seqlens/slot_mapping family split',
            'graph memory and warmup cost per template',
        ],
        'recommended_next_probes': [
            'same-total-tokens different-num-reqs controlled key-collapse probe',
            'runtime graph-key logger in CudaGraphDispatcher/model runner',
            'metadata arena for cu_seqlens/positions/slot_mapping/block tables',
            'calibrated action planner with CP/compile/eager/graph as peer actions',
        ],
    }


def profile_dinfer(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = data.get('rows', [])
    stats_rows = [row.get('graph_stats', {}) for row in rows]
    totals = defaultdict(float)
    for row_stats in stats_rows:
        for key in ('captures', 'replays', 'eager_forwards', 'validation_fallbacks'):
            totals[key] += row_stats.get(key, 0)
        for key in ('capture_seconds', 'replay_seconds'):
            totals[key] += row_stats.get(key, 0.0)
    same_tokens = [bool(row.get('same_tokens')) for row in rows]
    diagnoses = []
    if data.get('validate_replay') and totals['validation_fallbacks']:
        diagnoses.append('validated graph is dominated by capture plus validation fallback overhead')
    if rows and not all(same_tokens):
        diagnoses.append('unsafe replay changes generated tokens; this cannot be claimed as acceleration')
    if totals['captures'] and totals['replays'] <= totals['captures'] * 2:
        diagnoses.append('templates are not reused enough to amortize capture cost')
    return {
        'kind': 'dinfer',
        'source': str(path),
        'model': data.get('model'),
        'workload': data.get('workload'),
        'num_samples': data.get('num_samples'),
        'validate_replay': data.get('validate_replay'),
        'all_same_tokens': data.get('all_same_tokens'),
        'total_speedup': data.get('total_speedup'),
        'aggregate_graph_stats': dict(totals),
        'per_request': [
            {
                'idx': row.get('idx'),
                'prompt_len': row.get('prompt_len'),
                'speedup': row.get('speedup'),
                'same_tokens': row.get('same_tokens'),
                'graph_stats': row.get('graph_stats'),
            }
            for row in rows
        ],
        'diagnoses': diagnoses,
        'missing_instrumentation': [
            'mask/update-set dynamicity by diffusion step',
            'graph key frequency and disabled-key reasons',
            'top1/token-equivalence validation separate from bit-exact logits',
            'MoE expert routing/count/permutation metadata dynamics',
        ],
        'recommended_next_probes': [
            'token/top1 validation admission before latency claims',
            'fixed-address mask/update/expert metadata arena',
            'per-step template family rather than whole-generation capture',
            'template blacklist with logged semantic mismatch reason',
        ],
    }


def write_markdown(profiles, output):
    lines = ['# Staticity Recovery Gap Profile', '']
    for profile in profiles:
        lines += [f"## {profile['kind']}: `{profile['source']}`", '']
        for diagnosis in profile.get('diagnoses', []):
            lines.append(f'- Diagnosis: {diagnosis}')
        if profile['kind'] == 'vllm':
            lines += ['', '| Range | N | Best action | vLLM avg ms | Ours avg ms | CP avg ms |', '|---|---:|---|---:|---:|---:|']
            for rng, row in profile['per_range'].items():
                actions = row['actions']
                lines.append(
                    f"| {rng} | {row['n']} | {row['best_action']} | "
                    f"{fmt(actions.get('vllm_default_graph_or_compile', {}).get('avg'))} | "
                    f"{fmt(actions.get('ours_candidate_graph', {}).get('avg'))} | "
                    f"{fmt(actions.get('chunked_prefill_graph', {}).get('avg'))} |"
                )
        if profile['kind'] == 'dinfer':
            lines += [
                '',
                f"- Total speedup: {profile.get('total_speedup')}",
                f"- All same tokens: {profile.get('all_same_tokens')}",
                f"- Aggregate graph stats: `{json.dumps(profile.get('aggregate_graph_stats'), sort_keys=True)}`",
            ]
        lines += ['', 'Missing instrumentation:']
        lines += [f'- {item}' for item in profile.get('missing_instrumentation', [])]
        lines += ['', 'Recommended next probes:']
        lines += [f'- {item}' for item in profile.get('recommended_next_probes', [])]
        lines.append('')
    Path(output).write_text('\n'.join(lines), encoding='utf-8')


def fmt(value):
    return '' if value is None else f'{value:.2f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vllm', action='append', default=[])
    parser.add_argument('--dinfer', action='append', default=[])
    parser.add_argument('--output', required=True)
    parser.add_argument('--markdown-output')
    args = parser.parse_args()

    profiles = []
    profiles.extend(profile_vllm(path) for path in args.vllm)
    profiles.extend(profile_dinfer(path) for path in args.dinfer)
    result = {'profiles': profiles}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    if args.markdown_output:
        write_markdown(profiles, args.markdown_output)
    print(json.dumps({'output': str(out), 'num_profiles': len(profiles)}, indent=2))


if __name__ == '__main__':
    main()
