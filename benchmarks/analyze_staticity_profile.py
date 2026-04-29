#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gap-profile', default='results/staticity_gap_profile.json')
    parser.add_argument('--key-profile')
    parser.add_argument('--attention-profile')
    parser.add_argument('--moe-profile')
    parser.add_argument('--scheduler-profile')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    report = {
        'readiness': {},
        'blockers': [],
        'next_actions': [],
    }
    if Path(args.gap_profile).exists():
        gap = load(args.gap_profile)
        for profile in gap.get('profiles', []):
            kind = profile.get('kind')
            if kind == 'vllm':
                report['blockers'].append('vLLM still needs runtime graph-key evidence and metadata arena integration')
            if kind == 'dinfer' and profile.get('all_same_tokens') is False:
                report['blockers'].append('dInfer unsafe graph changes tokens; correctness-aware admission is mandatory')
            if kind == 'dinfer' and profile.get('validate_replay') and (profile.get('total_speedup') or 0) < 1.0:
                report['blockers'].append('dInfer validated graph is slower; capture/fallback amortization must improve')
    if args.key_profile and Path(args.key_profile).exists():
        key = load(args.key_profile)
        diagnosis = key.get('diagnosis', {})
        report['readiness']['vllm_key_profile'] = diagnosis
        if diagnosis.get('same_token_multiple_layout_candidates', 0) > 0:
            report['next_actions'].append('implement metadata arena/key collapse for same-token multi-layout candidates')
        if diagnosis.get('over_max_capture_count', 0) > 0:
            report['next_actions'].append('route over-max captures to CP/compile/eager instead of graph')
        if diagnosis.get('no_matching_key_count', 0) > 0:
            report['next_actions'].append('inspect missing descriptors before adding capture sizes')
    if args.attention_profile and Path(args.attention_profile).exists():
        attn = load(args.attention_profile)
        diagnosis = attn.get('diagnosis', {})
        report['readiness']['vllm_attention_profile'] = diagnosis
        if diagnosis.get('dynamic_tensor_records', 0) <= 4:
            report['blockers'].append('vLLM dense attention metadata is already mostly canonical; do not spend P0 on generic attention arenas')
            report['next_actions'].append('shift vLLM profiling to prefix-cache, mixed batch, and MoE routing metadata')
    if args.moe_profile and Path(args.moe_profile).exists():
        moe = load(args.moe_profile)
        diagnosis = moe.get('diagnosis', {})
        report['readiness']['vllm_moe_profile'] = diagnosis
        if not diagnosis.get('has_moe_events'):
            report['blockers'].append('vLLM MoE routing profile has no events; model/run did not exercise MoE capturer')
        else:
            if diagnosis.get('layers_with_count_dynamicity', 0) > 0:
                report['next_actions'].append('design expert-count capacity buckets and fallback for rare routing templates')
            if diagnosis.get('layers_with_address_dynamicity', 0) > 0:
                report['next_actions'].append('design persistent topk/expert-count/permutation metadata arena')
    if args.scheduler_profile and Path(args.scheduler_profile).exists():
        scheduler = load(args.scheduler_profile)
        diagnosis = scheduler.get('diagnosis', {})
        report['readiness']['template_scheduler'] = diagnosis
        if diagnosis.get('scheduler_staticity_available'):
            report['next_actions'].append('turn offline template grouping into bounded-wait scheduler experiment with TTFT accounting')
        else:
            report['blockers'].append('template scheduler simulation found little batching/staticity opportunity in this trace slice')
    report['next_actions'].extend([
        'run vllm_keycollapse_probe on GPU and summarize dispatcher/runner logs',
        'run vLLM MoE/prefix/mixed-batch profiling before implementing another dense attention arena',
        'run dInfer with --validation-mode decoded and periodic validation before claiming admission speedups',
    ])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
