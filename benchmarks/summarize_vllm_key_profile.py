#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_descriptor(text):
    fields = {}
    if not text:
        return fields
    for name in ('num_tokens', 'num_reqs', 'uniform', 'has_lora'):
        match = re.search(rf'{name}=([^,)]+)', text)
        if match:
            raw = match.group(1)
            if raw in ('True', 'False'):
                fields[name] = raw == 'True'
            elif raw == 'None':
                fields[name] = None
            else:
                try:
                    fields[name] = int(raw)
                except ValueError:
                    fields[name] = raw
    return fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dispatcher-log', required=True)
    parser.add_argument('--runner-log')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    dispatch_events = []
    for line in Path(args.dispatcher_log).read_text(encoding='utf-8').splitlines():
        if line.strip():
            event = json.loads(line)
            event['descriptor_fields'] = event.get('batch_descriptor') or parse_descriptor(event.get('result_descriptor'))
            dispatch_events.append(event)

    runner_events = []
    if args.runner_log and Path(args.runner_log).exists():
        for line in Path(args.runner_log).read_text(encoding='utf-8').splitlines():
            if line.strip():
                event = json.loads(line)
                event['descriptor_fields'] = event.get('batch_descriptor') or parse_descriptor(event.get('result_descriptor'))
                runner_events.append(event)

    reason_counts = Counter(str(event.get('reason')) for event in dispatch_events)
    mode_counts = Counter(str(event.get('result_mode') or event.get('mode')) for event in dispatch_events)
    descriptor_counts = Counter(json.dumps(event.get('descriptor_fields') or event.get('result_descriptor'), sort_keys=True) for event in dispatch_events)
    token_to_num_reqs = defaultdict(Counter)
    token_to_layouts = defaultdict(Counter)
    for event in runner_events:
        if event.get('component') != 'gpu_model_runner_padding':
            continue
        fields = event.get('descriptor_fields', {})
        num_tokens = fields.get('num_tokens') or event.get('num_tokens_after_sp_padding') or event.get('num_tokens_after_dp_padding')
        num_reqs = event.get('num_reqs')
        scheduled = tuple(event.get('num_scheduled_tokens') or [])
        token_to_num_reqs[num_tokens][num_reqs] += 1
        token_to_layouts[num_tokens][scheduled] += 1

    collapse_candidates = []
    for num_tokens, req_counter in token_to_num_reqs.items():
        layout_count = len(token_to_layouts[num_tokens])
        if len(req_counter) > 1 or layout_count > 1:
            collapse_candidates.append({
                'num_tokens': num_tokens,
                'num_reqs_values': dict(req_counter),
                'layout_count': layout_count,
                'total_events': sum(req_counter.values()),
            })
    collapse_candidates.sort(key=lambda row: (-row['layout_count'], row['num_tokens']))

    result = {
        'dispatcher_log': args.dispatcher_log,
        'runner_log': args.runner_log,
        'num_dispatch_events': len(dispatch_events),
        'num_runner_events': len(runner_events),
        'reason_counts': dict(reason_counts),
        'mode_counts': dict(mode_counts),
        'unique_result_descriptors': len(descriptor_counts),
        'top_result_descriptors': descriptor_counts.most_common(20),
        'collapse_candidates': collapse_candidates[:100],
        'diagnosis': {
            'no_graph_fraction': mode_counts.get('NONE', 0) / len(dispatch_events) if dispatch_events else None,
            'over_max_capture_count': reason_counts.get('over_max_capture_size', 0) + reason_counts.get('num_tokens_gt_max_capture', 0),
            'no_matching_key_count': reason_counts.get('no_matching_key', 0),
            'same_token_multiple_layout_candidates': len(collapse_candidates),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('num_dispatch_events', 'num_runner_events', 'reason_counts', 'mode_counts', 'diagnosis')}, indent=2))


if __name__ == '__main__':
    main()
