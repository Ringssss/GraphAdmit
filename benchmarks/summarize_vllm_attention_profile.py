#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attention-log', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    events = []
    for line in Path(args.attention_log).read_text(encoding='utf-8').splitlines():
        if line.strip():
            event = json.loads(line)
            if event.get('component') == 'attention_metadata':
                events.append(event)

    by_template = defaultdict(list)
    tensor_shape_variants = defaultdict(Counter)
    tensor_address_variants = defaultdict(Counter)
    for event in events:
        key = (
            event.get('num_tokens_padded'),
            event.get('num_reqs_padded'),
            event.get('max_query_len'),
            event.get('for_cudagraph_capture'),
            event.get('use_spec_decode'),
        )
        by_template[str(key)].append(event)
        for tensor in event.get('tensors', []):
            tensor_key = (str(key), tensor['name'])
            tensor_shape_variants[tensor_key][tuple(tensor['shape'])] += 1
            tensor_address_variants[tensor_key][tensor['data_ptr']] += 1

    templates = []
    for key, rows in by_template.items():
        tensor_rows = []
        names = sorted({t['name'] for row in rows for t in row.get('tensors', [])})
        for name in names:
            tensor_key = (key, name)
            tensor_rows.append({
                'name': name,
                'shape_variants': len(tensor_shape_variants[tensor_key]),
                'address_variants': len(tensor_address_variants[tensor_key]),
                'top_shapes': [[list(shape), count] for shape, count in tensor_shape_variants[tensor_key].most_common(5)],
                'top_addresses': tensor_address_variants[tensor_key].most_common(5),
            })
        templates.append({
            'template': key,
            'events': len(rows),
            'tensor_dynamics': tensor_rows,
        })
    templates.sort(key=lambda row: (-row['events'], row['template']))

    hot_dynamic_tensors = []
    for row in templates:
        for tensor in row['tensor_dynamics']:
            if tensor['shape_variants'] > 1 or tensor['address_variants'] > 1:
                hot_dynamic_tensors.append({
                    'template': row['template'],
                    'events': row['events'],
                    **tensor,
                })
    result = {
        'attention_log': args.attention_log,
        'num_events': len(events),
        'num_templates': len(templates),
        'templates': templates[:100],
        'hot_dynamic_tensors': hot_dynamic_tensors[:100],
        'diagnosis': {
            'templates_with_dynamic_tensors': len({x['template'] for x in hot_dynamic_tensors}),
            'dynamic_tensor_records': len(hot_dynamic_tensors),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('num_events', 'num_templates', 'diagnosis')}, indent=2))


if __name__ == '__main__':
    main()
