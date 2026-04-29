#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_events(path):
    events = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get('component') in ('moe_routed_experts', 'dinfer_llada_moe_gate'):
            events.append(event)
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--moe-log', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    events = load_events(args.moe_log)
    by_layer = defaultdict(list)
    template_counter = Counter()
    metadata_template_counter = Counter()
    capacity_counter = Counter()
    route_hashes = defaultdict(Counter)
    count_hashes = defaultdict(Counter)
    address_variants = defaultdict(Counter)
    address_templates = defaultdict(Counter)
    active_expert_values = defaultdict(Counter)
    token_count_values = defaultdict(Counter)
    max_count_values = defaultdict(Counter)

    for event in events:
        layer = int(event.get('layer_id', -1))
        by_layer[layer].append(event)
        template = (
            event.get('num_tokens'),
            event.get('num_experts_per_tok'),
            event.get('configured_num_experts'),
        )
        template_counter[str(template)] += 1
        metadata_template = event.get('metadata_template_id')
        if metadata_template is not None:
            metadata_template_counter[str(metadata_template)] += 1
        capacity_bucket = event.get('capacity_bucket')
        if capacity_bucket is not None:
            capacity_counter[int(capacity_bucket)] += 1
        route_hashes[layer][event.get('route_hash')] += 1
        count_hashes[layer][event.get('count_hash')] += 1
        address_variants[layer][event.get('data_ptr')] += 1
        address_templates[layer][event.get('address_template_id')] += 1
        active_expert_values[layer][event.get('active_experts')] += 1
        token_count_values[layer][event.get('num_tokens')] += 1
        max_count_values[layer][event.get('max_expert_count')] += 1

    layer_rows = []
    for layer, rows in sorted(by_layer.items()):
        num_events = len(rows)
        unique_route = len(route_hashes[layer])
        unique_counts = len(count_hashes[layer])
        unique_addr = len(address_variants[layer])
        unique_active = len(active_expert_values[layer])
        unique_tokens = len(token_count_values[layer])
        layer_rows.append({
            'layer_id': layer,
            'events': num_events,
            'unique_route_hashes': unique_route,
            'unique_count_hashes': unique_counts,
            'unique_topk_addresses': unique_addr,
            'unique_address_templates': len(address_templates[layer]),
            'unique_active_expert_counts': unique_active,
            'unique_token_counts': unique_tokens,
            'route_hash_reuse_ratio': 1.0 - unique_route / num_events if num_events else None,
            'count_hash_reuse_ratio': 1.0 - unique_counts / num_events if num_events else None,
            'top_active_expert_counts': active_expert_values[layer].most_common(8),
            'top_num_tokens': token_count_values[layer].most_common(8),
            'top_max_expert_counts': max_count_values[layer].most_common(8),
        })

    hot_dynamic_layers = [
        row for row in layer_rows
        if row['unique_count_hashes'] > 1 or row['unique_active_expert_counts'] > 1
    ]
    result = {
        'moe_log': args.moe_log,
        'num_events': len(events),
        'num_layers': len(by_layer),
        'templates': template_counter.most_common(50),
        'metadata_templates': metadata_template_counter.most_common(50),
        'capacity_buckets': capacity_counter.most_common(),
        'layers': layer_rows,
        'hot_dynamic_layers': hot_dynamic_layers[:100],
        'diagnosis': {
            'has_moe_events': bool(events),
            'layers_with_count_dynamicity': sum(1 for row in layer_rows if row['unique_count_hashes'] > 1),
            'layers_with_address_dynamicity': sum(1 for row in layer_rows if row['unique_topk_addresses'] > 1),
            'layers_with_token_count_dynamicity': sum(1 for row in layer_rows if row['unique_token_counts'] > 1),
            'expert_count_templates': len({event.get('count_hash') for event in events}),
            'route_value_templates': len({event.get('route_hash') for event in events}),
            'metadata_template_count': len(metadata_template_counter),
            'capacity_bucket_count': len(capacity_counter),
            'metadata_template_coverage_top1': (
                metadata_template_counter.most_common(1)[0][1] / len(events)
                if events and metadata_template_counter else 0.0
            ),
        },
        'staticity_interpretation': {
            'expert_ids': 'semantic value dynamic; do not collapse values blindly',
            'expert_counts': 'template dynamic; candidate for capacity bucketing and fallback',
            'topk_ids_address': 'address/metadata dynamic; candidate for persistent routing arena if address variants are high',
            'next_engineering_step': 'canonicalize routing metadata buffers while preserving dynamic expert values',
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('num_events', 'num_layers', 'diagnosis')}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
