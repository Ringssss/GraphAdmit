#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def make_prompt(words, token_budget_hint):
    text = (' '.join(words) + ' ') * max(1, token_budget_hint // max(1, len(words)))
    return text[: max(32, token_budget_hint * 6)]


def make_token_ids(length, seed):
    # Keep ids inside the normal low-id text range so the probe can exercise
    # exact prompt lengths without relying on tokenizer-specific string lengths.
    base = 1000 + (seed % 997)
    return [base + ((seed * 31 + i * 17) % 2000) for i in range(length)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/keycollapse_workload.json')
    parser.add_argument('--totals', default='1024,2048')
    parser.add_argument('--num-reqs', default='1,2,4,8,16')
    args = parser.parse_args()

    words = ['staticity', 'recovery', 'metadata', 'arena', 'ragged', 'prefill', 'template', 'graph']
    requests = []
    idx = 0
    for total in [int(x) for x in args.totals.split(',') if x]:
        for num_reqs in [int(x) for x in args.num_reqs.split(',') if x]:
            if num_reqs > total:
                continue
            layouts = []
            uniform = [total // num_reqs] * num_reqs
            uniform[-1] += total - sum(uniform)
            layouts.append(('uniform', uniform))
            if num_reqs > 1:
                skewed = [max(1, total // (num_reqs * 4))] * num_reqs
                skewed[-1] += total - sum(skewed)
                layouts.append(('skewed', skewed))
                randomish = []
                remain = total
                for i in range(num_reqs):
                    if i == num_reqs - 1:
                        val = remain
                    else:
                        val = max(1, min(remain - (num_reqs - i - 1), ((i * 37 + 17) % max(2, total // num_reqs * 2))))
                    randomish.append(val)
                    remain -= val
                layouts.append(('randomish', randomish))
            for layout_name, lengths in layouts:
                for local_idx, length in enumerate(lengths):
                    requests.append({
                        'id': f'kc-{idx}',
                        'group_id': f'total{total}_reqs{num_reqs}_{layout_name}',
                        'layout': layout_name,
                        'group_total_tokens': total,
                        'group_num_reqs': num_reqs,
                        'group_local_idx': local_idx,
                        'target_input_length': length,
                        'actual_input_length': length,
                        'prompt_token_ids': make_token_ids(length, idx),
                        'prompt': make_prompt(words, length),
                    })
                    idx += 1
    result = {'kind': 'keycollapse_controlled', 'requests': requests}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(out), 'requests': len(requests)}, indent=2))


if __name__ == '__main__':
    main()
