#!/usr/bin/env python3
import json
from pathlib import Path


def extract_prompt(row):
    if 'prompt' in row and isinstance(row['prompt'], str):
        return row['prompt']
    if 'question' in row and isinstance(row['question'], str):
        return row['question']
    if 'turns' in row and row['turns']:
        return '\n'.join(str(x) for x in row['turns'])
    return json.dumps(row, ensure_ascii=False)


def load_morspec_prompts(dataset, limit=None, offset=0):
    path = Path(dataset)
    prompts = []
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
    rows = rows[offset: None if limit is None else offset + limit]
    for row in rows:
        prompts.append(extract_prompt(row))
    return prompts, rows


def wrap_llada_prompt(text):
    return '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>' + text + '<|role_end|><role>ASSISTANT</role>'
