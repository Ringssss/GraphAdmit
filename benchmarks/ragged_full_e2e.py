#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_cases(spec):
    cases = []
    for case in spec.split(';'):
        case = case.strip()
        if case:
            cases.append([int(x) for x in case.replace('+', ',').split(',') if x.strip()])
    return cases


def build_prompt(tokenizer, tokens, variant):
    base = ("The quick brown fox jumps over the lazy dog. "
            "In a world of artificial intelligence, technology transforms everything. "
            f"Request variant {variant}. ") * 400
    ids = tokenizer.encode(base)
    return tokenizer.decode(ids[:tokens], skip_special_tokens=True)


def digest(outputs):
    h = hashlib.sha256()
    texts = []
    for output in outputs:
        text = output.outputs[0].text
        texts.append(text)
        h.update(text.encode('utf-8', errors='ignore'))
        h.update(b'\0')
    return h.hexdigest(), texts


def run_mode(model, mode, cases, max_model_len, collapse):
    tag = mode + ('_collapse' if collapse else '')
    trace_file = f"traces/ragged_full_{tag}.jsonl"
    Path(trace_file).parent.mkdir(exist_ok=True, parents=True)
    Path(trace_file).write_text('')
    os.environ['VLLM_CG_TRACE_FILE'] = trace_file
    if collapse:
        os.environ['VLLM_FULL_KEY_COLLAPSE'] = '1'
    else:
        os.environ.pop('VLLM_FULL_KEY_COLLAPSE', None)

    tokenizer = AutoTokenizer.from_pretrained(model)
    prompt_cache = {}
    for ci, lengths in enumerate(cases):
        for ri, tokens in enumerate(lengths):
            prompt_cache[(ci, ri)] = build_prompt(tokenizer, tokens, ci * 100 + ri)
    del tokenizer

    t0 = time.monotonic()
    llm = None
    try:
        llm = LLM(
            model=model,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_model_len,
            gpu_memory_utilization=0.90,
            disable_log_stats=True,
            enable_chunked_prefill=False,
            enforce_eager=False,
            compilation_config={"cudagraph_mode": mode},
        )
        init_s = time.monotonic() - t0
        sp = SamplingParams(max_tokens=1, temperature=0.0)
        rows = []
        for ci, lengths in enumerate(cases):
            prompts = [prompt_cache[(ci, ri)] for ri in range(len(lengths))]
            _ = llm.generate(prompts, sp)
            torch.cuda.synchronize()
            ts = time.perf_counter()
            outputs = llm.generate(prompts, sp)
            torch.cuda.synchronize()
            batch_ms = (time.perf_counter() - ts) * 1000
            fp, texts = digest(outputs)
            rows.append({
                "case": ci,
                "lengths": lengths,
                "total_tokens": sum(lengths),
                "num_reqs": len(lengths),
                "batch_ms": batch_ms,
                "per_request_ms": batch_ms / len(lengths),
                "fingerprint": fp,
                "texts": texts,
            })
        return {"mode": mode, "collapse": collapse, "ok": True,
                "init_s": init_s, "trace_file": trace_file, "rows": rows}
    except Exception as exc:
        return {"mode": mode, "collapse": collapse, "ok": False,
                "error": repr(exc), "trace_file": trace_file,
                "init_s": time.monotonic() - t0}
    finally:
        if llm is not None:
            del llm
        os.environ.pop('VLLM_CG_TRACE_FILE', None)
        os.environ.pop('VLLM_FULL_KEY_COLLAPSE', None)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/mnt/models/Meta-Llama-3-8B-Instruct')
    ap.add_argument('--max-model-len', type=int, default=2048)
    ap.add_argument('--cases', default='768;384,384;192,192,192,192;500,200,68')
    ap.add_argument('--modes', nargs='+', default=['PIECEWISE', 'FULL'])
    ap.add_argument('--include-collapse', action='store_true')
    args = ap.parse_args()

    cases = parse_cases(args.cases)
    results = []
    for mode in args.modes:
        print(f"\n=== {mode} ===")
        res = run_mode(args.model, mode, cases, args.max_model_len, False)
        print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=2))
        results.append(res)
        if args.include_collapse and mode == 'FULL':
            print(f"\n=== {mode} collapse ===")
            res = run_mode(args.model, mode, cases, args.max_model_len, True)
            print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=2))
            results.append(res)
    Path('results').mkdir(exist_ok=True)
    fn = 'results/ragged_full_e2e.json'
    json.dump(results, open(fn, 'w'), indent=2)
    print(f"Saved to {fn}")


if __name__ == '__main__':
    main()
