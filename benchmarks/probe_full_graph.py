#!/usr/bin/env python3
import argparse
import gc
import json
import os
import sys
import hashlib
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def build_prompt(model, tokens, variant=0):
    tok = AutoTokenizer.from_pretrained(model)
    base = ("The quick brown fox jumps over the lazy dog. "
            "In a world of artificial intelligence, technology transforms everything. "
            f"Variant {variant} keeps this request distinct. ") * 300
    ids = tok.encode(base)
    text = tok.decode(ids[:tokens], skip_special_tokens=True)
    del tok
    return text


def parse_lengths(spec):
    return [int(x) for x in spec.replace("+", ",").split(",") if x.strip()]


def fingerprint_outputs(outputs):
    h = hashlib.sha256()
    texts = []
    for out in outputs:
        text = out.outputs[0].text
        texts.append(text)
        h.update(text.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest(), texts


def run_case(model, mode, lengths, max_model_len, trace_file, collapse_full_key=False):
    os.environ["VLLM_CG_TRACE_FILE"] = trace_file
    if collapse_full_key:
        os.environ["VLLM_FULL_KEY_COLLAPSE"] = "1"
    else:
        os.environ.pop("VLLM_FULL_KEY_COLLAPSE", None)
    Path(trace_file).parent.mkdir(parents=True, exist_ok=True)
    Path(trace_file).write_text("")
    prompts = [build_prompt(model, tokens, i) for i, tokens in enumerate(lengths)]
    t0 = time.monotonic()
    err = None
    result = None
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
        warm = llm.generate(prompts, sp)
        torch.cuda.synchronize()
        ts = time.perf_counter()
        outputs = llm.generate(prompts, sp)
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - ts) * 1000
        digest, texts = fingerprint_outputs(outputs)
        warm_digest, _ = fingerprint_outputs(warm)
        result = {
            "mode": mode,
            "ok": True,
            "lengths": lengths,
            "collapse_full_key": collapse_full_key,
            "init_s": init_s,
            "batch_ttft_ms": ttft_ms,
            "per_request_ms": ttft_ms / len(prompts),
            "fingerprint": digest,
            "warmup_fingerprint": warm_digest,
            "texts": texts,
        }
        del llm
    except Exception as exc:
        err = repr(exc)
        result = {"mode": mode, "ok": False, "error": err, "init_s": time.monotonic() - t0}
    finally:
        os.environ.pop("VLLM_CG_TRACE_FILE", None)
        os.environ.pop("VLLM_FULL_KEY_COLLAPSE", None)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/models/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--tokens", type=int, default=768)
    ap.add_argument("--lengths", default=None,
                    help="Comma- or plus-separated ragged request lengths, e.g. 768 or 384,384 or 500,200,68")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--modes", nargs="+", default=["PIECEWISE", "FULL", "FULL_AND_PIECEWISE"])
    ap.add_argument("--collapse-full-key", action="store_true")
    args = ap.parse_args()
    lengths = parse_lengths(args.lengths) if args.lengths else [args.tokens]
    tag = "x".join(map(str, lengths))
    if args.collapse_full_key:
        tag += "_collapse"
    out = []
    for mode in args.modes:
        print(f"\n=== probing {mode} ===")
        trace_file = f"traces/probe_full_{mode}_{tag}.jsonl"
        res = run_case(args.model, mode, lengths, args.max_model_len, trace_file,
                       collapse_full_key=args.collapse_full_key)
        res["trace_file"] = trace_file
        print(json.dumps(res, indent=2))
        out.append(res)
    Path("results").mkdir(exist_ok=True)
    fn = f"results/probe_full_{tag}.json"
    json.dump(out, open(fn, "w"), indent=2)
    print(f"Saved to {fn}")


if __name__ == "__main__":
    main()
