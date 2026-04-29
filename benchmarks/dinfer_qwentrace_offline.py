#!/usr/bin/env python3
import argparse
import gc
import json
import sys
import time
import copy
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.morspec_loader import wrap_llada_prompt
from benchmarks.llada2_dinfer_kvcache_graph_generate import (
    init_vllm,
    make_dllm,
    run_generate,
    run_generate_with_dllm,
    run_with_forward_context,
    tensor_hash,
)
from prefill_graph.runtime import DynamicityProfiler

DINFER_ROOT = Path('/home/zhujianian/eurosys/dInfer')
if str(DINFER_ROOT / 'python') not in sys.path:
    sys.path.insert(0, str(DINFER_ROOT / 'python'))
from dinfer.model import LLaDA2MoeModelLM


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def make_eager_result(seconds, output_ids, nfe, stats=None):
    return {
        'seconds': seconds,
        'nfe': nfe,
        'output_ids': output_ids.clone(),
        'stats': stats,
        'shape': list(output_ids.shape),
        'used_graph': False,
        'fallback_reason': 'planner_rejected',
    }


def planner_allows_graph(args, prompt_len, idx):
    if prompt_len < args.graph_min_prompt_len:
        return False, f'prompt_len<{args.graph_min_prompt_len}'
    if args.graph_max_prompt_len and prompt_len > args.graph_max_prompt_len:
        return False, f'prompt_len>{args.graph_max_prompt_len}'
    if idx < args.graph_start_idx:
        return False, f'idx<{args.graph_start_idx}'
    return True, None


def initialize_vllm_moe_weights(model):
    initialized = 0
    for module in model.modules():
        quant_method = getattr(module, 'quant_method', None)
        if quant_method is None or not hasattr(quant_method, 'process_weights_after_loading'):
            continue
        quant_method.process_weights_after_loading(module)
        initialized += 1
    return initialized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/mnt/models/LLaDA2.0-mini')
    parser.add_argument('--workload', default='results/qwentrace_morspec_llada_32_2048.json')
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--gen-length', type=int, default=64)
    parser.add_argument('--block-length', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=0.99)
    parser.add_argument('--warmups', type=int, default=0)
    parser.add_argument('--graph-cross-block', action='store_true',
                        help='also capture cross-block KV update forwards instead of forcing them eager')
    parser.add_argument('--validate-replay', action='store_true')
    parser.add_argument('--validation-atol', type=float, default=0.0)
    parser.add_argument('--validation-rtol', type=float, default=0.0)
    parser.add_argument('--validation-mode', choices=['logits', 'top1', 'decoded'], default='logits')
    parser.add_argument('--admit-after', type=int, default=0)
    parser.add_argument('--validation-interval', type=int, default=0,
                        help='after admission, revalidate every N graph replays; 0 disables periodic validation')
    parser.add_argument('--validation-check-kv', action='store_true',
                        help='also compare graph/eager KV-cache outputs during validation')
    parser.add_argument('--persistent-runner', action='store_true',
                        help='reuse one graph-enabled dInfer runner across requests so graph templates persist')
    parser.add_argument('--graph-min-prompt-len', type=int, default=0,
                        help='latency-aware guardrail: use graph only for prompts at least this long')
    parser.add_argument('--graph-max-prompt-len', type=int, default=0,
                        help='latency-aware guardrail: use graph only up to this prompt length; 0 disables upper bound')
    parser.add_argument('--graph-start-idx', type=int, default=0,
                        help='latency-aware guardrail: use graph only after this request index so early requests can calibrate/fallback')
    parser.add_argument('--enable-ep', action=argparse.BooleanOptionalAction, default=True,
                        help='initialize vLLM expert-parallel metadata required by LLaDA2 MoE fused kernels')
    parser.add_argument('--model-warmup-tokens', type=int, default=128,
                        help='tokens used to initialize fused-MoE kernels after loading')
    parser.add_argument('--graph-max-templates', type=int, default=0,
                        help='memory guardrail: stop capturing new graph templates after this count; 0 disables')
    parser.add_argument('--graph-min-free-memory-mb', type=int, default=512,
                        help='memory guardrail: reject new graph captures below this free CUDA memory threshold')
    parser.add_argument('--cleanup-between-requests', action='store_true',
                        help='release per-request output tensors and cached CUDA graph pools after each request; cleanup time is reported separately')
    parser.add_argument('--cleanup-policy', choices=['never', 'always', 'memory_guard'], default=None,
                        help='cleanup policy for graph private pools; defaults to always when --cleanup-between-requests is set, otherwise never')
    parser.add_argument('--cleanup-min-free-memory-mb', type=int, default=8192,
                        help='for memory_guard cleanup, cleanup only when free CUDA memory is below this threshold')
    parser.add_argument('--cleanup-max-interval', type=int, default=0,
                        help='for memory_guard cleanup, force cleanup after this many requests since the last cleanup; 0 disables interval cleanup')
    parser.add_argument('--port', type=int, default=46411)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    workload = json.loads(Path(args.workload).read_text(encoding='utf-8'))
    reqs = workload['requests'][:args.limit]
    ctx = init_vllm(rank=0, world_size=1, port=args.port, enable_ep=args.enable_ep)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        model = LLaDA2MoeModelLM(config=config).eval()
        load_start = sync_time()
        model.load_weights(args.model, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        moe_postload_modules = initialize_vllm_moe_weights(model)
        if args.model_warmup_tokens > 0:
            warmup_ids = torch.arange(args.model_warmup_tokens, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                run_with_forward_context(lambda: model(warmup_ids, use_cache=False))
                run_with_forward_context(lambda: model(warmup_ids, use_cache=True))
        load_s = sync_time() - load_start
        mask_id = 156895
        eos_id = 156892
        persistent_eager = None
        persistent_graph = None
        if args.persistent_runner:
            persistent_eager, _ = make_dllm(
                model, args.gen_length, args.block_length, args.threshold, mask_id, eos_id,
                False, False, args.warmups, args.graph_cross_block, 1, 15)
            persistent_graph, _ = make_dllm(
                model, args.gen_length, args.block_length, args.threshold, mask_id, eos_id,
                False, True, args.warmups, args.graph_cross_block, 1, 15,
                args.validate_replay, args.validation_atol, args.validation_rtol,
                args.validation_mode, args.admit_after, args.validation_interval,
                args.validation_check_kv, args.graph_max_templates,
                args.graph_min_free_memory_mb)
        rows = []
        out = Path(args.output)
        cleanup_total_s = 0.0
        cleanup_count = 0
        last_cleanup_idx = -1
        cleanup_policy = args.cleanup_policy
        if cleanup_policy is None:
            cleanup_policy = 'always' if args.cleanup_between_requests else 'never'
        dynamicity_profiler = DynamicityProfiler()

        def build_result(partial):
            nonlocal cleanup_count
            eager_total = sum(r['eager_s'] for r in rows)
            graph_total = sum(r['graph_s'] for r in rows)
            return {
                'partial': partial,
                'model': args.model,
                'workload': args.workload,
                'num_samples': len(rows),
                'requested_limit': args.limit,
                'gen_length': args.gen_length,
                'block_length': args.block_length,
                'threshold': args.threshold,
                'graph_cross_block': args.graph_cross_block,
                'persistent_runner': args.persistent_runner,
                'validate_replay': args.validate_replay,
                'validation_atol': args.validation_atol,
                'validation_rtol': args.validation_rtol,
                'validation_mode': args.validation_mode,
                'validation_check_kv': args.validation_check_kv,
                'admit_after': args.admit_after,
                'validation_interval': args.validation_interval,
                'graph_min_prompt_len': args.graph_min_prompt_len,
                'graph_max_prompt_len': args.graph_max_prompt_len,
                'graph_start_idx': args.graph_start_idx,
                'graph_max_templates': args.graph_max_templates,
                'graph_min_free_memory_mb': args.graph_min_free_memory_mb,
                'cleanup_between_requests': args.cleanup_between_requests,
                'cleanup_policy': cleanup_policy,
                'cleanup_min_free_memory_mb': args.cleanup_min_free_memory_mb,
                'cleanup_max_interval': args.cleanup_max_interval,
                'cleanup_count': cleanup_count,
                'enable_ep': args.enable_ep,
                'model_warmup_tokens': args.model_warmup_tokens,
                'moe_postload_modules': moe_postload_modules,
                'admission_policy': (
                    'validate_before_admit_then_periodic'
                    if args.validate_replay and args.admit_after and args.validation_interval > 0
                    else 'validate_before_admit'
                    if args.validate_replay and args.admit_after
                    else 'validate_every_replay'
                    if args.validate_replay
                    else 'unvalidated_graph'
                ),
                'unsafe_admission': bool(not args.validate_replay),
                'load_s': load_s,
                'eager_total_s': eager_total,
                'graph_total_s': graph_total,
                'cleanup_total_s': cleanup_total_s,
                'graph_total_with_cleanup_s': graph_total + cleanup_total_s,
                'total_speedup': eager_total / graph_total if graph_total > 0 else None,
                'total_speedup_with_cleanup': eager_total / (graph_total + cleanup_total_s) if graph_total + cleanup_total_s > 0 else None,
                'all_same_tokens': all(r['same_tokens'] for r in rows),
                'dynamicity': dynamicity_profiler.summary(),
                'rows': rows,
            }

        def save_current(partial):
            result = build_result(partial)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
            if partial:
                print(f'Saved partial to {out}')
            return result

        for idx, req in enumerate(reqs):
            text = wrap_llada_prompt(req['prompt'])
            input_ids = torch.tensor(tokenizer(text)['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
            prompt_len = int(input_ids.shape[1])
            use_graph, fallback_reason = planner_allows_graph(args, prompt_len, idx)
            if args.persistent_runner:
                eager = run_generate_with_dllm(persistent_eager, input_ids, args.gen_length, args.block_length)
                if use_graph:
                    graph = run_generate_with_dllm(persistent_graph, input_ids, args.gen_length, args.block_length)
                    graph['used_graph'] = True
                    graph['fallback_reason'] = None
                else:
                    graph = make_eager_result(
                        eager['seconds'],
                        eager['output_ids'],
                        eager['nfe'],
                        copy.deepcopy(eager.get('stats')),
                    )
                    graph['fallback_reason'] = fallback_reason
            else:
                eager = run_generate(model, input_ids, args.gen_length, args.block_length, args.threshold, mask_id, eos_id,
                                     False, False, args.warmups, args.graph_cross_block, 1, 15)
                if use_graph:
                    graph = run_generate(model, input_ids, args.gen_length, args.block_length, args.threshold, mask_id, eos_id,
                                         False, True, args.warmups, args.graph_cross_block, 1, 15, args.validate_replay, args.validation_atol, args.validation_rtol, args.validation_mode, args.admit_after, args.validation_interval, args.validation_check_kv, args.graph_max_templates, args.graph_min_free_memory_mb)
                    graph['used_graph'] = True
                    graph['fallback_reason'] = None
                else:
                    graph = make_eager_result(
                        eager['seconds'],
                        eager['output_ids'],
                        eager['nfe'],
                        copy.deepcopy(eager.get('stats')),
                    )
                    graph['fallback_reason'] = fallback_reason
            same = torch.equal(eager['output_ids'], graph['output_ids'])
            row = {
                'idx': idx,
                'trace_target_len': int(req['target_input_length']),
                'prompt_len': prompt_len,
                'eager_s': eager['seconds'],
                'graph_s': graph['seconds'],
                'speedup': eager['seconds'] / graph['seconds'] if graph['seconds'] > 0 else None,
                'same_tokens': same,
                'used_graph': bool(graph.get('used_graph', use_graph)),
                'fallback_reason': graph.get('fallback_reason'),
                'eager_nfe': eager['nfe'],
                'graph_nfe': graph['nfe'],
                'eager_hash': tensor_hash(eager['output_ids']),
                'graph_hash': tensor_hash(graph['output_ids']),
                'graph_stats': graph['stats'],
                'graph_admission': graph.get('admission'),
            }
            graph_stats = graph.get('stats') or {}
            dynamicity_profiler.observe(
                'num_tokens',
                prompt_len,
                in_graph_key=True,
                semantic=True,
                component='dinfer_qwentrace',
            )
            dynamicity_profiler.observe(
                'mask_positions',
                graph_stats.get('metadata_updates', 0),
                in_graph_key=True,
                semantic=True,
                component='dinfer_qwentrace',
            )
            dynamicity_profiler.observe(
                'kv_cache',
                graph_stats.get('template_count', 0),
                in_graph_key=True,
                semantic=True,
                component='dinfer_qwentrace',
            )
            dynamicity_profiler.observe(
                'expert_ids',
                graph_stats.get('template_misses', 0),
                in_graph_key=True,
                semantic=True,
                component='dinfer_qwentrace',
            )
            rows.append(row)
            print(f"[{idx+1}/{len(reqs)}] trace_len={row['trace_target_len']} prompt_len={row['prompt_len']} eager={row['eager_s']:.3f}s graph={row['graph_s']:.3f}s speedup={row['speedup']:.2f} same={same}")
            save_current(partial=True)
            free_mb = None
            if torch.cuda.is_available():
                free_bytes, _ = torch.cuda.mem_get_info()
                free_mb = free_bytes / (1024 * 1024)
            should_cleanup = cleanup_policy == 'always'
            if cleanup_policy == 'memory_guard':
                below_memory = free_mb is not None and free_mb < args.cleanup_min_free_memory_mb
                over_interval = (
                    args.cleanup_max_interval > 0
                    and idx - last_cleanup_idx >= args.cleanup_max_interval
                )
                should_cleanup = below_memory or over_interval
                row['cleanup_free_memory_mb_before'] = free_mb
                row['cleanup_reason'] = (
                    'memory_guard'
                    if below_memory
                    else 'interval'
                    if over_interval
                    else None
                )
            if should_cleanup:
                cleanup_start = sync_time()
                del eager
                del graph
                del input_ids
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                cleanup_s = time.perf_counter() - cleanup_start
                cleanup_total_s += cleanup_s
                row['cleanup_s'] = cleanup_s
                cleanup_count += 1
                last_cleanup_idx = idx
        result = save_current(partial=False)
        print(json.dumps({k: result[k] for k in result if k != 'rows'}, indent=2, ensure_ascii=False))
        print(f'Saved to {out}')
    finally:
        ctx.__exit__(None, None, None)


if __name__ == '__main__':
    main()
