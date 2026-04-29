#!/usr/bin/env python3
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from types import MethodType

import torch
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prefill_graph.runtime import TemplateAdmissionController

DINFER_ROOT = Path('/home/zhujianian/eurosys/dInfer')
if str(DINFER_ROOT / 'python') not in sys.path:
    sys.path.insert(0, str(DINFER_ROOT / 'python'))

from dinfer import BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory, ThresholdParallelDecoder
from dinfer.decoding.generate_uniform import BlockDiffusionIteration, BlockDiffusionRunner
from dinfer.decoding.utils import KVCache, TokenArray
from dinfer.model import LLaDA2MoeModelLM
from vllm import distributed
from vllm.config import ParallelConfig, VllmConfig, get_current_vllm_config, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.v1.worker.workspace import init_workspace_manager


def init_vllm(rank: int, world_size: int, port: int, enable_ep: bool):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', str(port))
    parallel_config = ParallelConfig(enable_expert_parallel=enable_ep)
    vllm_config = VllmConfig(parallel_config=parallel_config)
    vllm_config.compilation_config.fast_moe_cold_start = False
    ctx = set_current_vllm_config(vllm_config)
    ctx.__enter__()
    distributed.init_distributed_environment(world_size, rank, 'env://', rank, 'nccl')
    distributed.initialize_model_parallel(world_size, backend='nccl')
    if torch.cuda.is_available():
        init_workspace_manager(torch.device(f'cuda:{torch.cuda.current_device()}'))
    return ctx


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def run_with_forward_context(fn):
    vllm_config = get_current_vllm_config()
    with set_forward_context(None, vllm_config):
        return fn()


def make_prompt(tokenizer, text: str, device):
    prompt = '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>' + text + '<|role_end|><role>ASSISTANT</role>'
    return torch.tensor(tokenizer(prompt)['input_ids'], dtype=torch.long, device=device).unsqueeze(0)


def tensor_hash(tensor):
    flat = tensor.detach().to(torch.int64).flatten()
    if flat.numel() == 0:
        return 0
    weights = torch.arange(1, flat.numel() + 1, device=flat.device, dtype=torch.int64)
    return int(((flat * weights).sum() % 2147483647).item())


def snapshot_stats(stats):
    return copy.deepcopy(stats) if stats is not None else None


def delta_stats(before, after):
    if after is None:
        return None
    if before is None:
        return snapshot_stats(after)
    delta = {}
    for key, value in after.items():
        old_value = before.get(key)
        if isinstance(value, (int, float)) and isinstance(old_value, (int, float)):
            delta[key] = value - old_value
        elif isinstance(value, list) and isinstance(old_value, list):
            delta[key] = value[len(old_value):]
        else:
            delta[key] = copy.deepcopy(value)
    delta['_cumulative'] = snapshot_stats(after)
    return delta


class GraphKVBlockDiffusionIteration(BlockDiffusionIteration):
    def __init__(self, warmups: int = 1, graph_cross_block: bool = False, min_replays: int = 1, validate_replay: bool = False, atol: float = 0.0, rtol: float = 0.0, validation_mode: str = 'logits', admit_after: int = 0, validation_interval: int = 0, validation_check_kv: bool = False, max_templates: int = 0, min_free_memory_mb: int = 512):
        super().__init__()
        self.warmups = warmups
        self.graph_cross_block = graph_cross_block
        self.min_replays = min_replays
        self.validate_replay = validate_replay
        self.atol = atol
        self.rtol = rtol
        self.validation_mode = validation_mode
        self.admit_after = admit_after
        self.validation_interval = validation_interval
        self.validation_check_kv = validation_check_kv
        self.max_templates = max_templates
        self.min_free_memory_bytes = int(min_free_memory_mb) * 1024 * 1024
        self.admission = TemplateAdmissionController(
            max_templates=max_templates,
            min_free_memory_bytes=self.min_free_memory_bytes,
            admit_after_passes=max(1, admit_after or 1),
        )
        self.validation_passes = {}
        self.key_replays = {}
        self.admitted_keys = set()
        self.disabled_keys = set()
        self.templates = {}
        self.template = None
        self.stats = {
            'captures': 0,
            'replays': 0,
            'eager_forwards': 0,
            'capture_seconds': 0.0,
            'replay_seconds': 0.0,
            'fallback_errors': [],
            'validation_fallbacks': 0,
            'validation_logit_mismatches': 0,
            'validation_top1_mismatches': 0,
            'validation_decoded_mismatches': 0,
            'validation_kv_mismatches': 0,
            'validation_passes': 0,
            'periodic_validations': 0,
            'post_admission_validation_failures': 0,
            'admitted_replays': 0,
            'metadata_updates': 0,
            'template_hits': 0,
            'template_misses': 0,
            'template_count': 0,
            'memory_guard_rejections': 0,
            'oom_fallbacks': 0,
        }

    def reset_template(self):
        self.template = None

    def _key(self, block, pos, attn_mask, past_key_values, replace_position, is_cross_block):
        cache_shape = None
        if past_key_values is not None and hasattr(past_key_values, '_data'):
            cache_shape = tuple(past_key_values._data.shape)
        attn_shape = None if attn_mask is None else tuple(attn_mask.shape)
        return (tuple(block.shape), tuple(pos.shape), attn_shape, cache_shape, tuple(replace_position), bool(is_cross_block))


    def _clone_token_array(self, x):
        cloned = copy.copy(x)
        cloned.data = x.data.clone()
        if hasattr(x, 'prompt') and torch.is_tensor(x.prompt):
            cloned.prompt = x.prompt.clone()
        return cloned

    def _kv_tensors(self, output):
        cache = getattr(output, 'past_key_values', None)
        data = getattr(cache, '_data', None)
        if data is None:
            return []
        if torch.is_tensor(data):
            return [data]
        return list(data)

    def _kv_validation_passed(self, output, eager_output):
        if not self.validation_check_kv:
            return True
        graph_tensors = self._kv_tensors(output)
        eager_tensors = self._kv_tensors(eager_output)
        if len(graph_tensors) != len(eager_tensors):
            self.stats['validation_kv_mismatches'] += 1
            return False
        for graph_tensor, eager_tensor in zip(graph_tensors, eager_tensors):
            if graph_tensor.shape != eager_tensor.shape:
                self.stats['validation_kv_mismatches'] += 1
                return False
            if not torch.allclose(graph_tensor, eager_tensor, atol=self.atol, rtol=self.rtol):
                self.stats['validation_kv_mismatches'] += 1
                return False
        return True

    def _validation_passed(self, output, eager_output, decoder=None, x=None, block_start=None, block_end=None):
        if self.validation_mode == 'decoded':
            if decoder is None or x is None or block_start is None or block_end is None:
                raise ValueError('decoded validation requires decoder, x, block_start, and block_end')
            graph_x = self._clone_token_array(x)
            eager_x = self._clone_token_array(x)
            decoder.decode(output.logits, block_start, block_end, graph_x)
            decoder.decode(eager_output.logits, block_start, block_end, eager_x)
            passed = torch.equal(
                graph_x.data[:, block_start:block_end],
                eager_x.data[:, block_start:block_end],
            )
            if not passed:
                self.stats['validation_decoded_mismatches'] += 1
            return passed and self._kv_validation_passed(output, eager_output)
        if self.validation_mode == 'top1':
            output_top1 = torch.argmax(output.logits, dim=-1)
            eager_top1 = torch.argmax(eager_output.logits, dim=-1)
            passed = torch.equal(output_top1, eager_top1)
            if not passed:
                self.stats['validation_top1_mismatches'] += 1
            return passed and self._kv_validation_passed(output, eager_output)
        passed = torch.allclose(output.logits, eager_output.logits, atol=self.atol, rtol=self.rtol)
        if not passed:
            self.stats['validation_logit_mismatches'] += 1
        return passed and self._kv_validation_passed(output, eager_output)

    def _needs_validation(self, key):
        if not self.validate_replay:
            return False
        if key not in self.admitted_keys:
            return True
        if self.validation_interval <= 0:
            return False
        replay_count = self.key_replays.get(key, 0)
        return replay_count > 0 and replay_count % self.validation_interval == 0

    def _make_static_cache(self, past_key_values):
        static_cache = KVCache(past_key_values._data.clone(), backend='vllm')
        static_cache.consolidate()
        return static_cache

    def _sync_static_cache(self, static_cache, past_key_values):
        static_cache._data.copy_(past_key_values._data)

    def _eager_model(self, model, block, pos, past_key_values, replace_position, backend, attn_mask=None):
        kwargs = {
            'position_ids': pos.clone(memory_format=torch.contiguous_format),
            'use_cache': True,
            'past_key_values': past_key_values,
            'replace_position': (0, 0) if backend == 'sglang' else replace_position,
        }
        if attn_mask is not None:
            kwargs['attention_mask'] = attn_mask
        return model(block.clone(memory_format=torch.contiguous_format), **kwargs)

    def _capture_template(self, model, block, pos, past_key_values, replace_position, backend, attn_mask, key):
        free_bytes = None
        if torch.cuda.is_available() and self.min_free_memory_bytes > 0:
            free_bytes, _ = torch.cuda.mem_get_info()
        can_capture, reject_reason = self.admission.can_capture(key, free_memory_bytes=free_bytes)
        if not can_capture:
            self.stats['memory_guard_rejections'] += 1
            raise RuntimeError(reject_reason or 'admission_rejected_capture')
        static_block = block.clone(memory_format=torch.contiguous_format)
        static_pos = pos.clone(memory_format=torch.contiguous_format)
        static_attn = None if attn_mask is None else attn_mask.contiguous()
        static_cache = self._make_static_cache(past_key_values)
        kwargs = {
            'position_ids': static_pos,
            'use_cache': True,
            'past_key_values': static_cache,
            'replace_position': (0, 0) if backend == 'sglang' else replace_position,
        }
        if static_attn is not None:
            kwargs['attention_mask'] = static_attn
        pool = torch.cuda.graph_pool_handle()
        output = None
        start = sync_time()
        for _ in range(self.warmups):
            self._sync_static_cache(static_cache, past_key_values)
            output = model(static_block, **kwargs)
        torch.cuda.synchronize()
        self._sync_static_cache(static_cache, past_key_values)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            output = model(static_block, **kwargs)
        torch.cuda.synchronize()
        self._sync_static_cache(static_cache, past_key_values)
        graph.replay()
        torch.cuda.synchronize()
        capture_seconds = time.perf_counter() - start
        self.stats['capture_seconds'] += capture_seconds
        self.stats['captures'] += 1
        self.admission.record_capture(key, capture_seconds)
        template = {
            'key': key,
            'static_block': static_block,
            'static_pos': static_pos,
            'static_attn': static_attn,
            'static_cache': static_cache,
            'graph': graph,
            'output': output,
        }
        self.templates[key] = template
        self.template = template
        self.stats['template_count'] = len(self.templates)
        return output

    def _validate_or_disable(self, key, output, eager_output, decoder, x, block_start, block_end, post_admission=False):
        if self._validation_passed(output, eager_output, decoder, x, block_start, block_end):
            self.stats['validation_passes'] += 1
            self.admission.record_validation(key, True)
            if post_admission:
                self.stats['periodic_validations'] += 1
            self.validation_passes[key] = self.validation_passes.get(key, 0) + 1
            if self.admit_after and self.validation_passes[key] >= self.admit_after:
                self.admitted_keys.add(key)
            return True
        self.admission.record_validation(key, False, reason='validation_failed')
        self.disabled_keys.add(key)
        self.templates.pop(key, None)
        self.template = None
        self.stats['template_count'] = len(self.templates)
        self.stats['validation_fallbacks'] += 1
        if post_admission:
            self.stats['post_admission_validation_failures'] += 1
            self.admitted_keys.discard(key)
        return False

    def _graph_or_eager_model(self, model, block, pos, past_key_values, replace_position, backend, attn_mask=None, is_cross_block=False, decoder=None, x=None, block_start=None, block_end=None):
        if past_key_values is None or (is_cross_block and not self.graph_cross_block):
            self.stats['eager_forwards'] += 1
            return self._eager_model(model, block, pos, past_key_values, replace_position, backend, attn_mask)
        key = self._key(block, pos, attn_mask, past_key_values, replace_position, is_cross_block)
        try:
            if key in self.disabled_keys:
                self.stats['eager_forwards'] += 1
                return self._eager_model(model, block, pos, past_key_values, replace_position, backend, attn_mask)
            template = self.templates.get(key)
            if template is None:
                self.stats['template_misses'] += 1
                output = self._capture_template(model, block, pos, past_key_values, replace_position, backend, attn_mask, key)
                if self._needs_validation(key):
                    eager_output = self._eager_model(model, block, pos, past_key_values, replace_position, backend, attn_mask)
                    if not self._validate_or_disable(key, output, eager_output, decoder, x, block_start, block_end, post_admission=key in self.admitted_keys):
                        self.stats['eager_forwards'] += 1
                        return eager_output
                return output
            self.stats['template_hits'] += 1
            self.template = template
            template['static_block'].copy_(block)
            template['static_pos'].copy_(pos)
            if attn_mask is not None and template['static_attn'] is not None:
                template['static_attn'].copy_(attn_mask)
            self._sync_static_cache(template['static_cache'], past_key_values)
            self.stats['metadata_updates'] += 1
            start = sync_time()
            template['graph'].replay()
            torch.cuda.synchronize()
            replay_seconds = time.perf_counter() - start
            self.stats['replay_seconds'] += replay_seconds
            self.stats['replays'] += 1
            self.admission.record_replay(key, replay_seconds)
            self.key_replays[key] = self.key_replays.get(key, 0) + 1
            output = template['output']
            if self._needs_validation(key):
                eager_output = self._eager_model(model, block, pos, past_key_values, replace_position, backend, attn_mask)
                if not self._validate_or_disable(key, output, eager_output, decoder, x, block_start, block_end, post_admission=key in self.admitted_keys):
                    self.stats['eager_forwards'] += 1
                    return eager_output
            elif key in self.admitted_keys:
                self.stats['admitted_replays'] += 1
            return output
        except Exception as exc:
            if len(self.stats['fallback_errors']) < 5:
                self.stats['fallback_errors'].append(repr(exc))
            if isinstance(exc, torch.cuda.OutOfMemoryError) or 'out of memory' in repr(exc).lower():
                self.disabled_keys.add(key)
                self.admission.disable(key, 'oom_fallback')
                self.templates.pop(key, None)
                self.stats['template_count'] = len(self.templates)
                self.stats['oom_fallbacks'] += 1
                torch.cuda.empty_cache()
            self.template = None
            self.stats['eager_forwards'] += 1
            return self._eager_model(model, block, pos, past_key_values, replace_position, backend, attn_mask)

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask, past_key_values, replace_position, backend, is_cross_block=False, block_length=32):
        if kv_cache is None:
            output = model(
                x.data[:, :block_loc.end],
                attention_mask=attn_mask[:, :block_loc.end, :block_loc.end],
                position_ids=pos_ids[:, :block_loc.end],
            )
            logits = output.logits[:, block_loc.start:block_loc.end]
            self.stats['eager_forwards'] += 1
        else:
            pos = pos_ids[:, block_loc.start:block_loc.end]
            mask = attn_mask if is_cross_block else None
            output = self._graph_or_eager_model(
                model, block, pos, past_key_values, replace_position, backend, mask, is_cross_block,
                decoder=decoder, x=x, block_start=block_loc.start, block_end=block_loc.end
            )
            logits = output.logits
            if backend == 'vllm':
                kv_cache.update(output.past_key_values)
        if is_cross_block:
            decoder.decode(logits[:, block_length:], block_loc.start + block_length, block_loc.end, x)
        else:
            decoder.decode(logits, block_loc.start, block_loc.end, x)
        self.num_forwards += 1
        self.iter_no += 1
        return output


class GraphKVBlockDiffusionRunner(BlockDiffusionRunner):
    def decode(self, *args, **kwargs):
        if hasattr(self.diff_iteration, 'reset_template'):
            self.diff_iteration.reset_template()
        return super().decode(*args, **kwargs)


def attach_graph_runner(dllm, warmups: int, graph_cross_block: bool, maximum_unroll: int, expected_tpf: int, validate_replay: bool = False, validation_atol: float = 0.0, validation_rtol: float = 0.0, validation_mode: str = 'logits', admit_after: int = 0, validation_interval: int = 0, validation_check_kv: bool = False, max_templates: int = 0, min_free_memory_mb: int = 512):
    graph_iter = GraphKVBlockDiffusionIteration(warmups=warmups, graph_cross_block=graph_cross_block, validate_replay=validate_replay, atol=validation_atol, rtol=validation_rtol, validation_mode=validation_mode, admit_after=admit_after, validation_interval=validation_interval, validation_check_kv=validation_check_kv, max_templates=max_templates, min_free_memory_mb=min_free_memory_mb)
    dllm.diff_iteration = graph_iter
    dllm.block_runner = GraphKVBlockDiffusionRunner(
        graph_iter,
        dllm.early_stop,
        maximum_unroll,
        expected_tpf,
        dllm.backend,
    )
    return graph_iter


@torch.no_grad()
def run_generate(model, input_ids, gen_length, block_length, threshold, mask_id, eos_id, early_stop, graph, warmups, graph_cross_block, maximum_unroll, expected_tpf, validate_replay=False, validation_atol=0.0, validation_rtol=0.0, validation_mode='logits', admit_after=0, validation_interval=0, validation_check_kv=False, max_templates=0, min_free_memory_mb=512):
    decoder = ThresholdParallelDecoder(temperature=0, threshold=threshold, mask_id=mask_id, eos_id=eos_id)
    cache_factory = KVCacheFactory('prefix', is_bd_model=True)
    dllm = BlockDiffusionLLM(
        model,
        decoder,
        BlockIteratorFactory(start_block_align=True, use_block_diffusion=True),
        cache_factory=cache_factory,
        early_stop=early_stop,
        maximum_unroll=maximum_unroll,
        expected_tpf=expected_tpf,
    )
    graph_iter = attach_graph_runner(dllm, warmups, graph_cross_block, maximum_unroll, expected_tpf, validate_replay, validation_atol, validation_rtol, validation_mode, admit_after, validation_interval, validation_check_kv, max_templates, min_free_memory_mb) if graph else None
    old_get_generated_tokens = TokenArray.get_generated_tokens
    def _return_full_buffer(self):
        return self.data.clone()
    TokenArray.get_generated_tokens = _return_full_buffer
    try:
        start = sync_time()
        out = run_with_forward_context(
            lambda: dllm.generate(input_ids, gen_length=gen_length, block_length=block_length)
        )
        seconds = sync_time() - start
    finally:
        TokenArray.get_generated_tokens = old_get_generated_tokens
    stats = graph_iter.stats if graph_iter is not None else None
    admission_summary = graph_iter.admission.summary() if graph_iter is not None else None
    return {
        'seconds': seconds,
        'nfe': dllm.num_forwards,
        'output_ids': out.clone(),
        'stats': stats,
        'admission': admission_summary,
        'shape': list(out.shape),
    }


@torch.no_grad()
def make_dllm(model, gen_length, block_length, threshold, mask_id, eos_id, early_stop, graph, warmups, graph_cross_block, maximum_unroll, expected_tpf, validate_replay=False, validation_atol=0.0, validation_rtol=0.0, validation_mode='logits', admit_after=0, validation_interval=0, validation_check_kv=False, max_templates=0, min_free_memory_mb=512):
    decoder = ThresholdParallelDecoder(temperature=0, threshold=threshold, mask_id=mask_id, eos_id=eos_id)
    cache_factory = KVCacheFactory('prefix', is_bd_model=True)
    dllm = BlockDiffusionLLM(
        model,
        decoder,
        BlockIteratorFactory(start_block_align=True, use_block_diffusion=True),
        cache_factory=cache_factory,
        early_stop=early_stop,
        maximum_unroll=maximum_unroll,
        expected_tpf=expected_tpf,
    )
    graph_iter = attach_graph_runner(dllm, warmups, graph_cross_block, maximum_unroll, expected_tpf, validate_replay, validation_atol, validation_rtol, validation_mode, admit_after, validation_interval, validation_check_kv, max_templates, min_free_memory_mb) if graph else None
    return dllm, graph_iter


@torch.no_grad()
def run_generate_with_dllm(dllm, input_ids, gen_length, block_length):
    old_get_generated_tokens = TokenArray.get_generated_tokens
    def _return_full_buffer(self):
        return self.data.clone()
    TokenArray.get_generated_tokens = _return_full_buffer
    before_nfe = dllm.num_forwards
    before_stats = snapshot_stats(getattr(dllm.diff_iteration, 'stats', None))
    try:
        block_runner = getattr(dllm, 'block_runner', None)
        if block_runner is not None:
            if hasattr(block_runner, 'need_cross_block_update'):
                block_runner.need_cross_block_update = False
            if hasattr(block_runner, 'cache_update_count'):
                block_runner.cache_update_count = 0
            if hasattr(block_runner, 'hidden_cache_update_count'):
                block_runner.hidden_cache_update_count = 0
        start = sync_time()
        out = run_with_forward_context(
            lambda: dllm.generate(input_ids, gen_length=gen_length, block_length=block_length)
        )
        seconds = sync_time() - start
    finally:
        TokenArray.get_generated_tokens = old_get_generated_tokens
    after_stats = snapshot_stats(getattr(dllm.diff_iteration, 'stats', None))
    admission_summary = (
        dllm.diff_iteration.admission.summary()
        if hasattr(getattr(dllm, 'diff_iteration', None), 'admission')
        else None
    )
    return {
        'seconds': seconds,
        'nfe': dllm.num_forwards - before_nfe,
        'output_ids': out.clone(),
        'stats': delta_stats(before_stats, after_stats),
        'admission': admission_summary,
        'shape': list(out.shape),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/mnt/models/LLaDA2.0-mini')
    parser.add_argument('--prompt', default='Say hi in five words.')
    parser.add_argument('--gen-length', type=int, default=64)
    parser.add_argument('--block-length', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--maximum-unroll', type=int, default=1)
    parser.add_argument('--expected-tpf', type=int, default=15)
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--graph-cross-block', action='store_true')
    parser.add_argument('--validate-replay', action='store_true')
    parser.add_argument('--validation-atol', type=float, default=0.0)
    parser.add_argument('--validation-rtol', type=float, default=0.0)
    parser.add_argument('--validation-mode', choices=['logits', 'top1', 'decoded'], default='logits')
    parser.add_argument('--admit-after', type=int, default=0)
    parser.add_argument('--validation-interval', type=int, default=0,
                        help='after admission, revalidate every N graph replays; 0 disables periodic validation')
    parser.add_argument('--validation-check-kv', action='store_true',
                        help='also compare graph/eager KV-cache outputs during validation')
    parser.add_argument('--port', type=int, default=46211)
    parser.add_argument('--output', default='results/llada2_dinfer_kvcache_graph_generate.json')
    parser.add_argument('--enable-ep', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    ctx = init_vllm(rank=0, world_size=1, port=args.port, enable_ep=args.enable_ep)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        model = LLaDA2MoeModelLM(config=config).eval()
        load_start = sync_time()
        model.load_weights(args.model, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        load_s = sync_time() - load_start
        input_ids = make_prompt(tokenizer, args.prompt, device)
        mask_id = 156895
        eos_id = 156892

        eager = run_generate(model, input_ids, args.gen_length, args.block_length, args.threshold, mask_id, eos_id,
                             args.early_stop, False, args.warmups, args.graph_cross_block, args.maximum_unroll, args.expected_tpf)
        graph = run_generate(model, input_ids, args.gen_length, args.block_length, args.threshold, mask_id, eos_id,
                             args.early_stop, True, args.warmups, args.graph_cross_block, args.maximum_unroll, args.expected_tpf, args.validate_replay, args.validation_atol, args.validation_rtol, args.validation_mode, args.admit_after, args.validation_interval, args.validation_check_kv)
        same = torch.equal(eager['output_ids'], graph['output_ids'])
        result = {
            'model': args.model,
            'prompt_len': input_ids.shape[1],
            'gen_length': args.gen_length,
            'block_length': args.block_length,
            'threshold': args.threshold,
            'early_stop': args.early_stop,
            'maximum_unroll': args.maximum_unroll,
            'expected_tpf': args.expected_tpf,
            'validate_replay': args.validate_replay,
            'validation_atol': args.validation_atol,
            'validation_rtol': args.validation_rtol,
            'validation_mode': args.validation_mode,
            'validation_check_kv': args.validation_check_kv,
            'admit_after': args.admit_after,
            'validation_interval': args.validation_interval,
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
            'eager_dinfer': {
                'seconds': eager['seconds'],
                'nfe': eager['nfe'],
                'hash': tensor_hash(eager['output_ids']),
                'text': tokenizer.decode(eager['output_ids'][0], skip_special_tokens=False),
                'shape': eager['shape'],
            },
            'graph_dinfer': {
                'seconds': graph['seconds'],
                'nfe': graph['nfe'],
                'hash': tensor_hash(graph['output_ids']),
                'text': tokenizer.decode(graph['output_ids'][0], skip_special_tokens=False),
                'shape': graph['shape'],
                'stats': graph['stats'],
            },
            'same_tokens': same,
            'speedup_total': eager['seconds'] / graph['seconds'] if graph['seconds'] > 0 else None,
            'speedup_replay_only': (eager['seconds'] / graph['stats']['replay_seconds']
                                    if graph['stats'] and graph['stats']['replay_seconds'] > 0 else None),
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2))
    finally:
        ctx.__exit__(None, None, None)


if __name__ == '__main__':
    main()
