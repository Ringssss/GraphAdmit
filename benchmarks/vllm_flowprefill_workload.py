#!/usr/bin/env python3
import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prefill_graph.planner.dp_solver import CostModel, VLLM_DEFAULT_SIZES, generate_candidates, solve_bucket_dp
from benchmarks.online_admission_policy_refresh import build_online_policy, read_rows
from prefill_graph.runtime import OnlineSelfLearningAdmissionController


def exact_dp_sizes(lengths, max_s, max_buckets, memory_budget_mb, warmup_budget_s):
    if not lengths:
        return [], None
    arr = np.array(lengths, dtype=np.int64)
    candidates = generate_candidates(max_size=max_s, fine_grain_up_to=min(256, max_s), fine_step=8, coarse_step=64)
    candidates = sorted(set(candidates) | {s for s in VLLM_DEFAULT_SIZES if s <= max_s} | {max_s})
    plan = solve_bucket_dp(
        token_counts=arr,
        max_buckets=max_buckets,
        candidate_sizes=candidates,
        cost_model=CostModel(),
        memory_budget_mb=memory_budget_mb,
        warmup_budget_s=warmup_budget_s,
        lambda_mem=0.1,
        lambda_warmup=100.0,
        lambda_fallback=1.0,
    )
    return plan.bucket_sizes, plan


def parse_int_list(value):
    if value is None or not value.strip():
        return []
    return sorted({int(item.strip()) for item in value.split(',') if item.strip()})


def runtime_policy_capture_sizes(path, base_capture_size):
    if not path:
        return []
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception as exc:
        print(f'warning: failed to read runtime policy capture sizes from {path}: {exc}')
        return []
    policy = data.get('runtime_policy', data)
    sizes = set()
    residual = policy.get('residual_capture')
    if isinstance(residual, dict):
        for value in residual.get('extra_capture_sizes') or []:
            try:
                sizes.add(int(value))
            except (TypeError, ValueError):
                pass
    for rule in policy.get('fixed_metadata_arena_ranges') or []:
        if not isinstance(rule, dict):
            continue
        try:
            sizes.add(int(rule.get('template_tokens')))
        except (TypeError, ValueError):
            pass
    for rule in policy.get('rules') or []:
        if not isinstance(rule, dict):
            continue
        action = str(rule.get('action', ''))
        # "default"/"cp" are graph actions only inside vLLM's base capture
        # family.  Extra capture sizes must come from explicit GraphAdmit
        # templates; otherwise a demand-filtered policy would accidentally
        # pre-capture every fallback bucket that happens to carry
        # template_tokens for range documentation.
        if action not in {'ours', 'ours_cp'}:
            continue
        try:
            sizes.add(int(rule.get('template_tokens')))
        except (TypeError, ValueError):
            pass
    return sorted(size for size in sizes if size > int(base_capture_size))


def runtime_policy_graph_template(
    path,
    tokens,
    *,
    num_reqs=1,
    template_id_style='bucket',
):
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:
        return None
    policy = data.get('runtime_policy', data)
    default_action = str(policy.get('default_action', 'default'))
    graph_actions = set(policy.get('single_engine_graph_actions') or ['default', 'ours', 'cp', 'ours_cp'])
    fallback_actions = set(policy.get('single_engine_fallback_actions') or ['eager', 'compile', 'compiled', 'fallback', 'none'])
    fallback_actions -= graph_actions
    for rule in policy.get('rules') or []:
        if not isinstance(rule, dict):
            continue
        try:
            lo = int(rule.get('lo', rule.get('low', 0)))
            hi = int(rule.get('hi', rule.get('high', 0)))
        except (TypeError, ValueError):
            continue
        if not (lo < int(tokens) <= hi):
            continue
        action = str(rule.get('action', default_action))
        if action in fallback_actions or action not in graph_actions:
            return None
        try:
            template_tokens = int(rule.get('template_tokens'))
        except (TypeError, ValueError):
            return None
        if action not in {'ours', 'ours_cp'}:
            return None
        req_part = '*' if num_reqs is None else str(int(num_reqs))
        range_template_id = (
            f'{action}:{lo}:{hi}:template={template_tokens}:reqs={req_part}'
        )
        wildcard_template_id = (
            f'{action}:{lo}:{hi}:template={template_tokens}:reqs=*'
        )
        bucket_template_id = f'tokens={template_tokens}'
        exact_template_id = (
            f'{action}:{lo}:{hi}:tokens={int(tokens)}:'
            f'template={template_tokens}:reqs={req_part}'
        )
        aliases = [range_template_id]
        if wildcard_template_id not in aliases:
            aliases.append(wildcard_template_id)
        if bucket_template_id not in aliases:
            aliases.append(bucket_template_id)
        if exact_template_id not in aliases:
            aliases.insert(0, exact_template_id)
        return {
            'template_id': (
                exact_template_id
                if template_id_style == 'exact'
                else range_template_id
                if template_id_style == 'range'
                else bucket_template_id
            ),
            'template_aliases': aliases,
            'action': action,
            'template_tokens': template_tokens,
            'lo': lo,
            'hi': hi,
        }
    return None


def append_live_admission_observation(path, *, template_id, graph_ms, fallback_ms, correct, tokens, request_index, extra=None):
    record = {
        'template_id': str(template_id),
        'graph_ms': float(graph_ms),
        'fallback_ms': float(fallback_ms),
        'correct': bool(correct),
        'token_correct': bool(correct),
        'tokens': int(tokens),
        'request_index': int(request_index),
        'ts': time.time(),
    }
    if extra:
        record.update(extra)
    obs = Path(path)
    obs.parent.mkdir(parents=True, exist_ok=True)
    with obs.open('a', encoding='utf-8') as fp:
        fp.write(json.dumps(record, sort_keys=True) + '\n')


def _set_env_temporarily(updates):
    old = {key: os.environ.get(key) for key in updates}
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    return old


def _restore_env(saved):
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _extract_first_generation(request_outputs):
    output = (
        request_outputs[0].outputs[0]
        if request_outputs and request_outputs[0].outputs
        else None
    )
    token_ids = list(output.token_ids) if output is not None else []
    text = output.text if output is not None else ''
    return [int(x) for x in token_ids], text


def _timed_generate_one(llm, prompt, sampling_params):
    torch.cuda.synchronize()
    start = time.perf_counter()
    request_outputs = llm.generate([prompt], sampling_params)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000
    token_ids, text = _extract_first_generation(request_outputs)
    return ms, token_ids, text


def _write_runtime_control(path, **values):
    if not path:
        return
    control = Path(path)
    control.parent.mkdir(parents=True, exist_ok=True)
    tmp = control.with_name(f'{control.name}.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(values, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(control)


def _count_live_rows(path, *, sources=None):
    if not path or not Path(path).exists():
        return {
            'observations': 0,
            'correct': 0,
            'useful': 0,
        }
    source_set = set(sources or [])
    counts = {
        'observations': 0,
        'correct': 0,
        'useful': 0,
    }
    with Path(path).open('r', encoding='utf-8') as fp:
        for line in fp:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if source_set and row.get('source') not in source_set:
                continue
            counts['observations'] += 1
            counts['correct'] += int(bool(row.get('correct')))
            counts['useful'] += int(bool(row.get('useful')))
    return counts


def _candidate_matches(runtime_policy, lens, *, template_id_style):
    matches = []
    seen = set()
    for index, tokens in enumerate(lens):
        match = runtime_policy_graph_template(
            runtime_policy,
            tokens,
            num_reqs=1,
            template_id_style=template_id_style,
        )
        if not match:
            continue
        key = match['template_id']
        if key in seen:
            continue
        seen.add(key)
        matches.append((index, tokens, match))
    return matches


def _append_probe_crash_blacklist(
    live_observations,
    *,
    runtime_policy,
    lens,
    validate_n,
    template_id_style,
    returncode,
):
    if not live_observations:
        return 0
    written = 0
    for vi, tokens, match in _candidate_matches(
        runtime_policy,
        lens[:validate_n],
        template_id_style=template_id_style,
    ):
        append_live_admission_observation(
            live_observations,
            template_id=match['template_id'],
            graph_ms=1.0e9,
            fallback_ms=0.0,
            correct=False,
            tokens=tokens,
            request_index=vi,
            extra={
                'action': match['action'],
                'template_aliases': match.get('template_aliases', []),
                'template_tokens': match['template_tokens'],
                'lo': match['lo'],
                'hi': match['hi'],
                'source': 'trusted_live_graph_replay',
                'trusted_graph_replay': True,
                'same_engine': True,
                'shadow_engine': True,
                'validation_mode': 'isolated_probe_crash_blacklist',
                'probe_crashed': True,
                'probe_returncode': int(returncode),
                'useful': False,
            },
        )
        written += 1
    return written


def _run_isolated_live_probe(payload):
    probe_payload = Path(payload['probe_payload'])
    probe_payload.parent.mkdir(parents=True, exist_ok=True)
    probe_payload.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8')
    before = _count_live_rows(
        payload['live_observations'],
        sources={'live_graph_replay', 'trusted_live_graph_replay'},
    )
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        '--live-admission-probe-worker',
        str(probe_payload),
    ]
    timeout = float(payload.get('probe_timeout_s') or 0.0)
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            timeout=timeout if timeout > 0 else None,
        )
        returncode = int(proc.returncode)
    except subprocess.TimeoutExpired:
        returncode = 124
    if returncode != 0:
        validate_n = int(payload.get('validate_n') or len(payload['prompts']))
        blacklisted = _append_probe_crash_blacklist(
            payload['live_observations'],
            runtime_policy=payload['runtime_policy'],
            lens=payload['lens'],
            validate_n=validate_n,
            template_id_style=payload['live_observation_template_id'],
            returncode=returncode,
        )
        print(
            f'  isolated live probe failed rc={returncode}; '
            f'wrote {blacklisted} fail-closed blacklist observations'
        )
    after = _count_live_rows(
        payload['live_observations'],
        sources={'live_graph_replay', 'trusted_live_graph_replay'},
    )
    return {
        'returncode': returncode,
        'observations': after['observations'] - before['observations'],
        'correct': after['correct'] - before['correct'],
        'useful': after['useful'] - before['useful'],
    }


def run_live_admission_probe_worker(payload_path):
    payload = json.loads(Path(payload_path).read_text(encoding='utf-8'))
    from vllm import LLM, SamplingParams

    runtime_control_path = payload['runtime_control']
    _write_runtime_control(
        runtime_control_path,
        unsafe_live_explore_replay=False,
        force_runtime_fallback=False,
    )
    env_updates = {
        'STATICITY_VLLM_RUNTIME_POLICY': payload['runtime_policy'],
        'STATICITY_VLLM_BASE_CAPTURE_SIZE': str(payload['runtime_base_capture_size']),
        'STATICITY_VLLM_RUNTIME_ACTIVE': '1',
        'STATICITY_VLLM_LIVE_ADMISSION': '1',
        'STATICITY_VLLM_LIVE_EXPLORE': '1',
        'STATICITY_VLLM_LIVE_MIN_SAMPLES': str(payload['live_min_samples']),
        'STATICITY_VLLM_LIVE_MIN_USEFUL_RATE': str(payload['live_min_useful_rate']),
        'STATICITY_VLLM_LIVE_MIN_SAVING_MS': str(payload['live_min_saving_ms']),
        'STATICITY_VLLM_LIVE_CAPTURE': '1',
        'STATICITY_VLLM_ALLOW_RUNTIME_CUDAGRAPH_CAPTURE': '1',
        'STATICITY_VLLM_RUNTIME_CONTROL': runtime_control_path,
        'STATICITY_VLLM_LIVE_OBSERVATIONS': payload['live_observations'],
    }
    if payload.get('live_max_p95_regression_ms') is not None:
        env_updates['STATICITY_VLLM_LIVE_MAX_P95_REGRESSION_MS'] = str(
            payload['live_max_p95_regression_ms'])
    if payload.get('fixed_metadata_arena'):
        env_updates['STATICITY_VLLM_FIXED_METADATA_ARENA'] = '1'
    if payload.get('fixed_metadata_arena_max_reqs'):
        env_updates['STATICITY_VLLM_FIXED_METADATA_ARENA_MAX_REQS'] = str(
            payload['fixed_metadata_arena_max_reqs'])
    if payload.get('fixed_metadata_arena_min_tokens'):
        env_updates['STATICITY_VLLM_FIXED_METADATA_ARENA_MIN_TOKENS'] = str(
            payload['fixed_metadata_arena_min_tokens'])
    if payload.get('fixed_metadata_arena_max_tokens'):
        env_updates['STATICITY_VLLM_FIXED_METADATA_ARENA_MAX_TOKENS'] = str(
            payload['fixed_metadata_arena_max_tokens'])
    if payload.get('full_key_collapse'):
        env_updates['VLLM_FULL_KEY_COLLAPSE'] = '1'
    if payload.get('moe_capacity_buckets'):
        env_updates['STATICITY_VLLM_MOE_CAPACITY_BUCKETS'] = str(
            payload['moe_capacity_buckets'])
    if payload.get('live_observation_template_id') == 'exact':
        env_updates['STATICITY_VLLM_EXACT_TOKEN_TEMPLATES'] = '1'
    saved = _set_env_temporarily(env_updates)
    try:
        cc = {
            'cudagraph_mode': payload['cudagraph_mode'],
            'cudagraph_capture_sizes': payload['cap_sizes'],
            'max_cudagraph_capture_size': max(payload['cap_sizes']),
        }
        kw = dict(
            model=payload['model'],
            tensor_parallel_size=int(payload['tp']),
            max_model_len=int(payload['max_model_len']),
            gpu_memory_utilization=float(payload['gmu']),
            enforce_eager=False,
            disable_log_stats=True,
            enable_chunked_prefill=True,
            enable_return_routed_experts=bool(
                payload.get('enable_return_routed_experts')),
            compilation_config=cc,
        )
        if payload.get('disable_prefix_caching'):
            kw['enable_prefix_caching'] = False
        if payload.get('max_num_seqs'):
            kw['max_num_seqs'] = int(payload['max_num_seqs'])
        llm = LLM(**kw)
        sp = SamplingParams(max_tokens=int(payload['max_tokens']), temperature=0.0)
        prompts = payload['prompts']
        lens = [int(x) for x in payload['lens']]
        _ = llm.generate([prompts[0]], sp)
        validate_n = int(payload.get('validate_n') or len(prompts))
        for vi, prompt in enumerate(prompts[:validate_n]):
            match = runtime_policy_graph_template(
                payload['runtime_policy'],
                lens[vi],
                num_reqs=1,
                template_id_style=payload['live_observation_template_id'],
            )
            if not match:
                continue
            graph_env = _set_env_temporarily({
                'STATICITY_VLLM_RUNTIME_ACTIVE': '1',
                'STATICITY_VLLM_LIVE_EXPLORE': '1',
                'STATICITY_VLLM_LIVE_MIN_SAMPLES': str(
                    max(1, int(payload['live_min_samples']))),
                'STATICITY_VLLM_UNSAFE_LIVE_EXPLORE_REPLAY': '1',
                'STATICITY_VLLM_FORCE_RUNTIME_FALLBACK': None,
            })
            try:
                _write_runtime_control(
                    runtime_control_path,
                    unsafe_live_explore_replay=True,
                    force_runtime_fallback=False,
                )
                capture_ms, _, _ = _timed_generate_one(llm, prompt, sp)
                graph_ms, graph_tokens, graph_text = _timed_generate_one(
                    llm, prompt, sp)
            finally:
                _write_runtime_control(
                    runtime_control_path,
                    unsafe_live_explore_replay=False,
                    force_runtime_fallback=False,
                )
                _restore_env(graph_env)
            fallback_env = _set_env_temporarily({
                'STATICITY_VLLM_RUNTIME_ACTIVE': '1',
                'STATICITY_VLLM_UNSAFE_LIVE_EXPLORE_REPLAY': None,
                'STATICITY_VLLM_FORCE_RUNTIME_FALLBACK': '1',
            })
            try:
                _write_runtime_control(
                    runtime_control_path,
                    unsafe_live_explore_replay=False,
                    force_runtime_fallback=True,
                )
                fallback_ms, fallback_tokens, fallback_text = _timed_generate_one(
                    llm, prompt, sp)
            finally:
                _write_runtime_control(
                    runtime_control_path,
                    unsafe_live_explore_replay=False,
                    force_runtime_fallback=False,
                )
                _restore_env(fallback_env)
            correct = graph_tokens == fallback_tokens
            useful = correct and graph_ms < fallback_ms
            append_live_admission_observation(
                payload['live_observations'],
                template_id=match['template_id'],
                graph_ms=graph_ms,
                fallback_ms=fallback_ms,
                correct=correct,
                tokens=lens[vi],
                request_index=vi,
                extra={
                    'action': match['action'],
                    'template_aliases': match.get('template_aliases', []),
                    'template_tokens': match['template_tokens'],
                    'lo': match['lo'],
                    'hi': match['hi'],
                    'source': 'live_graph_replay',
                    'trusted_graph_replay': True,
                    'same_engine': True,
                    'shadow_engine': True,
                    'validation_mode': 'isolated_same_engine_probe',
                    'fallback_config': 'forced_runtime_fallback',
                    'candidate_capture_ms': capture_ms,
                    'useful': bool(useful),
                    'graph_output_token_ids': graph_tokens,
                    'fallback_output_token_ids': fallback_tokens,
                    'graph_output_text': graph_text,
                    'fallback_output_text': fallback_text,
                },
            )
            print(
                f"    isolated [{vi+1}/{validate_n}] len={lens[vi]} "
                f"graph={graph_ms:.2f} ms fallback={fallback_ms:.2f} ms "
                f"correct={int(correct)} useful={int(useful)}"
            )
    finally:
        _restore_env(saved)


def run_config(
    model,
    prompts,
    lens,
    name,
    tp,
    max_model_len,
    gmu,
    eager=False,
    chunked=False,
    cap_sizes=None,
    max_cap=None,
    max_tokens=1,
    profile_prefix=None,
    enable_return_routed_experts=False,
    runtime_policy=None,
    runtime_base_capture_size=512,
    cudagraph_mode='FULL_AND_PIECEWISE',
    max_num_seqs=0,
    template_scheduler=False,
    template_scheduler_max_wait_ms=0.0,
    template_scheduler_max_scan=16,
    batch_mode=False,
    fixed_metadata_arena=False,
    fixed_metadata_arena_max_reqs=0,
    fixed_metadata_arena_min_tokens=0,
    fixed_metadata_arena_max_tokens=0,
    full_key_collapse=False,
    live_admission=False,
    live_observations=None,
    live_explore=False,
    live_min_samples=0,
    live_min_useful_rate=0.0,
    live_min_saving_ms=0.0,
    live_max_p95_regression_ms=None,
    live_capture=False,
    live_shadow_graph_replay=False,
    live_observations_clear=False,
    live_observation_source='offline_shadow_baseline',
    live_observation_template_id='exact',
    live_observation_trusted=False,
    live_shadow_rows=None,
    live_same_engine_validate=False,
    live_same_engine_validate_limit=0,
    live_isolated_probe=False,
    live_isolated_probe_timeout_s=0.0,
    moe_capacity_buckets=None,
    disable_prefix_caching=False,
):
    from vllm import LLM, SamplingParams
    old_dispatch_profile = os.environ.get('STATICITY_VLLM_CG_PROFILE')
    old_runner_profile = os.environ.get('STATICITY_VLLM_RUNNER_PROFILE')
    old_attn_profile = os.environ.get('STATICITY_VLLM_ATTN_PROFILE')
    old_moe_profile = os.environ.get('STATICITY_VLLM_MOE_PROFILE')
    old_runtime_policy = os.environ.get('STATICITY_VLLM_RUNTIME_POLICY')
    old_runtime_base_capture_size = os.environ.get('STATICITY_VLLM_BASE_CAPTURE_SIZE')
    old_runtime_active = os.environ.get('STATICITY_VLLM_RUNTIME_ACTIVE')
    old_template_scheduler = os.environ.get('STATICITY_VLLM_TEMPLATE_SCHEDULER')
    old_template_scheduler_policy = os.environ.get('STATICITY_VLLM_TEMPLATE_SCHEDULER_POLICY')
    old_template_scheduler_wait = os.environ.get('STATICITY_VLLM_TEMPLATE_SCHEDULER_MAX_WAIT_MS')
    old_template_scheduler_scan = os.environ.get('STATICITY_VLLM_TEMPLATE_SCHEDULER_MAX_SCAN')
    old_scheduler_profile = os.environ.get('STATICITY_VLLM_SCHEDULER_PROFILE')
    old_fixed_metadata_arena = os.environ.get('STATICITY_VLLM_FIXED_METADATA_ARENA')
    old_fixed_metadata_arena_max_reqs = os.environ.get('STATICITY_VLLM_FIXED_METADATA_ARENA_MAX_REQS')
    old_fixed_metadata_arena_min_tokens = os.environ.get('STATICITY_VLLM_FIXED_METADATA_ARENA_MIN_TOKENS')
    old_fixed_metadata_arena_max_tokens = os.environ.get('STATICITY_VLLM_FIXED_METADATA_ARENA_MAX_TOKENS')
    old_full_key_collapse = os.environ.get('VLLM_FULL_KEY_COLLAPSE')
    old_live_admission = os.environ.get('STATICITY_VLLM_LIVE_ADMISSION')
    old_live_observations = os.environ.get('STATICITY_VLLM_LIVE_OBSERVATIONS')
    old_live_explore = os.environ.get('STATICITY_VLLM_LIVE_EXPLORE')
    old_live_min_samples = os.environ.get('STATICITY_VLLM_LIVE_MIN_SAMPLES')
    old_live_min_useful_rate = os.environ.get('STATICITY_VLLM_LIVE_MIN_USEFUL_RATE')
    old_live_min_saving_ms = os.environ.get('STATICITY_VLLM_LIVE_MIN_SAVING_MS')
    old_live_max_p95_regression_ms = os.environ.get('STATICITY_VLLM_LIVE_MAX_P95_REGRESSION_MS')
    old_live_capture = os.environ.get('STATICITY_VLLM_LIVE_CAPTURE')
    old_allow_runtime_capture = os.environ.get('STATICITY_VLLM_ALLOW_RUNTIME_CUDAGRAPH_CAPTURE')
    old_unsafe_live_explore_replay = os.environ.get('STATICITY_VLLM_UNSAFE_LIVE_EXPLORE_REPLAY')
    old_force_runtime_fallback = os.environ.get('STATICITY_VLLM_FORCE_RUNTIME_FALLBACK')
    old_runtime_control = os.environ.get('STATICITY_VLLM_RUNTIME_CONTROL')
    old_moe_capacity_buckets = os.environ.get('STATICITY_VLLM_MOE_CAPACITY_BUCKETS')
    old_exact_token_templates = os.environ.get('STATICITY_VLLM_EXACT_TOKEN_TEMPLATES')
    old_trust_shadow_positive = os.environ.get('STATICITY_VLLM_TRUST_SHADOW_POSITIVE_OBSERVATIONS')
    old_positive_alias_expansion = os.environ.get('STATICITY_VLLM_POSITIVE_ALIAS_EXPANSION')
    dispatch_profile = None
    runner_profile = None
    attn_profile = None
    moe_profile = None
    scheduler_profile = None
    if profile_prefix:
        safe_name = ''.join(ch.lower() if ch.isalnum() else '_' for ch in name).strip('_')
        dispatch_profile = f'{profile_prefix}_{safe_name}_dispatcher.jsonl'
        runner_profile = f'{profile_prefix}_{safe_name}_runner.jsonl'
        attn_profile = f'{profile_prefix}_{safe_name}_attn.jsonl'
        moe_profile = f'{profile_prefix}_{safe_name}_moe.jsonl'
        scheduler_profile = f'{profile_prefix}_{safe_name}_scheduler.jsonl'
        Path(dispatch_profile).parent.mkdir(parents=True, exist_ok=True)
        Path(dispatch_profile).write_text('')
        Path(runner_profile).write_text('')
        Path(attn_profile).write_text('')
        Path(moe_profile).write_text('')
        Path(scheduler_profile).write_text('')
        os.environ['STATICITY_VLLM_CG_PROFILE'] = dispatch_profile
        os.environ['VLLM_CG_TRACE_FILE'] = dispatch_profile
        os.environ['STATICITY_VLLM_RUNNER_PROFILE'] = runner_profile
        os.environ['STATICITY_VLLM_ATTN_PROFILE'] = attn_profile
        os.environ['STATICITY_VLLM_MOE_PROFILE'] = moe_profile
        os.environ['STATICITY_VLLM_SCHEDULER_PROFILE'] = scheduler_profile
    runtime_control_path = None
    if (live_same_engine_validate or live_isolated_probe) and live_admission:
        control_base = Path(live_observations or dispatch_profile or 'results/staticity_runtime_control.json')
        runtime_control_path = str(control_base.with_suffix(control_base.suffix + '.control.json'))
        _write_runtime_control(
            runtime_control_path,
            unsafe_live_explore_replay=False,
            force_runtime_fallback=False,
        )
        os.environ['STATICITY_VLLM_RUNTIME_CONTROL'] = runtime_control_path
    if runtime_policy:
        os.environ['STATICITY_VLLM_RUNTIME_POLICY'] = runtime_policy
        os.environ['STATICITY_VLLM_BASE_CAPTURE_SIZE'] = str(runtime_base_capture_size)
        os.environ['STATICITY_VLLM_RUNTIME_ACTIVE'] = '1'
    if template_scheduler:
        if not runtime_policy:
            raise ValueError('template scheduler requires runtime_policy')
        os.environ['STATICITY_VLLM_TEMPLATE_SCHEDULER'] = '1'
        os.environ['STATICITY_VLLM_TEMPLATE_SCHEDULER_POLICY'] = runtime_policy
        os.environ['STATICITY_VLLM_TEMPLATE_SCHEDULER_MAX_WAIT_MS'] = str(template_scheduler_max_wait_ms)
        os.environ['STATICITY_VLLM_TEMPLATE_SCHEDULER_MAX_SCAN'] = str(template_scheduler_max_scan)
    if fixed_metadata_arena:
        os.environ['STATICITY_VLLM_FIXED_METADATA_ARENA'] = '1'
        if fixed_metadata_arena_max_reqs:
            os.environ['STATICITY_VLLM_FIXED_METADATA_ARENA_MAX_REQS'] = str(fixed_metadata_arena_max_reqs)
        if fixed_metadata_arena_min_tokens:
            os.environ['STATICITY_VLLM_FIXED_METADATA_ARENA_MIN_TOKENS'] = str(fixed_metadata_arena_min_tokens)
        if fixed_metadata_arena_max_tokens:
            os.environ['STATICITY_VLLM_FIXED_METADATA_ARENA_MAX_TOKENS'] = str(fixed_metadata_arena_max_tokens)
    if full_key_collapse:
        os.environ['VLLM_FULL_KEY_COLLAPSE'] = '1'
    if live_admission:
        os.environ['STATICITY_VLLM_LIVE_ADMISSION'] = '1'
        if live_observations:
            os.environ['STATICITY_VLLM_LIVE_OBSERVATIONS'] = live_observations
            if live_observations_clear:
                obs = Path(live_observations)
                obs.parent.mkdir(parents=True, exist_ok=True)
                obs.write_text('', encoding='utf-8')
        if (live_same_engine_validate or live_isolated_probe) and not live_observations:
            raise ValueError('live graph validation requires live_observations')
        if live_explore:
            os.environ['STATICITY_VLLM_LIVE_EXPLORE'] = '1'
        os.environ['STATICITY_VLLM_LIVE_MIN_SAMPLES'] = str(int(live_min_samples))
        os.environ['STATICITY_VLLM_LIVE_MIN_USEFUL_RATE'] = str(float(live_min_useful_rate))
        os.environ['STATICITY_VLLM_LIVE_MIN_SAVING_MS'] = str(float(live_min_saving_ms))
        if live_max_p95_regression_ms is not None:
            os.environ['STATICITY_VLLM_LIVE_MAX_P95_REGRESSION_MS'] = str(float(live_max_p95_regression_ms))
    if live_same_engine_validate:
        if not live_admission:
            raise ValueError('same-engine live validation requires live_admission')
        if live_isolated_probe:
            raise ValueError('same-engine inline validation and isolated probe are mutually exclusive')
        if not live_capture:
            raise ValueError('same-engine live validation requires live_capture')
        if batch_mode:
            raise ValueError('same-engine live validation currently supports sequential LLM.generate only')
        if not disable_prefix_caching:
            print('  same-engine live validation forces prefix caching off for clean repeated-prompt comparisons')
            disable_prefix_caching = True
    if live_isolated_probe:
        if not live_admission:
            raise ValueError('isolated live probe requires live_admission')
        if not live_capture:
            raise ValueError('isolated live probe requires live_capture')
        if batch_mode:
            raise ValueError('isolated live probe currently supports sequential LLM.generate only')
        if not disable_prefix_caching:
            print('  isolated live probe forces prefix caching off for clean repeated-prompt comparisons')
            disable_prefix_caching = True
    if live_capture:
        os.environ['STATICITY_VLLM_LIVE_CAPTURE'] = '1'
        os.environ['STATICITY_VLLM_ALLOW_RUNTIME_CUDAGRAPH_CAPTURE'] = '1'
    if live_shadow_graph_replay:
        os.environ['STATICITY_VLLM_UNSAFE_LIVE_EXPLORE_REPLAY'] = '1'
    if live_observation_template_id == 'exact':
        os.environ['STATICITY_VLLM_EXACT_TOKEN_TEMPLATES'] = '1'
    if live_observation_source == 'graph_replay_shadow' and not live_shadow_graph_replay:
        os.environ['STATICITY_VLLM_TRUST_SHADOW_POSITIVE_OBSERVATIONS'] = '0'
        os.environ['STATICITY_VLLM_POSITIVE_ALIAS_EXPANSION'] = '0'
    if moe_capacity_buckets:
        os.environ['STATICITY_VLLM_MOE_CAPACITY_BUCKETS'] = moe_capacity_buckets
    isolated_probe_counts = {'returncode': None, 'observations': 0, 'correct': 0, 'useful': 0}
    if live_isolated_probe and live_observations and runtime_policy:
        validate_n = (
            min(len(prompts), int(live_same_engine_validate_limit))
            if int(live_same_engine_validate_limit or 0) > 0
            else len(prompts)
        )
        probe_payload = {
            'probe_payload': str(
                Path(runtime_control_path).with_suffix('.probe.json')
                if runtime_control_path else Path(live_observations).with_suffix('.probe.json')
            ),
            'model': model,
            'prompts': prompts[:validate_n],
            'lens': lens[:validate_n],
            'tp': int(tp),
            'max_model_len': int(max_model_len),
            'gmu': float(gmu),
            'max_tokens': int(max_tokens),
            'cap_sizes': list(cap_sizes or []),
            'runtime_policy': runtime_policy,
            'runtime_base_capture_size': int(runtime_base_capture_size),
            'cudagraph_mode': cudagraph_mode,
            'max_num_seqs': int(max_num_seqs) if max_num_seqs else 0,
            'enable_return_routed_experts': bool(enable_return_routed_experts),
            'fixed_metadata_arena': bool(fixed_metadata_arena),
            'fixed_metadata_arena_max_reqs': int(fixed_metadata_arena_max_reqs or 0),
            'fixed_metadata_arena_min_tokens': int(fixed_metadata_arena_min_tokens or 0),
            'fixed_metadata_arena_max_tokens': int(fixed_metadata_arena_max_tokens or 0),
            'full_key_collapse': bool(full_key_collapse),
            'live_min_samples': int(live_min_samples),
            'live_min_useful_rate': float(live_min_useful_rate),
            'live_min_saving_ms': float(live_min_saving_ms),
            'live_max_p95_regression_ms': live_max_p95_regression_ms,
            'live_observations': live_observations,
            'live_observation_template_id': live_observation_template_id,
            'runtime_control': runtime_control_path,
            'moe_capacity_buckets': moe_capacity_buckets,
            'disable_prefix_caching': True,
            'validate_n': int(validate_n),
            'probe_timeout_s': float(live_isolated_probe_timeout_s or 0.0),
        }
        print(f'  isolated live probe: n={validate_n}')
        isolated_probe_counts = _run_isolated_live_probe(probe_payload)
        print(
            '  isolated live probe result: '
            f"rc={isolated_probe_counts['returncode']} "
            f"obs={isolated_probe_counts['observations']} "
            f"correct={isolated_probe_counts['correct']} "
            f"useful={isolated_probe_counts['useful']}"
        )
    cc = {}
    if not eager:
        cc['cudagraph_mode'] = cudagraph_mode
        if cap_sizes:
            cc['cudagraph_capture_sizes'] = cap_sizes
        if max_cap:
            cc['max_cudagraph_capture_size'] = max_cap
    kw = dict(
        model=model,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        gpu_memory_utilization=gmu,
        enforce_eager=eager,
        disable_log_stats=True,
        enable_chunked_prefill=chunked,
        enable_return_routed_experts=enable_return_routed_experts,
    )
    if disable_prefix_caching:
        kw['enable_prefix_caching'] = False
    if not chunked:
        kw['max_num_batched_tokens'] = max_model_len
    if max_num_seqs:
        kw['max_num_seqs'] = max_num_seqs
    if not eager:
        kw['compilation_config'] = cc
    print(f"\n=== {name} === eager={eager} chunked={chunked} cg={cudagraph_mode if not eager else 'NONE'} max_num_seqs={max_num_seqs or 'default'} template_scheduler={template_scheduler} batch_mode={batch_mode}")
    if cap_sizes:
        print(f'capture sizes={len(cap_sizes)} max={max(cap_sizes)}')
    t0 = time.monotonic()
    try:
        llm = LLM(**kw)
        init_s = time.monotonic() - t0
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        _ = llm.generate([prompts[0]], sp)
        ttfts = []
        outputs = []
        live_shadow_written = 0
        live_same_engine_written = 0
        live_same_engine_correct = 0
        live_same_engine_useful = 0
        live_same_engine_ms = 0.0
        batch_total_ms = None
        if (
            live_same_engine_validate
            and live_observations
            and runtime_policy
            and not batch_mode
        ):
            validate_n = (
                min(len(prompts), int(live_same_engine_validate_limit))
                if int(live_same_engine_validate_limit or 0) > 0
                else len(prompts)
            )
            print(f'  same-engine live validation: n={validate_n}')
            for vi, prompt in enumerate(prompts[:validate_n]):
                match = runtime_policy_graph_template(
                    runtime_policy,
                    lens[vi],
                    num_reqs=1,
                    template_id_style=live_observation_template_id,
                )
                if not match:
                    continue
                graph_env = _set_env_temporarily({
                    'STATICITY_VLLM_RUNTIME_ACTIVE': '1',
                    'STATICITY_VLLM_LIVE_EXPLORE': '1',
                    'STATICITY_VLLM_LIVE_MIN_SAMPLES': str(max(1, int(live_min_samples))),
                    'STATICITY_VLLM_UNSAFE_LIVE_EXPLORE_REPLAY': '1',
                    'STATICITY_VLLM_FORCE_RUNTIME_FALLBACK': None,
                })
                try:
                    _write_runtime_control(
                        runtime_control_path,
                        unsafe_live_explore_replay=True,
                        force_runtime_fallback=False,
                    )
                    capture_ms, _, _ = _timed_generate_one(llm, prompt, sp)
                    graph_ms, graph_tokens, graph_text = _timed_generate_one(llm, prompt, sp)
                finally:
                    _write_runtime_control(
                        runtime_control_path,
                        unsafe_live_explore_replay=False,
                        force_runtime_fallback=False,
                    )
                    _restore_env(graph_env)
                fallback_env = _set_env_temporarily({
                    'STATICITY_VLLM_RUNTIME_ACTIVE': '1',
                    'STATICITY_VLLM_UNSAFE_LIVE_EXPLORE_REPLAY': None,
                    'STATICITY_VLLM_FORCE_RUNTIME_FALLBACK': '1',
                })
                try:
                    _write_runtime_control(
                        runtime_control_path,
                        unsafe_live_explore_replay=False,
                        force_runtime_fallback=True,
                    )
                    fallback_ms, fallback_tokens, fallback_text = _timed_generate_one(llm, prompt, sp)
                finally:
                    _write_runtime_control(
                        runtime_control_path,
                        unsafe_live_explore_replay=False,
                        force_runtime_fallback=False,
                    )
                    _restore_env(fallback_env)
                correct = graph_tokens == fallback_tokens
                useful = correct and graph_ms < fallback_ms
                append_live_admission_observation(
                    live_observations,
                    template_id=match['template_id'],
                    graph_ms=graph_ms,
                    fallback_ms=fallback_ms,
                    correct=correct,
                    tokens=lens[vi],
                    request_index=vi,
                    extra={
                        'action': match['action'],
                        'template_aliases': match.get('template_aliases', []),
                        'template_tokens': match['template_tokens'],
                        'lo': match['lo'],
                        'hi': match['hi'],
                        'source': 'live_graph_replay',
                        'trusted_graph_replay': True,
                        'same_engine': True,
                        'shadow_engine': False,
                        'validation_mode': 'same_engine_two_pass',
                        'fallback_config': 'forced_runtime_fallback',
                        'candidate_capture_ms': capture_ms,
                        'useful': bool(useful),
                        'graph_output_token_ids': graph_tokens,
                        'fallback_output_token_ids': fallback_tokens,
                        'graph_output_text': graph_text,
                        'fallback_output_text': fallback_text,
                    },
                )
                live_same_engine_written += 1
                live_same_engine_correct += int(correct)
                live_same_engine_useful += int(useful)
                live_same_engine_ms += capture_ms + graph_ms + fallback_ms
                print(
                    f"    [{vi+1}/{validate_n}] len={lens[vi]} "
                    f"graph={graph_ms:.2f} ms fallback={fallback_ms:.2f} ms "
                    f"correct={int(correct)} useful={int(useful)}"
                )
        if batch_mode:
            torch.cuda.synchronize()
            start = time.perf_counter()
            batch_outputs = llm.generate(prompts, sp, use_tqdm=False)
            torch.cuda.synchronize()
            batch_ms = (time.perf_counter() - start) * 1000
            batch_total_ms = batch_ms
            per_req_ms = batch_ms / max(1, len(prompts))
            for i, request_output in enumerate(batch_outputs):
                ttfts.append(per_req_ms)
                output = request_output.outputs[0] if request_output.outputs else None
                token_ids = list(output.token_ids) if output is not None else []
                text = output.text if output is not None else ''
                outputs.append({'token_ids': [int(x) for x in token_ids], 'text': text})
                if live_shadow_rows is not None and live_observations and runtime_policy:
                    match = runtime_policy_graph_template(
                        runtime_policy,
                        lens[i],
                        num_reqs=1,
                        template_id_style=live_observation_template_id,
                    )
                    if match and i < len(live_shadow_rows):
                        ref = live_shadow_rows[i]
                        ref_tokens = ref.get('output_token_ids', [])
                        correct = [int(x) for x in token_ids] == [int(x) for x in ref_tokens]
                        append_live_admission_observation(
                            live_observations,
                            template_id=match['template_id'],
                            graph_ms=per_req_ms,
                            fallback_ms=float(ref['ms']),
                            correct=correct,
                            tokens=lens[i],
                            request_index=i,
                            extra={
                                'action': match['action'],
                                'template_aliases': match.get('template_aliases', []),
                                'template_tokens': match['template_tokens'],
                                'lo': match['lo'],
                                'hi': match['hi'],
                                'source': live_observation_source,
                                'trusted_graph_replay': bool(live_observation_trusted),
                                'shadow_engine': bool(live_shadow_graph_replay),
                                'fallback_config': ref.get('config', 'shadow_baseline'),
                            },
                        )
                        live_shadow_written += 1
                print(f"  [{i+1}/{len(prompts)}] len={lens[i]} batch_total={batch_ms:.2f} ms amortized={per_req_ms:.2f} ms")
        else:
            for i, prompt in enumerate(prompts):
                torch.cuda.synchronize()
                start = time.perf_counter()
                request_outputs = llm.generate([prompt], sp)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - start) * 1000
                ttfts.append(ms)
                output = request_outputs[0].outputs[0] if request_outputs and request_outputs[0].outputs else None
                token_ids = list(output.token_ids) if output is not None else []
                text = output.text if output is not None else ''
                outputs.append({'token_ids': [int(x) for x in token_ids], 'text': text})
                if live_shadow_rows is not None and live_observations and runtime_policy:
                    match = runtime_policy_graph_template(
                        runtime_policy,
                        lens[i],
                        num_reqs=1,
                        template_id_style=live_observation_template_id,
                    )
                    if match and i < len(live_shadow_rows):
                        ref = live_shadow_rows[i]
                        ref_tokens = ref.get('output_token_ids', [])
                        correct = [int(x) for x in token_ids] == [int(x) for x in ref_tokens]
                        append_live_admission_observation(
                            live_observations,
                            template_id=match['template_id'],
                            graph_ms=ms,
                            fallback_ms=float(ref['ms']),
                            correct=correct,
                            tokens=lens[i],
                            request_index=i,
                            extra={
                                'action': match['action'],
                                'template_aliases': match.get('template_aliases', []),
                                'template_tokens': match['template_tokens'],
                                'lo': match['lo'],
                                'hi': match['hi'],
                                'source': live_observation_source,
                                'trusted_graph_replay': bool(live_observation_trusted),
                                'shadow_engine': bool(live_shadow_graph_replay),
                                'fallback_config': ref.get('config', 'shadow_baseline'),
                            },
                        )
                        live_shadow_written += 1
                print(f"  [{i+1}/{len(prompts)}] len={lens[i]} {ms:.2f} ms")
        del llm
    finally:
        if profile_prefix:
            if old_dispatch_profile is None:
                os.environ.pop('STATICITY_VLLM_CG_PROFILE', None)
                os.environ.pop('VLLM_CG_TRACE_FILE', None)
            else:
                os.environ['STATICITY_VLLM_CG_PROFILE'] = old_dispatch_profile
                os.environ['VLLM_CG_TRACE_FILE'] = old_dispatch_profile
            if old_runner_profile is None:
                os.environ.pop('STATICITY_VLLM_RUNNER_PROFILE', None)
            else:
                os.environ['STATICITY_VLLM_RUNNER_PROFILE'] = old_runner_profile
            if old_attn_profile is None:
                os.environ.pop('STATICITY_VLLM_ATTN_PROFILE', None)
            else:
                os.environ['STATICITY_VLLM_ATTN_PROFILE'] = old_attn_profile
            if old_moe_profile is None:
                os.environ.pop('STATICITY_VLLM_MOE_PROFILE', None)
            else:
                os.environ['STATICITY_VLLM_MOE_PROFILE'] = old_moe_profile
            if old_scheduler_profile is None:
                os.environ.pop('STATICITY_VLLM_SCHEDULER_PROFILE', None)
            else:
                os.environ['STATICITY_VLLM_SCHEDULER_PROFILE'] = old_scheduler_profile
        if old_runtime_policy is None:
            os.environ.pop('STATICITY_VLLM_RUNTIME_POLICY', None)
        else:
            os.environ['STATICITY_VLLM_RUNTIME_POLICY'] = old_runtime_policy
        if old_runtime_base_capture_size is None:
            os.environ.pop('STATICITY_VLLM_BASE_CAPTURE_SIZE', None)
        else:
            os.environ['STATICITY_VLLM_BASE_CAPTURE_SIZE'] = old_runtime_base_capture_size
        if old_runtime_active is None:
            os.environ.pop('STATICITY_VLLM_RUNTIME_ACTIVE', None)
        else:
            os.environ['STATICITY_VLLM_RUNTIME_ACTIVE'] = old_runtime_active
        if old_template_scheduler is None:
            os.environ.pop('STATICITY_VLLM_TEMPLATE_SCHEDULER', None)
        else:
            os.environ['STATICITY_VLLM_TEMPLATE_SCHEDULER'] = old_template_scheduler
        if old_template_scheduler_policy is None:
            os.environ.pop('STATICITY_VLLM_TEMPLATE_SCHEDULER_POLICY', None)
        else:
            os.environ['STATICITY_VLLM_TEMPLATE_SCHEDULER_POLICY'] = old_template_scheduler_policy
        if old_template_scheduler_wait is None:
            os.environ.pop('STATICITY_VLLM_TEMPLATE_SCHEDULER_MAX_WAIT_MS', None)
        else:
            os.environ['STATICITY_VLLM_TEMPLATE_SCHEDULER_MAX_WAIT_MS'] = old_template_scheduler_wait
        if old_template_scheduler_scan is None:
            os.environ.pop('STATICITY_VLLM_TEMPLATE_SCHEDULER_MAX_SCAN', None)
        else:
            os.environ['STATICITY_VLLM_TEMPLATE_SCHEDULER_MAX_SCAN'] = old_template_scheduler_scan
        if old_fixed_metadata_arena is None:
            os.environ.pop('STATICITY_VLLM_FIXED_METADATA_ARENA', None)
        else:
            os.environ['STATICITY_VLLM_FIXED_METADATA_ARENA'] = old_fixed_metadata_arena
        if old_fixed_metadata_arena_max_reqs is None:
            os.environ.pop('STATICITY_VLLM_FIXED_METADATA_ARENA_MAX_REQS', None)
        else:
            os.environ['STATICITY_VLLM_FIXED_METADATA_ARENA_MAX_REQS'] = old_fixed_metadata_arena_max_reqs
        if old_fixed_metadata_arena_min_tokens is None:
            os.environ.pop('STATICITY_VLLM_FIXED_METADATA_ARENA_MIN_TOKENS', None)
        else:
            os.environ['STATICITY_VLLM_FIXED_METADATA_ARENA_MIN_TOKENS'] = old_fixed_metadata_arena_min_tokens
        if old_fixed_metadata_arena_max_tokens is None:
            os.environ.pop('STATICITY_VLLM_FIXED_METADATA_ARENA_MAX_TOKENS', None)
        else:
            os.environ['STATICITY_VLLM_FIXED_METADATA_ARENA_MAX_TOKENS'] = old_fixed_metadata_arena_max_tokens
        if old_full_key_collapse is None:
            os.environ.pop('VLLM_FULL_KEY_COLLAPSE', None)
        else:
            os.environ['VLLM_FULL_KEY_COLLAPSE'] = old_full_key_collapse
        if old_live_admission is None:
            os.environ.pop('STATICITY_VLLM_LIVE_ADMISSION', None)
        else:
            os.environ['STATICITY_VLLM_LIVE_ADMISSION'] = old_live_admission
        if old_live_observations is None:
            os.environ.pop('STATICITY_VLLM_LIVE_OBSERVATIONS', None)
        else:
            os.environ['STATICITY_VLLM_LIVE_OBSERVATIONS'] = old_live_observations
        if old_live_explore is None:
            os.environ.pop('STATICITY_VLLM_LIVE_EXPLORE', None)
        else:
            os.environ['STATICITY_VLLM_LIVE_EXPLORE'] = old_live_explore
        if old_live_min_samples is None:
            os.environ.pop('STATICITY_VLLM_LIVE_MIN_SAMPLES', None)
        else:
            os.environ['STATICITY_VLLM_LIVE_MIN_SAMPLES'] = old_live_min_samples
        if old_live_min_useful_rate is None:
            os.environ.pop('STATICITY_VLLM_LIVE_MIN_USEFUL_RATE', None)
        else:
            os.environ['STATICITY_VLLM_LIVE_MIN_USEFUL_RATE'] = old_live_min_useful_rate
        if old_live_min_saving_ms is None:
            os.environ.pop('STATICITY_VLLM_LIVE_MIN_SAVING_MS', None)
        else:
            os.environ['STATICITY_VLLM_LIVE_MIN_SAVING_MS'] = old_live_min_saving_ms
        if old_live_max_p95_regression_ms is None:
            os.environ.pop('STATICITY_VLLM_LIVE_MAX_P95_REGRESSION_MS', None)
        else:
            os.environ['STATICITY_VLLM_LIVE_MAX_P95_REGRESSION_MS'] = old_live_max_p95_regression_ms
        if old_live_capture is None:
            os.environ.pop('STATICITY_VLLM_LIVE_CAPTURE', None)
        else:
            os.environ['STATICITY_VLLM_LIVE_CAPTURE'] = old_live_capture
        if old_allow_runtime_capture is None:
            os.environ.pop('STATICITY_VLLM_ALLOW_RUNTIME_CUDAGRAPH_CAPTURE', None)
        else:
            os.environ['STATICITY_VLLM_ALLOW_RUNTIME_CUDAGRAPH_CAPTURE'] = old_allow_runtime_capture
        if old_unsafe_live_explore_replay is None:
            os.environ.pop('STATICITY_VLLM_UNSAFE_LIVE_EXPLORE_REPLAY', None)
        else:
            os.environ['STATICITY_VLLM_UNSAFE_LIVE_EXPLORE_REPLAY'] = old_unsafe_live_explore_replay
        if old_force_runtime_fallback is None:
            os.environ.pop('STATICITY_VLLM_FORCE_RUNTIME_FALLBACK', None)
        else:
            os.environ['STATICITY_VLLM_FORCE_RUNTIME_FALLBACK'] = old_force_runtime_fallback
        if old_runtime_control is None:
            os.environ.pop('STATICITY_VLLM_RUNTIME_CONTROL', None)
        else:
            os.environ['STATICITY_VLLM_RUNTIME_CONTROL'] = old_runtime_control
        if old_moe_capacity_buckets is None:
            os.environ.pop('STATICITY_VLLM_MOE_CAPACITY_BUCKETS', None)
        else:
            os.environ['STATICITY_VLLM_MOE_CAPACITY_BUCKETS'] = old_moe_capacity_buckets
        if old_exact_token_templates is None:
            os.environ.pop('STATICITY_VLLM_EXACT_TOKEN_TEMPLATES', None)
        else:
            os.environ['STATICITY_VLLM_EXACT_TOKEN_TEMPLATES'] = old_exact_token_templates
        if old_trust_shadow_positive is None:
            os.environ.pop('STATICITY_VLLM_TRUST_SHADOW_POSITIVE_OBSERVATIONS', None)
        else:
            os.environ['STATICITY_VLLM_TRUST_SHADOW_POSITIVE_OBSERVATIONS'] = old_trust_shadow_positive
        if old_positive_alias_expansion is None:
            os.environ.pop('STATICITY_VLLM_POSITIVE_ALIAS_EXPANSION', None)
        else:
            os.environ['STATICITY_VLLM_POSITIVE_ALIAS_EXPANSION'] = old_positive_alias_expansion
    gc.collect(); torch.cuda.empty_cache(); time.sleep(2)
    arr = np.array(ttfts)
    return {
        'config': name,
        'init_s': init_s,
        'avg_ms': float(arr.mean()),
        'p50_ms': float(np.percentile(arr, 50)),
        'p95_ms': float(np.percentile(arr, 95)),
        'p99_ms': float(np.percentile(arr, 99)),
        'batch_total_ms': float(batch_total_ms) if batch_total_ms is not None else None,
        'per_req': [
            {
                'tok': int(length),
                'ms': float(latency),
                'output_token_ids': output['token_ids'],
                'output_text': output['text'],
            }
            for length, latency, output in zip(lens, ttfts, outputs)
        ],
        'chunked': chunked,
        'eager': eager,
        'capture_sizes': [int(x) for x in cap_sizes] if cap_sizes else None,
        'runtime_policy': runtime_policy,
        'runtime_base_capture_size': runtime_base_capture_size if runtime_policy else None,
        'cudagraph_mode': cudagraph_mode if not eager else 'NONE',
        'max_num_seqs': int(max_num_seqs) if max_num_seqs else None,
        'template_scheduler': bool(template_scheduler),
        'template_scheduler_max_wait_ms': float(template_scheduler_max_wait_ms),
        'template_scheduler_max_scan': int(template_scheduler_max_scan),
        'batch_mode': bool(batch_mode),
        'fixed_metadata_arena': bool(fixed_metadata_arena),
        'fixed_metadata_arena_max_reqs': int(fixed_metadata_arena_max_reqs),
        'fixed_metadata_arena_min_tokens': int(fixed_metadata_arena_min_tokens),
        'fixed_metadata_arena_max_tokens': int(fixed_metadata_arena_max_tokens),
        'full_key_collapse': bool(full_key_collapse),
        'live_admission': bool(live_admission),
        'live_observations': live_observations,
        'live_explore': bool(live_explore),
        'live_min_samples': int(live_min_samples),
        'live_min_useful_rate': float(live_min_useful_rate),
        'live_min_saving_ms': float(live_min_saving_ms),
        'live_max_p95_regression_ms': float(live_max_p95_regression_ms) if live_max_p95_regression_ms is not None else None,
        'live_capture': bool(live_capture),
        'live_shadow_graph_replay': bool(live_shadow_graph_replay),
        'live_observation_source': live_observation_source,
        'live_observation_template_id': live_observation_template_id,
        'live_observation_trusted': bool(live_observation_trusted),
        'live_shadow_observations_written': int(live_shadow_written),
        'live_same_engine_validate': bool(live_same_engine_validate),
        'live_same_engine_validate_limit': int(live_same_engine_validate_limit),
        'live_same_engine_observations_written': int(live_same_engine_written),
        'live_same_engine_correct': int(live_same_engine_correct),
        'live_same_engine_useful': int(live_same_engine_useful),
        'live_same_engine_validation_ms': float(live_same_engine_ms),
        'live_isolated_probe': bool(live_isolated_probe),
        'live_isolated_probe_returncode': isolated_probe_counts.get('returncode'),
        'live_isolated_probe_observations': int(isolated_probe_counts.get('observations') or 0),
        'live_isolated_probe_correct': int(isolated_probe_counts.get('correct') or 0),
        'live_isolated_probe_useful': int(isolated_probe_counts.get('useful') or 0),
        'runtime_control': runtime_control_path,
        'moe_capacity_buckets': moe_capacity_buckets,
        'dispatcher_profile': dispatch_profile,
        'runner_profile': runner_profile,
        'attention_profile': attn_profile,
        'moe_profile': moe_profile,
        'disable_prefix_caching': bool(disable_prefix_caching),
        'scheduler_profile': scheduler_profile,
    }


def summarize_ranges(result):
    lens = np.array([x['tok'] for x in result['per_req']])
    ms = np.array([x['ms'] for x in result['per_req']])
    out = {}
    for lo, hi in [(0,512),(512,1024),(1024,2048),(2048,4096),(4096,8192),(8192,32768)]:
        mask = (lens > lo) & (lens <= hi)
        if mask.any():
            vals = ms[mask]
            out[f'({lo},{hi}]'] = {
                'n': int(mask.sum()),
                'avg_ms': float(vals.mean()),
                'p95_ms': float(np.percentile(vals, 95)),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--live-admission-probe-worker', default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument('--workload', default='results/flowprefill_morspec_workload_16.json')
    ap.add_argument('--model', default='/mnt/models/Meta-Llama-3-8B-Instruct')
    ap.add_argument('--tp-size', type=int, default=1)
    ap.add_argument('--max-model-len', type=int, default=4096)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.8)
    ap.add_argument('--our-max', type=int, default=2048)
    ap.add_argument('--max-buckets', type=int, default=64)
    ap.add_argument('--memory-budget-mb', type=float, default=2048.0)
    ap.add_argument('--warmup-budget-s', type=float, default=30.0)
    ap.add_argument('--max-tokens', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0,
                    help='limit the number of workload requests for smoke tests; 0 uses all requests')
    ap.add_argument('--max-num-seqs', type=int, default=0,
                    help='optional vLLM max_num_seqs guardrail; useful to reduce sampler warmup memory for large models')
    ap.add_argument('--planner-mode', choices=['dp','hybrid','safe'], default='dp',
                    help='dp=maximize token coverage; hybrid=keep vLLM small graphs and add DP long graphs; safe=latency-aware guardrail that only uses proven vLLM small graphs')
    ap.add_argument('--extra-capture-sizes', default=None,
                    help='comma-separated explicit capture sizes to add above vLLM defaults; overrides DP long-size choice for ours')
    ap.add_argument('--max-extra-capture-size', type=int, default=0,
                    help='drop explicit/DP extra capture sizes above this guardrail; 0 disables')
    ap.add_argument('--skip-eager', action='store_true')
    ap.add_argument('--configs', default=None,
                    help='comma-separated subset to run: eager,default,piecewise,full,ours,cp,piecewise_cp,full_cp,ours_cp,runtime,runtime_cp,runtime_piecewise_cp,runtime_full_cp; default runs eager unless skipped, plus default/ours/cp')
    ap.add_argument('--output', default=None)
    ap.add_argument('--profile-prefix', default=None,
                    help='write vLLM graph-key profiler JSONL files with this prefix')
    ap.add_argument('--enable-return-routed-experts', action='store_true',
                    help='enable vLLM MoE routed-expert capture so STATICITY_VLLM_MOE_PROFILE has events')
    ap.add_argument('--runtime-policy', default=None,
                    help='runtime policy JSON consumed inside vLLM cudagraph dispatcher for single-engine graph/fallback decisions')
    ap.add_argument('--runtime-base-capture-size', type=int, default=512,
                    help='default vLLM graph ceiling used by single-engine runtime policy for default/cp actions')
    ap.add_argument('--cudagraph-mode', default='FULL_AND_PIECEWISE',
                    choices=['PIECEWISE', 'FULL', 'FULL_DECODE_ONLY', 'FULL_AND_PIECEWISE'],
                    help='cudagraph mode for default/ours/cp/runtime configs unless a config name explicitly fixes it')
    ap.add_argument('--template-scheduler', action='store_true',
                    help='enable single-engine env-gated template-aware scheduler inside vLLM')
    ap.add_argument('--template-scheduler-max-wait-ms', type=float, default=0.0,
                    help='maximum head-of-line wait tolerated by template-aware scheduler')
    ap.add_argument('--template-scheduler-max-scan', type=int, default=16,
                    help='maximum FCFS waiting-queue entries scanned by template-aware scheduler')
    ap.add_argument('--batch-mode', action='store_true',
                    help='submit all prompts in one LLM.generate call so vLLM internal scheduler sees a real waiting queue')
    ap.add_argument('--allow-batch-extra-capture', action='store_true',
                    help='allow runtime batch-mode to use capture sizes above --runtime-base-capture-size; off by default because multi-request extra graphs require proven metadata key collapse')
    ap.add_argument('--fixed-metadata-arena', action='store_true',
                    help='enable fixed-address request metadata arena for runtime/full-key-collapse experiments')
    ap.add_argument('--fixed-metadata-arena-max-reqs', type=int, default=0,
                    help='request-axis arena width; 0 uses vLLM max_num_seqs')
    ap.add_argument('--fixed-metadata-arena-min-tokens', type=int, default=0,
                    help='only enable fixed-address metadata arena for requests above this token count; 0 disables the lower bound')
    ap.add_argument('--fixed-metadata-arena-max-tokens', type=int, default=0,
                    help='only enable fixed-address metadata arena for requests at or below this token count; 0 disables the upper bound')
    ap.add_argument('--full-key-collapse', action='store_true',
                    help='remove num_reqs from FULL mixed-prefill graph key; requires --fixed-metadata-arena unless unsafe env is set')
    ap.add_argument('--online-admission-refresh-output', default=None,
                    help='after running baseline+runtime configs, write a refreshed online self-learning runtime policy for the next run')
    ap.add_argument('--live-admission', action='store_true',
                    help='enable live self-learning admission inside the vLLM cudagraph dispatcher hot path')
    ap.add_argument('--live-admission-observations', default=None,
                    help='optional JSONL observation stream consumed by live admission: template_id, graph_ms, fallback_ms, correct')
    ap.add_argument('--live-admission-clear-observations', action='store_true',
                    help='clear --live-admission-observations at the start of configs that write shadow observations')
    ap.add_argument('--live-admission-trusted-shadow', action='store_true',
                    help='when paired with runtime_*_shadow configs, label observations as graph_replay_shadow/trusted_graph_replay; strict dispatcher still trusts these positives only if STATICITY_VLLM_TRUST_SHADOW_POSITIVE_OBSERVATIONS=1')
    ap.add_argument('--live-admission-template-id', choices=['bucket', 'range', 'exact'], default='exact',
                    help='template_id style for written live observations; exact avoids range/bucket-level over-admission')
    ap.add_argument('--live-admission-explore', action='store_true',
                    help='allow live admission to explore graph templates until min_samples is reached')
    ap.add_argument('--live-admission-min-samples', type=int, default=0,
                    help='minimum per-template live observations before graph admission can leave exploration')
    ap.add_argument('--live-admission-min-useful-rate', type=float, default=0.0,
                    help='minimum useful observation ratio for live admission')
    ap.add_argument('--live-admission-min-saving-ms', type=float, default=0.0,
                    help='minimum EWMA latency saving for live admission')
    ap.add_argument('--live-admission-max-p95-regression-ms', type=float, default=None,
                    help='maximum observed p95 regression tolerated by live admission')
    ap.add_argument('--live-admission-shadow-baseline', action='store_true',
                    help='stream fallback-vs-graph observations from the already-run baseline rows into the live admission file after each request')
    ap.add_argument('--live-capture', action='store_true',
                    help='enable same-engine capture machinery; strict live admission still keeps unvalidated templates on fallback by default')
    ap.add_argument('--live-admission-same-engine-validate', action='store_true',
                    help='before measured requests, validate candidate graph replay against forced fallback in the same LLM engine and stream trusted live_graph_replay observations')
    ap.add_argument('--live-admission-same-engine-validate-limit', type=int, default=0,
                    help='number of prompts to use for same-engine live validation; 0 validates the full selected workload')
    ap.add_argument('--live-admission-isolated-probe', action='store_true',
                    help='validate unsafe graph replay in an isolated worker before starting the measured serving engine; probe crashes become trusted negative observations')
    ap.add_argument('--live-admission-isolated-probe-timeout-s', type=float, default=0.0,
                    help='timeout for --live-admission-isolated-probe; 0 disables timeout')
    ap.add_argument('--moe-capacity-buckets', default=None,
                    help='comma-separated MoE expert capacity buckets used by routed-expert metadata profiling')
    ap.add_argument('--disable-prefix-caching', action='store_true',
                    help='disable vLLM prefix caching for cleaner cross-engine token-correctness comparisons')
    ap.add_argument('--online-admission-baseline-contains', default='vLLM graph max512 CP',
                    help='baseline config substring used by --online-admission-refresh-output')
    ap.add_argument('--online-admission-candidate-contains', default='Single-engine runtime',
                    help='candidate config substring used by --online-admission-refresh-output')
    ap.add_argument('--online-admission-template-buckets', default=None,
                    help='comma-separated template buckets for online admission refresh; defaults to runtime capture sizes above base')
    ap.add_argument('--online-admission-min-samples', type=int, default=3)
    ap.add_argument('--online-admission-min-useful-rate', type=float, default=0.75)
    ap.add_argument('--online-admission-min-saving-ms', type=float, default=0.5)
    ap.add_argument('--online-admission-max-p95-regression-ms', type=float, default=2.0)
    ap.add_argument('--online-admission-amortization-replays', type=int, default=32)
    args = ap.parse_args()
    if args.live_admission_probe_worker:
        run_live_admission_probe_worker(args.live_admission_probe_worker)
        return

    workload = json.loads(Path(args.workload).read_text(encoding='utf-8'))
    reqs = workload['requests']
    if args.limit:
        reqs = reqs[:args.limit]
    prompts = [r['prompt'] for r in reqs]
    lens = [int(r['actual_input_length']) for r in reqs]
    print(f"workload={args.workload} n={len(prompts)} p50={np.percentile(lens,50):.0f} p95={np.percentile(lens,95):.0f} >512={sum(x>512 for x in lens)} >2048={sum(x>2048 for x in lens)}")
    dp_sizes, dp_plan = exact_dp_sizes(lens, min(args.our_max, args.max_model_len), args.max_buckets, args.memory_budget_mb, args.warmup_budget_s)
    default_small = [s for s in VLLM_DEFAULT_SIZES if s <= 512]
    planner_note = 'pure DP token-coverage planner'
    our_sizes = dp_sizes
    plan_for_report = dp_plan
    explicit_extra = parse_int_list(args.extra_capture_sizes)
    policy_extra = runtime_policy_capture_sizes(
        args.runtime_policy,
        args.runtime_base_capture_size,
    )
    if policy_extra:
        print(f'runtime policy extra capture sizes={policy_extra}')
    if args.planner_mode == 'hybrid':
        long_dp, _ = exact_dp_sizes([x for x in lens if x > 512], min(args.our_max, args.max_model_len), args.max_buckets, args.memory_budget_mb, args.warmup_budget_s)
        extra = explicit_extra if explicit_extra else (policy_extra if policy_extra else [s for s in long_dp if s > 512])
        if args.max_extra_capture_size:
            extra = [s for s in extra if s <= args.max_extra_capture_size]
        our_sizes = sorted(set(default_small + [s for s in extra if 512 < s <= min(args.our_max, args.max_model_len)]))
        planner_note = (
            'vLLM default small graphs plus explicit latency-calibrated long graphs'
            if explicit_extra
            else 'vLLM default small graphs plus DP-selected long graphs'
        )
    elif args.planner_mode == 'safe':
        our_sizes = default_small
        planner_note = 'latency-aware guardrail: keep only vLLM-proven small graphs; long prefill falls back/CP unless calibrated profitable'
    if args.runtime_policy and policy_extra:
        capped_policy_extra = [
            size for size in policy_extra
            if size <= min(args.our_max, args.max_model_len)
            and (not args.max_extra_capture_size or size <= args.max_extra_capture_size)
        ]
        missing_policy_extra = [
            size for size in policy_extra
            if size not in capped_policy_extra
        ]
        if missing_policy_extra:
            print(f'warning: policy templates outside capture guardrail and will fallback if used: {missing_policy_extra}')
        our_sizes = sorted(set(our_sizes + capped_policy_extra))
    print(f'ours sizes={len(our_sizes)} max={max(our_sizes)} mode={args.planner_mode} note={planner_note}')
    runtime_sizes = our_sizes
    runtime_capture_note = planner_note
    arena_batch_extra_allowed = (
        args.allow_batch_extra_capture
        or (args.fixed_metadata_arena and args.full_key_collapse)
    )
    if args.batch_mode and not arena_batch_extra_allowed:
        safe_runtime_sizes = [s for s in default_small if s <= args.runtime_base_capture_size]
        if safe_runtime_sizes and max(our_sizes) > max(safe_runtime_sizes):
            runtime_sizes = safe_runtime_sizes
            runtime_capture_note = (
                f'batch-mode safety guard: runtime capture sizes capped at '
                f'{max(runtime_sizes)} until metadata key-collapse is proven'
            )
            print(f'runtime batch safety guard active: runtime sizes={len(runtime_sizes)} max={max(runtime_sizes)}')
    if dp_plan is not None:
        print(f'dp candidate plan hit={dp_plan.expected_hit_rate*100:.1f}% waste={dp_plan.expected_padding_waste_pct:.2f}% fallback={dp_plan.expected_fallback_count} mem={dp_plan.total_graph_memory_mb:.1f} warmup={dp_plan.total_warmup_time_s:.1f}')

    results = []
    shadow_baseline_rows = None
    out = Path(args.output) if args.output else Path('results') / f"vllm_flowprefill_{Path(args.workload).stem}.json"
    requested = (
        {item.strip() for item in args.configs.split(',') if item.strip()}
        if args.configs
        else ({'default', 'ours', 'cp'} | (set() if args.skip_eager else {'eager'}))
    )
    valid_configs = {
        'eager',
        'default',
        'ours',
        'cp',
        'ours_cp',
        'runtime',
        'runtime_cp',
        'runtime_shadow',
        'runtime_cp_shadow',
        'piecewise',
        'full',
        'piecewise_cp',
        'full_cp',
        'runtime_piecewise_cp',
        'runtime_full_cp',
    }
    unknown = requested - valid_configs
    if unknown:
        raise ValueError(f'unknown --configs entries: {sorted(unknown)}')
    if not results and not requested:
        raise ValueError('no configs selected')

    def save_current(partial):
        if not results:
            return
        base = results[0]['avg_ms']
        reference_result = results[0]
        reference_outputs = [row['output_token_ids'] for row in reference_result['per_req']]
        for result in results:
            result['speedup_vs_first'] = base / result['avg_ms'] if result['avg_ms'] > 0 else None
            result['ranges'] = summarize_ranges(result)
            result['same_outputs_vs_first'] = [
                row['output_token_ids'] == reference
                for row, reference in zip(result['per_req'], reference_outputs)
            ]
            result['all_same_outputs_vs_first'] = all(result['same_outputs_vs_first'])
            result['reference_config'] = reference_result['config']
            result['same_outputs_vs_reference'] = list(result['same_outputs_vs_first'])
            result['all_same_outputs_vs_reference'] = result['all_same_outputs_vs_first']
        output = {
            'partial': partial,
            'workload': args.workload,
            'model': args.model,
            'reference_config': reference_result['config'],
            'planner': {
                'mode': args.planner_mode,
                'note': planner_note,
                'bucket_sizes': [int(x) for x in our_sizes],
                'dp_candidate_bucket_sizes': [int(x) for x in dp_sizes],
                'explicit_extra_capture_sizes': [int(x) for x in explicit_extra],
                'max_extra_capture_size': int(args.max_extra_capture_size),
                'max_num_seqs': int(args.max_num_seqs) if args.max_num_seqs else None,
                'template_scheduler': bool(args.template_scheduler),
                'template_scheduler_max_wait_ms': float(args.template_scheduler_max_wait_ms),
                'template_scheduler_max_scan': int(args.template_scheduler_max_scan),
                'batch_mode': bool(args.batch_mode),
                'allow_batch_extra_capture': bool(arena_batch_extra_allowed),
                'fixed_metadata_arena': bool(args.fixed_metadata_arena),
                'fixed_metadata_arena_max_reqs': int(args.fixed_metadata_arena_max_reqs),
                'full_key_collapse': bool(args.full_key_collapse),
                'runtime_capture_bucket_sizes': [int(x) for x in runtime_sizes],
                'runtime_capture_note': runtime_capture_note,
                'dp_expected_hit_rate': dp_plan.expected_hit_rate if dp_plan is not None else None,
                'dp_expected_padding_waste_pct': dp_plan.expected_padding_waste_pct if dp_plan is not None else None,
                'dp_expected_fallback_count': int(dp_plan.expected_fallback_count) if dp_plan is not None else None,
                'dp_total_graph_memory_mb': dp_plan.total_graph_memory_mb if dp_plan is not None else None,
                'dp_total_warmup_time_s': dp_plan.total_warmup_time_s if dp_plan is not None else None,
            },
            'workload_stats': {
                'n': len(lens),
                'p50': float(np.percentile(lens, 50)),
                'p95': float(np.percentile(lens, 95)),
                'gt512': int(sum(x > 512 for x in lens)),
                'gt2048': int(sum(x > 2048 for x in lens)),
            },
            'lengths': lens,
            'results': results,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2), encoding='utf-8')
        print(f'Saved {"partial" if partial else "final"} to {out}')

    if 'eager' in requested and not args.skip_eager:
        results.append(run_config(args.model, prompts, lens, '1. Eager no-CP', args.tp_size, args.max_model_len, args.gpu_memory_utilization, eager=True, chunked=False, max_tokens=args.max_tokens, profile_prefix=args.profile_prefix, enable_return_routed_experts=args.enable_return_routed_experts, max_num_seqs=args.max_num_seqs, batch_mode=args.batch_mode, disable_prefix_caching=args.disable_prefix_caching))
        save_current(partial=True)
    if 'default' in requested:
        results.append(run_config(args.model, prompts, lens, f'2. vLLM graph max512 no-CP {args.cudagraph_mode}', args.tp_size, args.max_model_len, args.gpu_memory_utilization, chunked=False, max_tokens=args.max_tokens, profile_prefix=args.profile_prefix, enable_return_routed_experts=args.enable_return_routed_experts, cudagraph_mode=args.cudagraph_mode, max_num_seqs=args.max_num_seqs, batch_mode=args.batch_mode, disable_prefix_caching=args.disable_prefix_caching))
        save_current(partial=True)
    if 'piecewise' in requested:
        results.append(run_config(args.model, prompts, lens, '2p. vLLM graph max512 no-CP PIECEWISE', args.tp_size, args.max_model_len, args.gpu_memory_utilization, chunked=False, max_tokens=args.max_tokens, profile_prefix=args.profile_prefix, enable_return_routed_experts=args.enable_return_routed_experts, cudagraph_mode='PIECEWISE', max_num_seqs=args.max_num_seqs, batch_mode=args.batch_mode, disable_prefix_caching=args.disable_prefix_caching))
        save_current(partial=True)
    if 'full' in requested:
        results.append(run_config(args.model, prompts, lens, '2f. vLLM graph max512 no-CP FULL', args.tp_size, args.max_model_len, args.gpu_memory_utilization, chunked=False, max_tokens=args.max_tokens, profile_prefix=args.profile_prefix, enable_return_routed_experts=args.enable_return_routed_experts, cudagraph_mode='FULL', max_num_seqs=args.max_num_seqs, batch_mode=args.batch_mode, disable_prefix_caching=args.disable_prefix_caching))
        save_current(partial=True)
    if 'ours' in requested:
        results.append(run_config(args.model, prompts, lens, f'3. Ours {args.planner_mode} max{args.our_max} no-CP {args.cudagraph_mode}', args.tp_size, args.max_model_len, args.gpu_memory_utilization, chunked=False, cap_sizes=our_sizes, max_cap=max(our_sizes), max_tokens=args.max_tokens, profile_prefix=args.profile_prefix, enable_return_routed_experts=args.enable_return_routed_experts, cudagraph_mode=args.cudagraph_mode, max_num_seqs=args.max_num_seqs, batch_mode=args.batch_mode, disable_prefix_caching=args.disable_prefix_caching))
        save_current(partial=True)
    if 'cp' in requested:
        results.append(run_config(args.model, prompts, lens, f'4. vLLM graph max512 CP {args.cudagraph_mode}', args.tp_size, args.max_model_len, args.gpu_memory_utilization, chunked=True, max_tokens=args.max_tokens, profile_prefix=args.profile_prefix, enable_return_routed_experts=args.enable_return_routed_experts, cudagraph_mode=args.cudagraph_mode, max_num_seqs=args.max_num_seqs, batch_mode=args.batch_mode, disable_prefix_caching=args.disable_prefix_caching))
        shadow_baseline_rows = results[-1]['per_req']
        for row in shadow_baseline_rows:
            row['config'] = results[-1]['config']
        save_current(partial=True)
    if 'piecewise_cp' in requested:
        results.append(run_config(args.model, prompts, lens, '4p. vLLM graph max512 CP PIECEWISE', args.tp_size, args.max_model_len, args.gpu_memory_utilization, chunked=True, max_tokens=args.max_tokens, profile_prefix=args.profile_prefix, enable_return_routed_experts=args.enable_return_routed_experts, cudagraph_mode='PIECEWISE', max_num_seqs=args.max_num_seqs, batch_mode=args.batch_mode, disable_prefix_caching=args.disable_prefix_caching))
        save_current(partial=True)
    if 'full_cp' in requested:
        results.append(run_config(args.model, prompts, lens, '4f. vLLM graph max512 CP FULL', args.tp_size, args.max_model_len, args.gpu_memory_utilization, chunked=True, max_tokens=args.max_tokens, profile_prefix=args.profile_prefix, enable_return_routed_experts=args.enable_return_routed_experts, cudagraph_mode='FULL', max_num_seqs=args.max_num_seqs, batch_mode=args.batch_mode, disable_prefix_caching=args.disable_prefix_caching))
        save_current(partial=True)
    if 'ours_cp' in requested:
        results.append(run_config(args.model, prompts, lens, f'5. Ours {args.planner_mode} max{args.our_max} CP {args.cudagraph_mode}', args.tp_size, args.max_model_len, args.gpu_memory_utilization, chunked=True, cap_sizes=our_sizes, max_cap=max(our_sizes), max_tokens=args.max_tokens, profile_prefix=args.profile_prefix, enable_return_routed_experts=args.enable_return_routed_experts, cudagraph_mode=args.cudagraph_mode, max_num_seqs=args.max_num_seqs, batch_mode=args.batch_mode, disable_prefix_caching=args.disable_prefix_caching))
        save_current(partial=True)
    if 'runtime_shadow' in requested:
        if not args.runtime_policy:
            raise ValueError('--runtime-policy is required when --configs includes runtime_shadow')
        baseline_rows = shadow_baseline_rows or (results[0]['per_req'] if results else None)
        if baseline_rows is None:
            raise ValueError('runtime_shadow requires a previously run baseline config, usually cp')
        results.append(run_config(
            args.model,
            prompts,
            lens,
            f'6s. Shadow-engine runtime {args.planner_mode} max{args.our_max} no-CP',
            args.tp_size,
            args.max_model_len,
            args.gpu_memory_utilization,
            chunked=False,
            cap_sizes=runtime_sizes,
            max_cap=max(runtime_sizes),
            max_tokens=args.max_tokens,
            profile_prefix=args.profile_prefix,
            enable_return_routed_experts=args.enable_return_routed_experts,
            runtime_policy=args.runtime_policy,
            runtime_base_capture_size=args.runtime_base_capture_size,
            cudagraph_mode=args.cudagraph_mode,
            max_num_seqs=args.max_num_seqs,
            template_scheduler=args.template_scheduler,
            template_scheduler_max_wait_ms=args.template_scheduler_max_wait_ms,
            template_scheduler_max_scan=args.template_scheduler_max_scan,
            batch_mode=args.batch_mode,
            fixed_metadata_arena=args.fixed_metadata_arena,
            fixed_metadata_arena_max_reqs=args.fixed_metadata_arena_max_reqs,
            fixed_metadata_arena_min_tokens=args.fixed_metadata_arena_min_tokens,
            fixed_metadata_arena_max_tokens=args.fixed_metadata_arena_max_tokens,
            full_key_collapse=args.full_key_collapse,
            live_admission=True,
            live_observations=args.live_admission_observations,
            live_explore=True,
            live_min_samples=args.live_admission_min_samples,
            live_min_useful_rate=args.live_admission_min_useful_rate,
            live_min_saving_ms=args.live_admission_min_saving_ms,
            live_max_p95_regression_ms=args.live_admission_max_p95_regression_ms,
            live_capture=True,
            live_shadow_graph_replay=True,
            live_observations_clear=args.live_admission_clear_observations,
            live_observation_source=(
                'graph_replay_shadow'
                if args.live_admission_trusted_shadow else 'offline_shadow_baseline'
            ),
            live_observation_template_id=args.live_admission_template_id,
            live_observation_trusted=args.live_admission_trusted_shadow,
            live_shadow_rows=baseline_rows,
            moe_capacity_buckets=args.moe_capacity_buckets,
            disable_prefix_caching=args.disable_prefix_caching,
        ))
        save_current(partial=True)
    if 'runtime' in requested:
        if not args.runtime_policy:
            raise ValueError('--runtime-policy is required when --configs includes runtime')
        results.append(run_config(
            args.model,
            prompts,
            lens,
            f'6. Single-engine runtime {args.planner_mode} max{args.our_max} no-CP',
            args.tp_size,
            args.max_model_len,
            args.gpu_memory_utilization,
            chunked=False,
            cap_sizes=runtime_sizes,
            max_cap=max(runtime_sizes),
            max_tokens=args.max_tokens,
            profile_prefix=args.profile_prefix,
            enable_return_routed_experts=args.enable_return_routed_experts,
            runtime_policy=args.runtime_policy,
            runtime_base_capture_size=args.runtime_base_capture_size,
            cudagraph_mode=args.cudagraph_mode,
            max_num_seqs=args.max_num_seqs,
            template_scheduler=args.template_scheduler,
            template_scheduler_max_wait_ms=args.template_scheduler_max_wait_ms,
            template_scheduler_max_scan=args.template_scheduler_max_scan,
            batch_mode=args.batch_mode,
            fixed_metadata_arena=args.fixed_metadata_arena,
            fixed_metadata_arena_max_reqs=args.fixed_metadata_arena_max_reqs,
            fixed_metadata_arena_min_tokens=args.fixed_metadata_arena_min_tokens,
            fixed_metadata_arena_max_tokens=args.fixed_metadata_arena_max_tokens,
            full_key_collapse=args.full_key_collapse,
            live_admission=args.live_admission,
            live_observations=args.live_admission_observations,
            live_explore=args.live_admission_explore,
            live_min_samples=args.live_admission_min_samples,
            live_min_useful_rate=args.live_admission_min_useful_rate,
            live_min_saving_ms=args.live_admission_min_saving_ms,
            live_max_p95_regression_ms=args.live_admission_max_p95_regression_ms,
            live_capture=args.live_capture,
            live_observation_template_id=args.live_admission_template_id,
            live_shadow_rows=shadow_baseline_rows if args.live_admission_shadow_baseline else None,
            live_same_engine_validate=args.live_admission_same_engine_validate,
            live_same_engine_validate_limit=args.live_admission_same_engine_validate_limit,
            live_isolated_probe=args.live_admission_isolated_probe,
            live_isolated_probe_timeout_s=args.live_admission_isolated_probe_timeout_s,
            moe_capacity_buckets=args.moe_capacity_buckets,
            disable_prefix_caching=args.disable_prefix_caching,
        ))
        save_current(partial=True)
    if 'runtime_cp_shadow' in requested:
        if not args.runtime_policy:
            raise ValueError('--runtime-policy is required when --configs includes runtime_cp_shadow')
        baseline_rows = shadow_baseline_rows or (results[0]['per_req'] if results else None)
        if baseline_rows is None:
            raise ValueError('runtime_cp_shadow requires a previously run baseline config, usually cp')
        results.append(run_config(
            args.model,
            prompts,
            lens,
            f'7s. Shadow-engine runtime {args.planner_mode} max{args.our_max} CP',
            args.tp_size,
            args.max_model_len,
            args.gpu_memory_utilization,
            chunked=True,
            cap_sizes=runtime_sizes,
            max_cap=max(runtime_sizes),
            max_tokens=args.max_tokens,
            profile_prefix=args.profile_prefix,
            enable_return_routed_experts=args.enable_return_routed_experts,
            runtime_policy=args.runtime_policy,
            runtime_base_capture_size=args.runtime_base_capture_size,
            cudagraph_mode=args.cudagraph_mode,
            max_num_seqs=args.max_num_seqs,
            template_scheduler=args.template_scheduler,
            template_scheduler_max_wait_ms=args.template_scheduler_max_wait_ms,
            template_scheduler_max_scan=args.template_scheduler_max_scan,
            batch_mode=args.batch_mode,
            fixed_metadata_arena=args.fixed_metadata_arena,
            fixed_metadata_arena_max_reqs=args.fixed_metadata_arena_max_reqs,
            fixed_metadata_arena_min_tokens=args.fixed_metadata_arena_min_tokens,
            fixed_metadata_arena_max_tokens=args.fixed_metadata_arena_max_tokens,
            full_key_collapse=args.full_key_collapse,
            live_admission=True,
            live_observations=args.live_admission_observations,
            live_explore=True,
            live_min_samples=args.live_admission_min_samples,
            live_min_useful_rate=args.live_admission_min_useful_rate,
            live_min_saving_ms=args.live_admission_min_saving_ms,
            live_max_p95_regression_ms=args.live_admission_max_p95_regression_ms,
            live_capture=True,
            live_shadow_graph_replay=True,
            live_observations_clear=args.live_admission_clear_observations,
            live_observation_source=(
                'graph_replay_shadow'
                if args.live_admission_trusted_shadow else 'offline_shadow_baseline'
            ),
            live_observation_template_id=args.live_admission_template_id,
            live_observation_trusted=args.live_admission_trusted_shadow,
            live_shadow_rows=baseline_rows,
            moe_capacity_buckets=args.moe_capacity_buckets,
            disable_prefix_caching=args.disable_prefix_caching,
        ))
        save_current(partial=True)
    if 'runtime_cp' in requested:
        if not args.runtime_policy:
            raise ValueError('--runtime-policy is required when --configs includes runtime_cp')
        results.append(run_config(
            args.model,
            prompts,
            lens,
            f'7. Single-engine runtime {args.planner_mode} max{args.our_max} CP',
            args.tp_size,
            args.max_model_len,
            args.gpu_memory_utilization,
            chunked=True,
            cap_sizes=runtime_sizes,
            max_cap=max(runtime_sizes),
            max_tokens=args.max_tokens,
            profile_prefix=args.profile_prefix,
            enable_return_routed_experts=args.enable_return_routed_experts,
            runtime_policy=args.runtime_policy,
            runtime_base_capture_size=args.runtime_base_capture_size,
            cudagraph_mode=args.cudagraph_mode,
            max_num_seqs=args.max_num_seqs,
            template_scheduler=args.template_scheduler,
            template_scheduler_max_wait_ms=args.template_scheduler_max_wait_ms,
            template_scheduler_max_scan=args.template_scheduler_max_scan,
            batch_mode=args.batch_mode,
            fixed_metadata_arena=args.fixed_metadata_arena,
            fixed_metadata_arena_max_reqs=args.fixed_metadata_arena_max_reqs,
            fixed_metadata_arena_min_tokens=args.fixed_metadata_arena_min_tokens,
            fixed_metadata_arena_max_tokens=args.fixed_metadata_arena_max_tokens,
            full_key_collapse=args.full_key_collapse,
            live_admission=args.live_admission,
            live_observations=args.live_admission_observations,
            live_explore=args.live_admission_explore,
            live_min_samples=args.live_admission_min_samples,
            live_min_useful_rate=args.live_admission_min_useful_rate,
            live_min_saving_ms=args.live_admission_min_saving_ms,
            live_max_p95_regression_ms=args.live_admission_max_p95_regression_ms,
            live_capture=args.live_capture,
            live_shadow_rows=shadow_baseline_rows if args.live_admission_shadow_baseline else None,
            live_observation_template_id=args.live_admission_template_id,
            live_same_engine_validate=args.live_admission_same_engine_validate,
            live_same_engine_validate_limit=args.live_admission_same_engine_validate_limit,
            live_isolated_probe=args.live_admission_isolated_probe,
            live_isolated_probe_timeout_s=args.live_admission_isolated_probe_timeout_s,
            moe_capacity_buckets=args.moe_capacity_buckets,
            disable_prefix_caching=args.disable_prefix_caching,
        ))
        save_current(partial=True)
    if 'runtime_piecewise_cp' in requested:
        if not args.runtime_policy:
            raise ValueError('--runtime-policy is required when --configs includes runtime_piecewise_cp')
        results.append(run_config(
            args.model,
            prompts,
            lens,
            f'7p. Single-engine runtime {args.planner_mode} max{args.our_max} CP PIECEWISE',
            args.tp_size,
            args.max_model_len,
            args.gpu_memory_utilization,
            chunked=True,
            cap_sizes=runtime_sizes,
            max_cap=max(runtime_sizes),
            max_tokens=args.max_tokens,
            profile_prefix=args.profile_prefix,
            enable_return_routed_experts=args.enable_return_routed_experts,
            runtime_policy=args.runtime_policy,
            runtime_base_capture_size=args.runtime_base_capture_size,
            cudagraph_mode='PIECEWISE',
            max_num_seqs=args.max_num_seqs,
            template_scheduler=args.template_scheduler,
            template_scheduler_max_wait_ms=args.template_scheduler_max_wait_ms,
            template_scheduler_max_scan=args.template_scheduler_max_scan,
            batch_mode=args.batch_mode,
            fixed_metadata_arena=args.fixed_metadata_arena,
            fixed_metadata_arena_max_reqs=args.fixed_metadata_arena_max_reqs,
            fixed_metadata_arena_min_tokens=args.fixed_metadata_arena_min_tokens,
            fixed_metadata_arena_max_tokens=args.fixed_metadata_arena_max_tokens,
            full_key_collapse=args.full_key_collapse,
            live_admission=args.live_admission,
            live_observations=args.live_admission_observations,
            live_explore=args.live_admission_explore,
            live_min_samples=args.live_admission_min_samples,
            live_min_useful_rate=args.live_admission_min_useful_rate,
            live_min_saving_ms=args.live_admission_min_saving_ms,
            live_max_p95_regression_ms=args.live_admission_max_p95_regression_ms,
            live_capture=args.live_capture,
            live_shadow_rows=shadow_baseline_rows if args.live_admission_shadow_baseline else None,
            live_observation_template_id=args.live_admission_template_id,
            live_same_engine_validate=args.live_admission_same_engine_validate,
            live_same_engine_validate_limit=args.live_admission_same_engine_validate_limit,
            live_isolated_probe=args.live_admission_isolated_probe,
            live_isolated_probe_timeout_s=args.live_admission_isolated_probe_timeout_s,
            moe_capacity_buckets=args.moe_capacity_buckets,
            disable_prefix_caching=args.disable_prefix_caching,
        ))
        save_current(partial=True)
    if 'runtime_full_cp' in requested:
        if not args.runtime_policy:
            raise ValueError('--runtime-policy is required when --configs includes runtime_full_cp')
        results.append(run_config(
            args.model,
            prompts,
            lens,
            f'7f. Single-engine runtime {args.planner_mode} max{args.our_max} CP FULL',
            args.tp_size,
            args.max_model_len,
            args.gpu_memory_utilization,
            chunked=True,
            cap_sizes=runtime_sizes,
            max_cap=max(runtime_sizes),
            max_tokens=args.max_tokens,
            profile_prefix=args.profile_prefix,
            enable_return_routed_experts=args.enable_return_routed_experts,
            runtime_policy=args.runtime_policy,
            runtime_base_capture_size=args.runtime_base_capture_size,
            cudagraph_mode='FULL',
            max_num_seqs=args.max_num_seqs,
            template_scheduler=args.template_scheduler,
            template_scheduler_max_wait_ms=args.template_scheduler_max_wait_ms,
            template_scheduler_max_scan=args.template_scheduler_max_scan,
            batch_mode=args.batch_mode,
            fixed_metadata_arena=args.fixed_metadata_arena,
            fixed_metadata_arena_max_reqs=args.fixed_metadata_arena_max_reqs,
            fixed_metadata_arena_min_tokens=args.fixed_metadata_arena_min_tokens,
            fixed_metadata_arena_max_tokens=args.fixed_metadata_arena_max_tokens,
            full_key_collapse=args.full_key_collapse,
            live_admission=args.live_admission,
            live_observations=args.live_admission_observations,
            live_explore=args.live_admission_explore,
            live_min_samples=args.live_admission_min_samples,
            live_min_useful_rate=args.live_admission_min_useful_rate,
            live_min_saving_ms=args.live_admission_min_saving_ms,
            live_max_p95_regression_ms=args.live_admission_max_p95_regression_ms,
            live_capture=args.live_capture,
            live_shadow_rows=shadow_baseline_rows if args.live_admission_shadow_baseline else None,
            live_observation_template_id=args.live_admission_template_id,
            live_same_engine_validate=args.live_admission_same_engine_validate,
            live_same_engine_validate_limit=args.live_admission_same_engine_validate_limit,
            live_isolated_probe=args.live_admission_isolated_probe,
            live_isolated_probe_timeout_s=args.live_admission_isolated_probe_timeout_s,
            moe_capacity_buckets=args.moe_capacity_buckets,
            disable_prefix_caching=args.disable_prefix_caching,
        ))
        save_current(partial=True)

    for result in results:
        print(f"{result['config']}: avg={result['avg_ms']:.2f} p95={result['p95_ms']:.2f} p99={result['p99_ms']:.2f} speedup={result['speedup_vs_first']:.2f} init={result['init_s']:.1f}")
    save_current(partial=False)
    if args.online_admission_refresh_output and len(results) >= 2:
        baseline_snapshot = {
            'results': results,
        }
        buckets = parse_int_list(args.online_admission_template_buckets or '')
        if not buckets:
            buckets = sorted({int(x) for x in runtime_sizes if int(x) > int(args.runtime_base_capture_size)})
        if not buckets:
            buckets = sorted({int(x) for x in runtime_sizes})
        rows = read_rows(
            baseline_snapshot,
            baseline_contains=args.online_admission_baseline_contains,
            candidate_contains=args.online_admission_candidate_contains,
        )
        controller = OnlineSelfLearningAdmissionController(
            min_samples=args.online_admission_min_samples,
            min_useful_rate=args.online_admission_min_useful_rate,
            min_saving_ms=args.online_admission_min_saving_ms,
            max_p95_regression_ms=args.online_admission_max_p95_regression_ms,
            amortization_replays=args.online_admission_amortization_replays,
            fallback_action='cp',
        )
        refreshed = build_online_policy(
            rows,
            template_buckets=buckets,
            min_admit_tokens=args.fixed_metadata_arena_min_tokens,
            max_admit_tokens=args.fixed_metadata_arena_max_tokens,
            graph_action='ours_cp',
            default_action='cp',
            controller=controller,
        )
        refreshed['source_e2e'] = str(out)
        refreshed['source_baseline_contains'] = args.online_admission_baseline_contains
        refreshed['source_candidate_contains'] = args.online_admission_candidate_contains
        refreshed['single_engine_base_capture_size'] = int(args.runtime_base_capture_size)
        refreshed_path = Path(args.online_admission_refresh_output)
        refreshed_path.parent.mkdir(parents=True, exist_ok=True)
        refreshed_path.write_text(
            json.dumps({'runtime_policy': refreshed}, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        print(f'Saved online admission refreshed policy to {refreshed_path}')


if __name__ == '__main__':
    main()
