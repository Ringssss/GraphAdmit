# GraphAdmit E2E Review, 2026-05-01

This note records the current code review, the incremental fixes made in this
round, and the end-to-end results available from the local runs.  The main
metric below is average per-request E2E latency in milliseconds.  P95/P99 are
not used in this note.

## Code Changes

### 1. Dispatcher-grounded isolated live probe

The previous isolated-probe crash handling inferred the bad template from
prompt lengths.  That was too imprecise: vLLM tokenization, padding, or adjacent
bucket selection can make the actual replayed template differ from the prompt
length bucket.

The benchmark driver now:

- writes a probe-round dispatcher profile via `STATICITY_VLLM_CG_PROFILE`;
- reads the actual `staticity_runtime_admission.template_id` from the profile;
- blacklists only the last graph dispatch before a crash by default;
- retries the isolated probe for a bounded number of rounds;
- records `live_isolated_probe_rounds`, `live_isolated_probe_crash_blacklists`,
  and `live_isolated_probe_last_round_new_blacklists`.

This makes probe failures fail-closed against the template that actually
entered graph dispatch, rather than against a guessed length bucket.

### 2. Strict GraphAdmit-only dispatch mode

`--graphadmit-only` and `STATICITY_VLLM_GRAPHADMIT_ONLY=1` now mean:

- policy-level `default` and `cp` graph actions are rejected with
  `graphadmit_only_non_admitted_default`;
- native vLLM `FULL` and `PIECEWISE` key hits are also blocked unless the
  runtime policy selected a GraphAdmit-owned `ours` or `ours_cp` template.

The second point is important.  Without it, native vLLM graph keys can bypass
GraphAdmit admission after the policy function has accepted or rejected a
template.  The validator now constructs a fake dispatcher and checks:

- `cp:default:128` becomes `NONE` under GraphAdmit-only;
- an admitted `ours_cp` template still dispatches to `PIECEWISE`.

### 3. Patch distribution sync

The installed vLLM dispatcher, `external/FlowPrefill`, `patches/vllm_staticity.patch`,
and `graphadmit/resources/vllm_staticity.patch` now contain the same strict
GraphAdmit-only behavior.

## E2E Results

| Model / workload | vLLM best CG avg ms | GraphAdmit avg ms | SGLang local best avg ms | Speedup vs vLLM | Speedup vs SGLang | Correctness evidence |
|---|---:|---:|---:|---:|---:|---|
| Qwen3-32B, bs64 trace | 43.37 | 39.46 | 43.29 | 1.10x | 1.10x | full output match, 31/31 same-engine obs correct |
| Qwen3-32B, bs128 trace | 41.47 | 39.43 | n/a | 1.05x | n/a | full output match, 62/62 same-engine obs correct |
| Qwen3-235B-A22B, bs16 trace | 150.95 | 56.11 | 58.37 | 2.69x | 1.04x | full output match, 10/10 same-engine obs correct |
| Qwen3.5-35B-A3B, bs16 trace | 115.32 | 86.00 | 116.47 default CG | 1.34x | 1.35x | GraphAdmit probe: 25/28 correct, 3 crash blacklists; whole-file reference mismatch remains |

SGLang values are reused from the strongest local baseline JSON files already
present in `results/`:

- `results/sglang_qwen3_32b_64_piecewise_e2e_0510post1_warm3.json`
- `results/sglang_qwen3_235b_16_piecewise_e2e_0510post1_warm3.json`
- `results/sglang_qwen35_35b_16_default_cg.json`
- `results/sglang_qwen35_35b_16_piecewise.json`

## Qwen3.5 Status

Qwen3.5 is the most useful stress case because it combines MoE and hybrid
attention.  The fast run shows a real average latency win, but it is not yet a
paper-grade correctness claim for the entire engine output:

- `results/vllm_qwen35_probe_iter_16_20260501.json`
  reports GraphAdmit at 86.00 ms vs vLLM at 115.32 ms.
- The isolated probe produced 28 observations: 25 correct, 18 useful, and 3
  precise crash blacklists.
- The final measured run still has a reference mismatch on the whole output
  list.  Earlier dispatcher inspection indicated the mismatch was on a native
  vLLM CP/default path, not on a GraphAdmit extra-template observation.

This round therefore added strict GraphAdmit-only blocking for native key hits.
The dispatch-level validator passes.  A new Qwen3.5 strict E2E rerun could not
complete because another user's 8-GPU job occupied about 35 GiB per H100, leaving
only about 42.94 GiB free while the baseline requested about 57 GiB at
`--gpu-memory-utilization 0.72`.  That failure is an environment conflict, not a
GraphAdmit runtime error.

Until Qwen3.5 is rerun on idle GPUs with the strict native-key-hit block, the
clean paper-grade claims are Qwen3-32B and Qwen3-235B.  Qwen3.5 should be
reported as promising but still requiring strict-block E2E confirmation.

## Validation Commands

Syntax and hotpath validation:

```bash
conda run -n crossstage python -m py_compile benchmarks/vllm_flowprefill_workload.py benchmarks/validate_vllm_live_admission_hotpath.py
conda run -n crossstage python -m py_compile /home/zhujianian/miniconda3/envs/crossstage/lib/python3.10/site-packages/vllm/v1/cudagraph_dispatcher.py external/FlowPrefill/vllm/v1/cudagraph_dispatcher.py
conda run -n crossstage python benchmarks/validate_vllm_live_admission_hotpath.py --output results/vllm_live_admission_hotpath_validation_20260501_strict_dispatch.json
conda run -n crossstage pytest -q tests/test_staticity_control_plane.py tests/test_graphadmit_cli.py
```

Qwen3-32B, bs64:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n crossstage python benchmarks/vllm_flowprefill_workload.py --workload results/qwentrace_morspec_qwen_64_4096.json --model /mnt/models/Qwen3-32B --tp-size 4 --max-model-len 4096 --gpu-memory-utilization 0.8 --max-tokens 1 --planner-mode hybrid --our-max 1024 --extra-capture-sizes 640,768,832,896,1024 --skip-eager --configs cp,runtime_cp --runtime-policy results/runtime_policy_vllm_qwen3_32b_64_useful_auto_1024.json --runtime-base-capture-size 512 --cudagraph-mode FULL_AND_PIECEWISE --fixed-metadata-arena --fixed-metadata-arena-max-reqs 8 --full-key-collapse --live-admission --live-admission-explore --live-admission-min-samples 2 --live-admission-min-useful-rate 0.67 --live-admission-min-saving-ms 0.5 --live-admission-max-p95-regression-ms 5.0 --live-capture --live-admission-observations results/live_obs_qwen3_32b_itercheck_bs64_20260501.jsonl --live-admission-clear-observations --live-admission-template-id range --live-admission-same-engine-validate --disable-prefix-caching --output results/vllm_qwen3_32b_itercheck_bs64_20260501.json --profile-prefix results/vllm_qwen3_32b_itercheck_bs64_20260501
```

Qwen3-32B, bs128:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n crossstage python benchmarks/vllm_flowprefill_workload.py --workload results/qwentrace_morspec_qwen_128_4096.json --model /mnt/models/Qwen3-32B --tp-size 4 --max-model-len 4096 --gpu-memory-utilization 0.8 --max-tokens 1 --planner-mode hybrid --our-max 1024 --extra-capture-sizes 640,768,832,896,1024 --skip-eager --configs cp,runtime_cp --runtime-policy results/runtime_policy_vllm_qwen3_32b_64_useful_auto_1024.json --runtime-base-capture-size 512 --cudagraph-mode FULL_AND_PIECEWISE --fixed-metadata-arena --fixed-metadata-arena-max-reqs 8 --full-key-collapse --live-admission --live-admission-explore --live-admission-min-samples 2 --live-admission-min-useful-rate 0.67 --live-admission-min-saving-ms 0.5 --live-admission-max-p95-regression-ms 5.0 --live-capture --live-admission-observations results/live_obs_qwen3_32b_itercheck_bs128_20260501.jsonl --live-admission-clear-observations --live-admission-template-id range --live-admission-same-engine-validate --disable-prefix-caching --output results/vllm_qwen3_32b_itercheck_bs128_20260501.json --profile-prefix results/vllm_qwen3_32b_itercheck_bs128_20260501
```

Qwen3-235B:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 conda run -n crossstage python benchmarks/vllm_flowprefill_workload.py --workload results/qwentrace_morspec_qwen_16_4096.json --model /mnt/models/Qwen3-235B-A22B-Instruct-2507 --tp-size 8 --max-model-len 4096 --gpu-memory-utilization 0.8 --max-tokens 1 --planner-mode hybrid --our-max 4096 --extra-capture-sizes 768,832,896,3072 --skip-eager --configs cp,runtime_cp --runtime-policy results/runtime_policy_vllm_qwen3_235b_16_pcg_tail_learned.json --runtime-base-capture-size 512 --cudagraph-mode FULL_AND_PIECEWISE --fixed-metadata-arena --fixed-metadata-arena-max-reqs 8 --full-key-collapse --enable-return-routed-experts --live-admission --live-admission-explore --live-admission-min-samples 2 --live-admission-min-useful-rate 0.67 --live-admission-min-saving-ms 0.5 --live-admission-max-p95-regression-ms 5.0 --live-capture --live-admission-observations results/live_obs_qwen3_235b_itercheck_16_20260501.jsonl --live-admission-clear-observations --live-admission-template-id range --live-admission-same-engine-validate --disable-prefix-caching --output results/vllm_qwen3_235b_itercheck_16_20260501.json --profile-prefix results/vllm_qwen3_235b_itercheck_16_20260501
```

Qwen3.5 main run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n crossstage python benchmarks/vllm_flowprefill_workload.py --workload results/qwentrace_morspec_qwen_16_4096.json --model /mnt/models/Qwen3.5-35B-A3B --tp-size 4 --max-model-len 4096 --gpu-memory-utilization 0.72 --max-tokens 1 --planner-mode hybrid --our-max 4096 --skip-eager --configs cp,runtime_cp --runtime-policy results/runtime_policy_vllm_qwen35_35b_generic_pcg_live.json --runtime-base-capture-size 512 --cudagraph-mode FULL_AND_PIECEWISE --live-admission --live-admission-explore --live-admission-min-samples 2 --live-admission-min-useful-rate 0.67 --live-admission-min-saving-ms 0.5 --live-admission-max-p95-regression-ms 5.0 --live-capture --live-admission-observations results/live_obs_qwen35_probe_iter_16_20260501.jsonl --live-admission-clear-observations --live-admission-template-id range --live-admission-isolated-probe --live-admission-isolated-probe-timeout-s 300 --live-admission-isolated-probe-max-rounds 6 --disable-prefix-caching --output results/vllm_qwen35_probe_iter_16_20260501.json --profile-prefix results/vllm_qwen35_probe_iter_16_20260501 --max-num-seqs 8
```

## Bottom Line

The repo is stronger after this round in one specific way: GraphAdmit-only now
means graph replay is actually controlled by GraphAdmit admission, not merely by
the runtime policy function before vLLM's native key lookup.  This improves the
safety story even when it reduces coverage.

The current paper-grade performance claims should focus on:

- Qwen3-235B MoE: 56.11 ms vs vLLM 150.95 ms and SGLang 58.37 ms, with full
  output match.
- Qwen3-32B dense: 39.46 ms vs vLLM 43.37 ms and SGLang 43.29 ms at bs64, with
  full output match.

Qwen3.5 remains the next required validation target once the GPUs are idle:
rerun strict GraphAdmit-only with the native-key-hit block, collect fresh live
observations, and only then decide whether the hybrid-attention claim is
paper-grade.
