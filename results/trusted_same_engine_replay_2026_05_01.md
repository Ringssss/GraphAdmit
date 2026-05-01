# Trusted Same-Engine Replay Update

Date: 2026-05-01

This report records the stricter GraphAdmit runs that validate candidate graph
templates against fallback in the same engine when possible.  For models where
unsafe graph replay can crash the CUDA context, the harness runs the probe in
an isolated worker and converts crashes into trusted negative observations
before the measured serving engine starts.

The key change from earlier shadow-baseline results is that positive admission
is no longer granted from offline/shadow rows alone.  A template must have
trusted live replay evidence, and a crashing or token-wrong template is
blacklisted fail-closed.

## Summary

| Model / workload | Path | Avg ms | P50 ms | P95 ms | P99 ms | Speedup vs vLLM CP | Output identity | Trusted observations |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Qwen3-32B, bs64 | vLLM max512 CP | 43.66 | 44.22 | 114.24 | 127.32 | 1.00x | yes | 0 |
| Qwen3-32B, bs64 | GraphAdmit strict same-engine | 40.29 | 37.96 | 112.94 | 126.92 | 1.08x | yes | 31/31 correct, 19 useful |
| Qwen3-32B, bs128 | vLLM max512 CP | 41.13 | 37.36 | 111.46 | 125.08 | 1.00x | yes | 0 |
| Qwen3-32B, bs128 | GraphAdmit strict same-engine | 38.79 | 36.79 | 111.87 | 124.83 | 1.06x | yes | 62/62 correct, 23 useful |
| Qwen3-235B, bs16 | vLLM max512 CP | 150.64 | 211.55 | 236.07 | 243.44 | 1.00x | yes | 0 |
| Qwen3-235B, bs16 | GraphAdmit strict same-engine | 56.04 | 51.49 | 126.56 | 135.21 | 2.69x | yes | 10/10 correct, 10 useful |
| Qwen3.5-35B, bs16 | vLLM max512 CP | 119.90 | 117.90 | 144.74 | 155.20 | 1.00x | yes | 0 |
| Qwen3.5-35B, bs16 | GraphAdmit isolated-probe fail-closed | 86.69 | 104.11 | 129.96 | 136.49 | 1.38x | one default-path diff | 4 crash blacklists |

SGLang comparison points from the same local setup:

| Model / workload | SGLang path | Avg ms | P50 ms | P95 ms | P99 ms |
|---|---|---:|---:|---:|---:|
| Qwen3-235B, bs16 | piecewise CUDA Graph | 58.37 | 55.51 | 127.29 | 132.81 |
| Qwen3.5-35B, bs16 | default CUDA Graph | 116.47 | 114.61 | 132.17 | 135.07 |
| Qwen3.5-35B, bs16 | piecewise CUDA Graph | 286.58 | 249.53 | 702.29 | 790.08 |

Interpretation:

- Qwen3-235B is back at the README-level performance envelope under stricter
  same-engine validation: average and P95 slightly beat the local SGLang
  piecewise run, while P99 is slightly worse.
- Qwen3-32B gains are smaller but cleaner: every validated candidate is
  token-correct and the full measured outputs match the vLLM CP reference.
- Qwen3.5 is the important safety result.  Unsafe same-engine exploratory
  replay crashed on hybrid/MoE extra templates, so the isolated probe wrote
  trusted negative observations and the measured engine refused those
  templates.  The old 42.82 ms run should be treated as an optimistic
  shadow-baseline result, not the primary correctness claim.

## Trusted Observation Details

Qwen3-32B bs64:

| Template | Observations | Correct | Useful | Graph mean ms | Fallback mean ms |
|---|---:|---:|---:|---:|---:|
| `ours_cp:734:776:template=832:reqs=1` | 16 | 16 | 11 | 41.17 | 42.67 |
| `ours_cp:782:1012:template=1024:reqs=1` | 15 | 15 | 8 | 42.99 | 43.38 |

Qwen3-32B bs128:

| Template | Observations | Correct | Useful | Graph mean ms | Fallback mean ms |
|---|---:|---:|---:|---:|---:|
| `ours_cp:734:776:template=832:reqs=1` | 32 | 32 | 13 | 38.60 | 38.91 |
| `ours_cp:782:1012:template=1024:reqs=1` | 30 | 30 | 10 | 41.83 | 41.89 |

Qwen3-235B bs16:

| Template | Observations | Correct | Useful | Graph mean ms | Fallback mean ms |
|---|---:|---:|---:|---:|---:|
| `ours_cp:744:756:template=768:reqs=1` | 4 | 4 | 4 | 53.48 | 213.86 |
| `ours_cp:790:805:template=832:reqs=1` | 2 | 2 | 2 | 57.97 | 205.77 |
| `ours_cp:858:886:template=896:reqs=1` | 2 | 2 | 2 | 58.79 | 213.57 |
| `ours_cp:2828:3023:template=3072:reqs=1` | 2 | 2 | 2 | 134.28 | 241.23 |

Qwen3.5 bs16 isolated probe:

| Template | Observation | Result |
|---|---:|---|
| `ours_cp:704:768:template=768:reqs=1` | 1 | crash blacklist |
| `ours_cp:768:832:template=832:reqs=1` | 1 | crash blacklist |
| `ours_cp:832:896:template=896:reqs=1` | 1 | crash blacklist |
| `ours_cp:2816:3072:template=3072:reqs=1` | 1 | crash blacklist |

The Qwen3.5 measured run still improves average latency because it uses the
safe base graph family and refuses the unsafe extra graph family.  The single
output mismatch is on request index 7 in a small/default-path request, not an
admitted extra-template replay; it remains a backend nondeterminism caveat for
that model.

## Commands

The following commands are the commands used, or reconstructed directly from
the result metadata where shell history did not retain the full line.

### Validation

```bash
source /home/zhujianian/miniconda3/etc/profile.d/conda.sh
conda activate crossstage

python -m py_compile \
  benchmarks/vllm_flowprefill_workload.py \
  benchmarks/validate_vllm_live_admission_hotpath.py \
  scripts/materialize_vllm_staticity_patch.py \
  graphadmit/cli.py graphadmit/vllm.py graphadmit/policy.py

pytest -q tests/test_staticity_control_plane.py tests/test_graphadmit_cli.py

python benchmarks/validate_staticity_runtime_components.py \
  --output results/staticity_runtime_components_validation_20260501_isolated_probe.json

python benchmarks/validate_vllm_live_admission_hotpath.py \
  --output results/vllm_live_admission_hotpath_validation_20260501_final_probe.json
```

### Qwen3-32B bs64

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmarks/vllm_flowprefill_workload.py \
  --workload results/qwentrace_morspec_qwen_64_4096.json \
  --model /mnt/models/Qwen3-32B \
  --tp-size 4 --max-model-len 4096 --gpu-memory-utilization 0.8 \
  --max-tokens 1 --planner-mode hybrid --our-max 1024 \
  --extra-capture-sizes 640,768,832,896,1024 \
  --skip-eager --configs cp,runtime_cp \
  --runtime-policy results/runtime_policy_vllm_qwen3_32b_64_useful_auto_1024.json \
  --runtime-base-capture-size 512 \
  --cudagraph-mode FULL_AND_PIECEWISE \
  --fixed-metadata-arena \
  --fixed-metadata-arena-max-reqs 8 \
  --full-key-collapse \
  --live-admission \
  --live-admission-explore \
  --live-admission-min-samples 2 \
  --live-admission-min-useful-rate 0.67 \
  --live-admission-min-saving-ms 0.5 \
  --live-admission-max-p95-regression-ms 5.0 \
  --live-capture \
  --live-admission-observations results/live_obs_qwen3_32b_sameengine_range_bs64_20260501.jsonl \
  --live-admission-clear-observations \
  --live-admission-template-id range \
  --live-admission-same-engine-validate \
  --disable-prefix-caching \
  --output results/vllm_qwen3_32b_sameengine_range_bs64_20260501.json \
  --profile-prefix results/vllm_qwen3_32b_sameengine_range_bs64_20260501
```

### Qwen3-32B bs128

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmarks/vllm_flowprefill_workload.py \
  --workload results/qwentrace_morspec_qwen_128_4096.json \
  --model /mnt/models/Qwen3-32B \
  --tp-size 4 --max-model-len 4096 --gpu-memory-utilization 0.8 \
  --max-tokens 1 --planner-mode hybrid --our-max 1024 \
  --extra-capture-sizes 640,768,832,896,1024 \
  --skip-eager --configs cp,runtime_cp \
  --runtime-policy results/runtime_policy_vllm_qwen3_32b_64_useful_auto_1024.json \
  --runtime-base-capture-size 512 \
  --cudagraph-mode FULL_AND_PIECEWISE \
  --fixed-metadata-arena \
  --fixed-metadata-arena-max-reqs 8 \
  --full-key-collapse \
  --live-admission \
  --live-admission-explore \
  --live-admission-min-samples 2 \
  --live-admission-min-useful-rate 0.67 \
  --live-admission-min-saving-ms 0.5 \
  --live-admission-max-p95-regression-ms 5.0 \
  --live-capture \
  --live-admission-observations results/live_obs_qwen3_32b_sameengine_range_bs128_20260501.jsonl \
  --live-admission-clear-observations \
  --live-admission-template-id range \
  --live-admission-same-engine-validate \
  --disable-prefix-caching \
  --output results/vllm_qwen3_32b_sameengine_range_bs128_20260501.json \
  --profile-prefix results/vllm_qwen3_32b_sameengine_range_bs128_20260501
```

### Qwen3-235B bs16

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python benchmarks/vllm_flowprefill_workload.py \
  --workload results/qwentrace_morspec_qwen_16_4096.json \
  --model /mnt/models/Qwen3-235B-A22B-Instruct-2507 \
  --tp-size 8 --max-model-len 4096 --gpu-memory-utilization 0.8 \
  --max-tokens 1 --planner-mode hybrid --our-max 4096 \
  --extra-capture-sizes 768,832,896,3072 \
  --skip-eager --configs cp,runtime_cp \
  --runtime-policy results/runtime_policy_vllm_qwen3_235b_16_pcg_tail_learned.json \
  --runtime-base-capture-size 512 \
  --cudagraph-mode FULL_AND_PIECEWISE \
  --fixed-metadata-arena \
  --fixed-metadata-arena-max-reqs 8 \
  --full-key-collapse \
  --enable-return-routed-experts \
  --live-admission \
  --live-admission-explore \
  --live-admission-min-samples 2 \
  --live-admission-min-useful-rate 0.67 \
  --live-admission-min-saving-ms 0.5 \
  --live-admission-max-p95-regression-ms 5.0 \
  --live-capture \
  --live-admission-observations results/live_obs_qwen3_235b_pcg_tail_sameengine_range_16_20260501.jsonl \
  --live-admission-clear-observations \
  --live-admission-template-id range \
  --live-admission-same-engine-validate \
  --disable-prefix-caching \
  --output results/vllm_qwen3_235b_pcg_tail_sameengine_range_16_20260501.json \
  --profile-prefix results/vllm_qwen3_235b_pcg_tail_sameengine_range_16_20260501
```

### Qwen3.5-35B bs16

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmarks/vllm_flowprefill_workload.py \
  --workload results/qwentrace_morspec_qwen_16_4096.json \
  --model /mnt/models/Qwen3.5-35B-A3B \
  --tp-size 4 --max-model-len 4096 --gpu-memory-utilization 0.72 \
  --max-tokens 1 --planner-mode hybrid --our-max 4096 \
  --skip-eager --configs cp,runtime_cp \
  --runtime-policy results/runtime_policy_vllm_qwen35_35b_generic_pcg_live.json \
  --runtime-base-capture-size 512 \
  --cudagraph-mode FULL_AND_PIECEWISE \
  --live-admission \
  --live-admission-explore \
  --live-admission-min-samples 2 \
  --live-admission-min-useful-rate 0.67 \
  --live-admission-min-saving-ms 0.5 \
  --live-admission-max-p95-regression-ms 5.0 \
  --live-capture \
  --live-admission-observations results/live_obs_qwen35_isolated_probe_16_20260501.jsonl \
  --live-admission-clear-observations \
  --live-admission-template-id range \
  --live-admission-isolated-probe \
  --live-admission-isolated-probe-timeout-s 300 \
  --disable-prefix-caching \
  --output results/vllm_qwen35_isolated_probe_16_20260501.json \
  --profile-prefix results/vllm_qwen35_isolated_probe_16_20260501 \
  --max-num-seqs 8
```
