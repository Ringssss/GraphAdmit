# GraphAdmit: Online CUDA Graph Admission for Dynamic LLM Serving

GraphAdmit is an experimental serving-system prototype for making CUDA Graphs
useful under dynamic LLM workloads.  It does not try to force every dynamic
request into a CUDA Graph.  Instead, it treats graph replay as an online
admission problem:

> Dynamic workloads always enter the engine.  A graph template is used only
> after the engine has evidence that the replay path is token-correct and
> faster than fallback; otherwise the request falls back to the normal serving
> path.

The repository contains:

- a staticity control plane for classifying and canonicalizing dynamicity;
- runtime admission, residual capture, fixed metadata arenas, and partial graph
  helpers under `prefill_graph/runtime`;
- vLLM integration patches for runtime graph dispatch, live admission,
  template-aware scheduling, and MoE metadata profiling;
- benchmark harnesses for vLLM, SGLang, dInfer/LLaDA-style generation, and
  compositional CUDA Graph microbenchmarks;
- curated result summaries for Qwen3, Qwen3.5, and SGLang comparisons.

This is a research artifact.  Paths, model names, and GPU counts in the
commands below match the local experimental setup used for the reported
numbers and should be adjusted for a new machine.

## Repository Layout

```text
prefill_graph/runtime/          Core runtime policy, admission, arenas, scheduler helpers
prefill_graph/planner/          Bucket planner and cost model
benchmarks/                     E2E harnesses, policy builders, analyzers, microbenches
scripts/                        vLLM patch materialization utilities
patches/vllm_staticity.patch    vLLM/FlowPrefill patch for runtime staticity recovery
figures/                        Figure 1 motivation plot
tests/                          Unit tests for the control plane
results/*.md                    Curated result reports
```

Large raw profiles (`results/*.jsonl`), model weights, Python caches, and the
nested FlowPrefill checkout are intentionally excluded from git.  The vLLM
changes are distributed as `patches/vllm_staticity.patch` instead of vendoring
a patched vLLM tree.

## Core Ideas

GraphAdmit organizes dynamic CUDA Graph serving into three stages.

1. **Classify dynamicity.**

   The profiler and control plane classify shape, metadata address, function
   branch, MoE routing, and workload drift into semantic dynamicity,
   representational dynamicity, address dynamicity, or control dynamicity.

2. **Canonicalize recoverable dynamicity.**

   Recoverable dynamicity is mapped to graph-stable templates through fixed
   metadata arenas, token-axis buckets, key collapse, MoE metadata templates,
   and residual capture.

3. **Execute with admission and guards.**

   The vLLM dispatcher uses runtime policy rules plus live observations to
   admit, reject, blacklist, and evict templates.  Online admission uses
   token-correctness and latency evidence, not graph hit rate alone.

The target metric is **useful coverage**: requests served by graph replay that
are both correct and faster than fallback.

## Environment

The main experiments were run on NVIDIA H100 GPUs with CUDA 12.x.  The exact
software environment used locally was:

- Python 3.10 for the vLLM/GraphAdmit harness (`conda` env `crossstage`);
- vLLM 0.19.1 with the patch in `patches/vllm_staticity.patch`;
- Python 3.11 for SGLang (`conda` env `sglang-bench`);
- SGLang 0.5.10.post1;
- PyTorch 2.9.1+cu128 in the SGLang environment.

Example setup using the Tsinghua PyPI mirror:

```bash
conda create -n crossstage python=3.10 -y
conda activate crossstage
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U pip pytest numpy torch
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple vllm

conda create -n sglang-bench python=3.11 -y
conda activate sglang-bench
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sglang[all]
```

The commands above are a starting point.  For large-model experiments, install
the CUDA/PyTorch/vLLM/SGLang versions that match your driver and GPU cluster.

## Applying the vLLM Patch

The patch was generated against the local FlowPrefill/vLLM tree used in the
experiments.  The local base checkout was `HSIEHCHIACHI/FlowPrefill` at commit
`5d32324` before the uncommitted GraphAdmit changes.  Apply it from the root of
a compatible vLLM/FlowPrefill checkout:

```bash
git clone https://github.com/HSIEHCHIACHI/FlowPrefill.git external/FlowPrefill
cd external/FlowPrefill
git apply ../../patches/vllm_staticity.patch
pip install -e .
```

If you already have the patched local tree, `git apply --check` will fail
because the changes are present.  In that case, `git apply --reverse --check
../../patches/vllm_staticity.patch` should succeed.

The patch touches:

- `vllm/v1/cudagraph_dispatcher.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py`

Useful environment variables exposed by the patch:

```text
STATICITY_VLLM_RUNTIME_POLICY             Runtime policy JSON
STATICITY_VLLM_RUNTIME_ACTIVE             Enable/disable GraphAdmit runtime policy
STATICITY_VLLM_BASE_CAPTURE_SIZE          vLLM default graph ceiling, usually 512
STATICITY_VLLM_LIVE_ADMISSION             Enable live admission
STATICITY_VLLM_LIVE_OBSERVATIONS          JSONL observation stream
STATICITY_VLLM_LIVE_EXPLORE               Explore templates until min_samples
STATICITY_VLLM_LIVE_MIN_SAMPLES           Per-template support threshold
STATICITY_VLLM_LIVE_MIN_USEFUL_RATE       Useful-rate threshold
STATICITY_VLLM_LIVE_MIN_SAVING_MS         Latency saving threshold
STATICITY_VLLM_LIVE_MAX_P95_REGRESSION_MS Tail regression guard
STATICITY_VLLM_FIXED_METADATA_ARENA       Enable request metadata arena
STATICITY_VLLM_TEMPLATE_SCHEDULER         Enable template-aware scheduling
STATICITY_VLLM_MOE_PROFILE                MoE metadata profile output
```

## Quick Checks

Run unit tests:

```bash
conda activate crossstage
pytest -q tests/test_staticity_control_plane.py
```

Run the compositional CUDA Graph microbenchmark:

```bash
conda activate crossstage
CUDA_VISIBLE_DEVICES=0 python benchmarks/compositional_cuda_graph_microbench.py \
  --output results/compositional_cuda_graph_microbench.json \
  --repeat 30 --hidden 2048 --inter 8192 --layers 4 --dtype bfloat16
```

This benchmark isolates raw `torch.cuda.CUDAGraph` mechanisms.  A representative
run showed:

| Plan | Total ms | Speedup vs eager | Graph replays | Fallbacks | Padding waste | Correct |
|---|---:|---:|---:|---:|---:|---|
| eager_dynamic | 9.297 | 1.00x | 0 | 20 | 0.0% | yes |
| exact_only_graph_32_64 | 8.429 | 1.10x | 2 | 18 | 0.0% | yes |
| pad_to_next_template | 4.944 | 1.88x | 20 | 0 | 23.1% | yes |
| tile_32_composition | 9.537 | 0.97x | 39 | 0 | 23.1% | yes |
| pack_adjacent_requests | 2.608 | 3.56x | 10 | 0 | 0.0% | yes |
| residual_learn_then_replay | 5.166 | 1.80x | 19 | 0 | 47.0% | yes |

The weak baseline is exact-shape graph replay: it only hits shapes that were
captured exactly.  Padding, packing, and residual capture show why a dynamic
serving system needs more than a fixed list of graph sizes.

## Qwen3.5-35B Hybrid/MoE Experiment

This is the newest stress test in the repository.  The model used locally was:

```text
/mnt/models/Qwen3.5-35B-A3B
```

The model is MoE plus hybrid linear/full attention.  Its config includes 40
layers, 256 experts, 8 experts per token, and a repeating hybrid attention
layout.

Generate a generic live exploration policy.  This policy is not learned from
the Qwen3.5 trace:

```bash
conda activate crossstage
python benchmarks/plan_residual_capture_policy.py \
  --make-exploration-policy \
  --bucket-preset sglang-pcg \
  --exploration-max-tokens 4096 \
  --exploration-live-admission \
  --exploration-live-min-samples 2 \
  --exploration-live-min-useful-rate 0.67 \
  --exploration-live-min-saving-ms 0.5 \
  --exploration-live-max-p95-regression-ms 5.0 \
  --output results/runtime_policy_vllm_qwen35_35b_generic_pcg_live.json
```

Run vLLM baselines, blind extended CUDA Graph, and GraphAdmit live runtime:

```bash
conda activate crossstage
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmarks/vllm_flowprefill_workload.py \
  --workload results/qwentrace_morspec_qwen_16_4096.json \
  --model /mnt/models/Qwen3.5-35B-A3B \
  --tp-size 4 --max-model-len 4096 --gpu-memory-utilization 0.72 \
  --max-tokens 1 --planner-mode hybrid --our-max 4096 \
  --configs cp,ours_cp,runtime_cp \
  --runtime-policy results/runtime_policy_vllm_qwen35_35b_generic_pcg_live.json \
  --live-admission --live-admission-explore \
  --live-admission-shadow-baseline \
  --live-admission-observations results/live_obs_qwen35_35b_16.jsonl \
  --live-admission-min-samples 2 \
  --live-admission-min-useful-rate 0.67 \
  --live-admission-min-saving-ms 0.5 \
  --live-admission-max-p95-regression-ms 5.0 \
  --output results/vllm_qwen35_35b_16_live_compare.json \
  --profile-prefix results/vllm_qwen35_35b_16_live_compare \
  --max-num-seqs 8
```

Run SGLang baselines:

```bash
conda activate sglang-bench
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmarks/sglang_flowprefill_workload.py \
  --workload results/qwentrace_morspec_qwen_16_4096.json \
  --model /mnt/models/Qwen3.5-35B-A3B \
  --tp-size 4 --context-length 4096 --mem-fraction-static 0.72 \
  --max-new-tokens 1 --configs cg --warmup-count 3 \
  --output results/sglang_qwen35_35b_16_default_cg.json

CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmarks/sglang_flowprefill_workload.py \
  --workload results/qwentrace_morspec_qwen_16_4096.json \
  --model /mnt/models/Qwen3.5-35B-A3B \
  --tp-size 4 --context-length 4096 --mem-fraction-static 0.72 \
  --max-new-tokens 1 --configs piecewise \
  --piecewise-cuda-graph-max-tokens 4096 \
  --warmup-count 3 \
  --output results/sglang_qwen35_35b_16_piecewise.json
```

Representative results:

| System | Avg ms | P50 ms | P95 ms | P99 ms | Init s |
|---|---:|---:|---:|---:|---:|
| vLLM eager/no-CG | 314.66 | 137.22 | 904.00 | 991.73 | 74.0 |
| vLLM max512 CP | 254.59 | 115.63 | 759.81 | 811.99 | 82.7 |
| Blind extended CG | 64.25 | 39.82 | 143.41 | 351.57 | 82.0 |
| GraphAdmit live runtime | 42.82 | 40.25 | 56.68 | 60.02 | 82.8 |
| SGLang default CG | 116.47 | 114.61 | 132.17 | 135.07 | 50.6 |
| SGLang piecewise CG | 286.58 | 249.53 | 702.29 | 790.08 | 57.8 |

GraphAdmit live runtime is 5.95x faster than vLLM max512 CP on average and
2.72x faster than SGLang default CUDA Graph on this workload.

Online admission evidence:

- live observation file: `results/live_obs_qwen35_35b_16.jsonl`;
- 10 graph-template observations, 10/10 token-correct against the shadow CP
  fallback;
- observed templates: `tokens=768`, `tokens=832`, `tokens=896`,
  `tokens=3072`;
- dispatcher profile contains both `explore_until_min_samples` and
  `live_admitted`, so this is not just replaying an offline learned policy.

Correctness caveat:

- the extra-template GraphAdmit path is token-correct for the observed admitted
  templates;
- the whole run is not token-identical across all vLLM modes because this
  Qwen3.5 hybrid model shows one-token differences on small/default-path
  requests between eager, CP, and extended graph modes.  Do not claim whole-run
  token identity for this model without isolating that backend-level issue.

SGLang caveat:

- SGLang logs `Disabling overlap schedule since mamba no_buffer is not
  compatible with overlap schedule`, which likely explains the weaker SGLang
  result on this hybrid model.

Full report:

```text
results/qwen35_online_sglang_vllm_comparison_2026_04_29.md
```

## Qwen3 and SGLang Comparison

The previous Qwen3 runs compare GraphAdmit against vLLM max512 CP and SGLang
piecewise CUDA Graph:

| Model / trace | System | Avg ms | P95 ms | P99 ms | Notes |
|---|---:|---:|---:|---:|---|
| Qwen3-235B, n=16 | vLLM max512 CP | 141.29 | 239.79 | 246.08 | baseline |
| Qwen3-235B, n=16 | SGLang piecewise CG | 58.37 | 127.29 | 132.81 | broad PCG family |
| Qwen3-235B, n=16 | GraphAdmit PCG-tail | 55.48 | 125.22 | 131.12 | 4 extra templates: 768, 832, 896, 3072 |
| Qwen3-32B, n=64 | vLLM max512 CP | 43.37 | 113.69 | 126.89 | baseline |
| Qwen3-32B, n=64 | SGLang piecewise CG | 43.29 | 111.27 | 123.17 | broad PCG family |
| Qwen3-32B, n=64 | GraphAdmit PCG-tail | 38.99 | 115.85 | 127.24 | 6 extra templates |

The strongest result is Qwen3-235B: GraphAdmit matches or slightly beats SGLang
PCG while using fewer admitted extra templates and fail-closed admission.  On
Qwen3-32B, GraphAdmit improves average latency but SGLang still has slightly
better tail latency.

See:

```text
results/graphadmit_sglang_pcg_comparison_2026_04_29.md
results/sglang_baseline_comparison_2026_04_29.md
```

## Policy and Admission Utilities

Build a residual/exploration policy:

```bash
python benchmarks/plan_residual_capture_policy.py \
  --make-exploration-policy \
  --bucket-preset sglang-pcg \
  --exploration-max-tokens 4096 \
  --output results/runtime_policy_exploration.json
```

Refresh a policy from measured baseline/candidate rows:

```bash
python benchmarks/online_admission_policy_refresh.py \
  --input results/vllm_qwen35_35b_16_live_compare.json \
  --output results/runtime_policy_refreshed.json
```

Summarize dispatcher/runner profiles:

```bash
python benchmarks/analyze_cuda_graph_failure_modes.py \
  --e2e results/vllm_qwen35_35b_16_live_compare.json \
  --output results/qwen35_failure_analysis.json
```

## Figures

Figure 1 motivation plot:

```bash
python benchmarks/plot_figure1_motivation.py
```

Outputs:

```text
figures/figure1_motivation.svg
figures/figure1_motivation.pdf
```

The figure shows why static CUDA Graph replay is not always useful under a
dynamic Qwen trace: some graph replays are slower than fallback even when graph
replay is technically possible.

## What Is Not Included

The GitHub repository intentionally excludes:

- local model weights under `/mnt/models`;
- large raw `results/*.jsonl` dispatcher/runner/MoE profiles;
- `external/FlowPrefill/` as a nested git checkout;
- generated cache directories and Python bytecode.

The curated result JSON/Markdown files needed to inspect the main claims are
kept in `results/`.

## Limitations

- The current live admission harness uses a request-stream observation loop:
  request *i* writes graph-vs-fallback evidence, and request *i+1* can use that
  evidence.  It is stronger than offline trace calibration, but it is not yet
  same-request in-engine double execution.
- vLLM fixes CUDA graph capture sizes at engine initialization.  New residual
  capture templates can be learned online, but they require a new engine start
  unless the engine itself grows dynamic graph-capture support.
- The Qwen3.5 hybrid model shows backend-level one-token differences on
  small/default paths across vLLM modes.  The admitted extra templates were
  token-correct, but whole-run token identity should not be claimed for that
  model until this is isolated.
- SGLang is a strong baseline.  The paper claim should not be "GraphAdmit beats
  SGLang everywhere"; the defensible claim is fail-closed dynamic admission
  with useful coverage, plus strong wins on workloads where static/piecewise CG
  policies are unprofitable or disabled.

## Citation

This repository currently has no public paper citation.  If you use it in a
paper or artifact evaluation, cite the eventual GraphAdmit paper/repository
release.
