# GraphAdmit Summary and Code/Experiment Review

Date: 2026-05-01

This document summarizes what GraphAdmit currently does, what the code actually implements, what the experiments show, and how to interpret the comparisons against vLLM, SGLang, and raw Torch CUDA Graphs. It is written as the concise artifact-level summary for reviewers and future users.

## 1. One-Sentence Summary

GraphAdmit is not "capture every dynamic request into CUDA Graph." It is a fail-closed CUDA Graph admission layer for dynamic LLM serving:

> Every dynamic workload is allowed into the serving engine, but only graph templates that are guarded, token-correct, and measured faster than fallback are admitted. Everything else falls back to the normal engine path.

This is the right claim. It is stronger and more credible than claiming arbitrary dynamicity can always be graph replayed.

## 2. What We Built

GraphAdmit implements three stages.

### 2.1 Classify Dynamicity

The system profiles where dynamicity comes from:

- token/request/block shape changes;
- metadata address changes;
- graph-key explosion from request layout and `num_reqs`;
- MoE routing and expert-count changes;
- hybrid attention / diffusion / branch-level dynamic functions;
- workload drift where a once-useful template becomes bad.

Relevant code:

- `prefill_graph/runtime/dynamicity.py`
- `prefill_graph/runtime/profiler.py`
- `prefill_graph/runtime/control_plane.py`
- `benchmarks/analyze_vllm_keycollapse_runtime.py`
- `benchmarks/analyze_cuda_graph_failure_modes.py`
- `benchmarks/summarize_vllm_attention_profile.py`
- `benchmarks/summarize_vllm_moe_profile.py`

### 2.2 Canonicalize Recoverable Dynamicity

Recoverable dynamicity is transformed into static graph-compatible representation:

- fixed-address request metadata arena;
- token-axis padded arena for `positions`, `slot_mapping`, and active masks;
- arena-gated graph-key collapse;
- residual token templates;
- MoE expert metadata template abstraction;
- function-level partial graph segment abstraction.

Relevant code:

- `prefill_graph/runtime/arena.py`
- `prefill_graph/runtime/residual_capture.py`
- `prefill_graph/runtime/partial_graph.py`
- `prefill_graph/runtime/scheduler.py`
- vLLM integration code shipped through `patches/vllm_staticity.patch`

### 2.3 Execute With Guards and Online Admission

Graph replay is admitted only when evidence says it is safe and useful:

- correctness failures blacklist a template;
- insufficient sample count keeps a template in exploration;
- low useful-rate rejects it;
- weak latency saving rejects it;
- tail-regression guard rejects it;
- unsupported semantic dynamicity falls back.

Relevant code:

- `prefill_graph/runtime/admission.py`
- `prefill_graph/runtime/planner.py`
- `benchmarks/plan_residual_capture_policy.py`
- `benchmarks/online_admission_policy_refresh.py`
- `graphadmit/policy.py`
- `graphadmit/cli.py`
- `graphadmit/vllm.py`

The public installable layer is now:

```bash
pip install git+https://github.com/Ringssss/GraphAdmit.git
graphadmit doctor
graphadmit make-policy --bucket-preset sglang-pcg --max-tokens 4096 --output graphadmit_policy.json
graphadmit vllm-env --policy graphadmit_policy.json --observations graphadmit_observations.jsonl
```

## 3. What the vLLM Patch Does

The vLLM/FlowPrefill patch is distributed as:

- `patches/vllm_staticity.patch`
- `graphadmit/resources/vllm_staticity.patch`
- manifest: `patches/vllm_staticity.patch.manifest.json`

It modifies:

| File | Role |
|---|---|
| `vllm/v1/cudagraph_dispatcher.py` | Runtime policy hooks, graph/fallback dispatch, live admission, replay metadata |
| `vllm/v1/core/sched/scheduler.py` | Template-aware scheduling hints |
| `vllm/v1/worker/gpu_model_runner.py` | Runtime admission, observation export, graph/fallback control, metadata arena hooks |
| `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` | Routed expert metadata capture support |

The GitHub repository does not vendor the local `external/FlowPrefill` checkout. The source-level vLLM edits are distributed through the patch file above.

Important environment variables:

```text
STATICITY_VLLM_RUNTIME_POLICY
STATICITY_VLLM_RUNTIME_ACTIVE
STATICITY_VLLM_LIVE_ADMISSION
STATICITY_VLLM_LIVE_EXPLORE
STATICITY_VLLM_LIVE_OBSERVATIONS
STATICITY_VLLM_LIVE_MIN_SAMPLES
STATICITY_VLLM_LIVE_MIN_USEFUL_RATE
STATICITY_VLLM_LIVE_MIN_SAVING_MS
STATICITY_VLLM_LIVE_MAX_P95_REGRESSION_MS
STATICITY_VLLM_FIXED_METADATA_ARENA
STATICITY_VLLM_TEMPLATE_SCHEDULER
STATICITY_VLLM_MOE_PROFILE
```

Patch check/apply:

```bash
graphadmit vllm-patch --target external/FlowPrefill
graphadmit vllm-patch --target external/FlowPrefill --apply
```

## 4. Experiment Results

### 4.1 Trusted Same-Engine / Isolated-Probe Results

The 2026-05-01 update tightened the correctness contract.  Positive admission
now requires trusted graph replay evidence from the same engine when possible.
For architectures where unsafe exploratory replay can crash the CUDA context,
the harness probes in a child process and converts a crash into trusted
negative observations before starting the measured serving engine.

| Model / workload | System | Avg ms | P50 ms | P95 ms | P99 ms | Speedup | Correctness evidence |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen3-32B, bs64 | vLLM max512 CP | 43.66 | 44.22 | 114.24 | 127.32 | 1.00x | whole-run identical |
| Qwen3-32B, bs64 | GraphAdmit strict same-engine | 40.29 | 37.96 | 112.94 | 126.92 | 1.08x | 31/31 trusted observations correct |
| Qwen3-32B, bs128 | vLLM max512 CP | 41.13 | 37.36 | 111.46 | 125.08 | 1.00x | whole-run identical |
| Qwen3-32B, bs128 | GraphAdmit strict same-engine | 38.79 | 36.79 | 111.87 | 124.83 | 1.06x | 62/62 trusted observations correct |
| Qwen3-235B, bs16 | vLLM max512 CP | 150.64 | 211.55 | 236.07 | 243.44 | 1.00x | whole-run identical |
| Qwen3-235B, bs16 | GraphAdmit strict same-engine | 56.04 | 51.49 | 126.56 | 135.21 | 2.69x | 10/10 trusted observations correct |
| Qwen3.5-35B, bs16 | vLLM max512 CP | 119.90 | 117.90 | 144.74 | 155.20 | 1.00x | whole-run identical |
| Qwen3.5-35B, bs16 | GraphAdmit isolated-probe fail-closed | 86.69 | 104.11 | 129.96 | 136.49 | 1.38x | unsafe extra templates blacklisted; one small/default-path diff |

SGLang comparison points:

| Model / workload | SGLang path | Avg ms | P50 ms | P95 ms | P99 ms |
|---|---|---:|---:|---:|---:|
| Qwen3-235B, bs16 | piecewise CUDA Graph | 58.37 | 55.51 | 127.29 | 132.81 |
| Qwen3.5-35B, bs16 | default CUDA Graph | 116.47 | 114.61 | 132.17 | 135.07 |
| Qwen3.5-35B, bs16 | piecewise CUDA Graph | 286.58 | 249.53 | 702.29 | 790.08 |

Interpretation:

- Qwen3-235B is the strongest strict result: GraphAdmit is 2.69x faster than
  vLLM CP and slightly faster than local SGLang piecewise on mean/P95, with
  slightly worse P99.
- Qwen3-32B gains are modest but clean and token-identical.
- Qwen3.5 cannot honestly keep the old 42.82 ms primary claim under strict
  replay validation.  Unsafe extra-template replay crashed, so the correct
  system behavior is to blacklist those templates and run fail-closed.  Even
  then, the measured path is 1.38x faster than vLLM CP and faster on average
  than local SGLang default CG.

Full report:

```text
results/trusted_same_engine_replay_2026_05_01.md
```

### 4.2 Legacy Qwen3.5-35B-A3B Hybrid Attention + MoE

Model:

```text
/mnt/models/Qwen3.5-35B-A3B
```

Workload:

```text
results/qwentrace_morspec_qwen_16_4096.json
```

This is the most important stress test because the model has MoE plus hybrid linear/full attention.

| System | Avg ms | P50 ms | P95 ms | P99 ms | Init s |
|---|---:|---:|---:|---:|---:|
| vLLM eager/no-CG | 314.66 | 137.22 | 904.00 | 991.73 | 74.0 |
| vLLM max512 CP | 254.59 | 115.63 | 759.81 | 811.99 | 82.7 |
| Blind extended CG | 64.25 | 39.82 | 143.41 | 351.57 | 82.0 |
| GraphAdmit legacy shadow-baseline runtime | 42.82 | 40.25 | 56.68 | 60.02 | 82.8 |
| SGLang default CG | 116.47 | 114.61 | 132.17 | 135.07 | 50.6 |
| SGLang piecewise CG | 286.58 | 249.53 | 702.29 | 790.08 | 57.8 |

Interpretation:

- This legacy run is 5.95x faster than vLLM max512 CP on average.
- It is 2.72x faster than SGLang default CG on average for this hybrid/MoE workload.
- It has much better tail than blind extended CG: P95 `56.68 ms` vs `143.41 ms`, P99 `60.02 ms` vs `351.57 ms`.
- The policy used here is a generic PCG-family exploration policy, not learned from Qwen3.5 trace data.

Online admission evidence:

- `results/live_obs_qwen35_35b_16.jsonl`
- 10 graph-template observations;
- 10/10 token-correct for observed extra-template GraphAdmit paths;
- observed templates: `768`, `832`, `896`, `3072`;
- dispatcher profile contains both `explore_until_min_samples` and `live_admitted`.

Correctness caveat:

- This is now a legacy optimistic result because it trusts shadow-baseline
  evidence rather than strict same-engine replay validation.
- The strict isolated-probe run found that Qwen3.5 extra templates can crash
  during unsafe replay.  The system now converts that into blacklist evidence
  and falls back.
- Therefore the primary Qwen3.5 claim should be fail-closed safety plus the
  stricter 86.69 ms result, not the old 42.82 ms number.

### 4.3 Qwen3-235B-A22B

Workload:

```text
results/qwentrace_morspec_qwen_16_4096.json
```

Best strict same-engine comparison:

| System | Avg ms | P50 ms | P95 ms | P99 ms | Init s | Correct |
|---|---:|---:|---:|---:|---:|---|
| vLLM max512 CP | 150.64 | 211.55 | 236.07 | 243.44 | 144.83 | true |
| SGLang piecewise CG | 58.37 | 55.51 | 127.29 | 132.81 | 201.36 | true |
| GraphAdmit strict same-engine | 56.04 | 51.49 | 126.56 | 135.21 | 138.12 | true |

Interpretation:

- GraphAdmit is 2.69x faster than vLLM max512 CP on average.
- GraphAdmit slightly beats SGLang PCG on this run by average and P95, while
  SGLang is slightly better at P99.
- The important point is not a huge latency margin over SGLang; the important point is that GraphAdmit reaches SGLang-level performance with fail-closed admission and fewer selected templates.

Earlier residual-capture result:

- One admitted `896`-token residual template improved average latency from `139.04 ms` to `73.03 ms`, a 1.90x average speedup over vLLM CP.
- Tail was near-neutral.
- This showed the mid-token MoE window had high value.

### 4.4 Qwen3-32B Dense

Workload:

```text
results/qwentrace_morspec_qwen_64_4096.json
```

Best strict same-engine comparison:

| System | Avg ms | P50 ms | P95 ms | P99 ms | Init s | Correct |
|---|---:|---:|---:|---:|---:|---|
| vLLM max512 CP, bs64 | 43.66 | 44.22 | 114.24 | 127.32 | 72.19 | true |
| SGLang piecewise CG | 43.29 | 40.74 | 111.27 | 123.17 | 91.38 | true |
| GraphAdmit strict same-engine, bs64 | 40.29 | 37.96 | 112.94 | 126.92 | 66.91 | true |
| vLLM max512 CP, bs128 | 41.13 | 37.36 | 111.46 | 125.08 | 71.86 | true |
| GraphAdmit strict same-engine, bs128 | 38.79 | 36.79 | 111.87 | 124.83 | 65.97 | true |

Interpretation:

- GraphAdmit is 1.08x faster than vLLM max512 CP on bs64 and 1.06x faster on bs128.
- Against the local SGLang PCG bs64 result, GraphAdmit is better on mean and P95 but still slightly worse on P99.
- So the honest claim is "clean useful-coverage gain, tail not universally dominant."

### 4.5 dInfer / LLaDA2.0-mini

Representative validated run:

| Path | Total s | Speedup | Correct |
|---|---:|---:|---|
| Eager dInfer | 143.37 | 1.00x | true |
| Runtime graph admission | 83.94 | 1.708x | true |
| Runtime graph admission + cleanup | 90.84 | 1.578x | true |

Interpretation:

- The earlier unsafe graph path could be fast but token-wrong.
- GraphAdmit-style admission turns it into a correct graph path by requiring decoded-token validation, template budget, periodic validation, and memory guard.
- This is strong evidence that "correctness-aware admission" is necessary for dynamic function workloads.

### 4.6 Torch CUDA Graph Microbenchmark

File:

```text
results/torch_naive_cudagraph_compositional_microbench_qwen35_context.json
```

| Plan | Total ms | Speedup vs eager | Graph replays | Fallbacks | Padding waste | Correct |
|---|---:|---:|---:|---:|---:|---|
| eager_dynamic | 9.297 | 1.00x | 0 | 20 | 0.0% | true |
| exact_only_graph_32_64 | 8.429 | 1.10x | 2 | 18 | 0.0% | true |
| pad_to_next_template | 4.944 | 1.88x | 20 | 0 | 23.1% | true |
| tile_32_composition | 9.537 | 0.97x | 39 | 0 | 23.1% | true |
| pack_adjacent_requests | 2.608 | 3.56x | 10 | 0 | 0.0% | true |
| residual_learn_then_replay | 5.166 | 1.80x | 19 | 0 | 47.0% | true |

Important interpretation:

- The strong `3.56x` result is not naive Torch CUDA Graph.
- The naive Torch baseline is `exact_only_graph_32_64`, which only gives 1.10x because it only replays exact captured shapes and most dynamic sizes fall back.
- `pack_adjacent_requests` is a compositional mechanism experiment. It packs independent token-local requests such as `55+41` into one `96`-token graph. That is why it can look stronger than SGLang in this microbenchmark.
- This microbenchmark is not an end-to-end LLM serving comparison. It has no tensor parallelism, no attention metadata, no KV cache, no MoE routing, no scheduler, no network/IPC overhead, and the token-local MLP makes packing semantically exact.
- Therefore the correct claim is: Torch raw CUDAGraph demonstrates mechanism potential; it does not prove Torch serving beats SGLang.

## 5. Why GraphAdmit Beats vLLM CUDA Graph

vLLM CUDA Graph is strong for static or limited-shape paths, but its default policy is mostly static:

- graph family and capture sizes are chosen at engine initialization;
- max512 CP misses or falls back on many long/dynamic prefill shapes;
- static graph replay does not know whether a replay is faster than fallback for the current workload;
- graph keys can still encode request layout and metadata shape/address dynamics;
- semantic dynamicity is not separated from representational/address dynamicity by default.

GraphAdmit improves this by:

- generating residual/exploration templates beyond the base vLLM capture ceiling;
- stabilizing metadata addresses inside admitted windows;
- collapsing graph keys only when fixed arenas make it safe;
- measuring graph vs fallback latency;
- rejecting templates with low useful rate, weak saving, token mismatch, or tail regression;
- falling back instead of forcing semantic dynamicity into graph replay.

This is why GraphAdmit beats vLLM strongly on Qwen3-235B, modestly on
Qwen3-32B, and still improves Qwen3.5 average latency after blacklisting unsafe
hybrid/MoE extra templates.

## 6. Why GraphAdmit Can Beat SGLang, and Where SGLang Is Still Strong

SGLang piecewise CUDA Graph is a strong baseline. It has:

- broad predefined token/batch graph families;
- padding to captured sizes;
- persistent CUDA Graph buffers;
- strong decode/piecewise engineering;
- low serving overhead in many standard workloads.

GraphAdmit can beat SGLang when:

- SGLang PCG is disabled or weakened by model architecture constraints;
- a hybrid attention / MoE model forces SGLang to disable overlap scheduling;
- a narrow high-value template window dominates latency;
- online admission avoids negative graph templates that broad PCG would still try;
- fewer admitted templates recover most of the useful coverage.

Evidence:

- Qwen3.5 hybrid/MoE strict fail-closed: GraphAdmit `86.69 ms`, SGLang default CG `116.47 ms`, SGLang piecewise CG `286.58 ms`.
- Qwen3.5 legacy optimistic shadow-baseline: `42.82 ms`; useful for diagnosing potential, not the primary correctness claim.
- Qwen3-235B strict same-engine: GraphAdmit `56.04 ms`, SGLang PCG `58.37 ms`.

But SGLang remains strong:

- On Qwen3-32B, GraphAdmit wins average but SGLang has better P95/P99.
- Earlier old-GraphAdmit runs lost to latest SGLang PCG on Qwen3-235B until tail-aware PCG residual templates were added.
- Therefore the paper should not claim "GraphAdmit dominates SGLang everywhere." The stronger claim is:

> GraphAdmit adds fail-closed correctness/latency admission and dynamicity classification. It can match or beat SGLang PCG on selected dynamic workloads while preserving a more conservative safety boundary, but SGLang remains a strong steady-state PCG baseline.

## 7. Code Review: What Is Solid and What Is Still a Research Prototype

### Solid / Implemented

| Area | Status |
|---|---|
| Installable CLI | Implemented in `graphadmit/`; `pip install git+...` works |
| Policy generator | Implemented in `graphadmit/policy.py` and `benchmarks/plan_residual_capture_policy.py` |
| vLLM patch helper | Implemented in `graphadmit/vllm.py` |
| Online admission core | Implemented in `prefill_graph/runtime/admission.py` |
| Residual capture planner | Implemented in `prefill_graph/runtime/residual_capture.py` |
| Fixed metadata arena abstraction | Implemented in `prefill_graph/runtime/arena.py` |
| Token-axis canonicalizer | Implemented as standalone runtime abstraction |
| MoE metadata abstraction | Implemented and E2E profiled, not yet dispatch-controlling |
| Partial graph manager | Implemented as standalone abstraction |
| Scheduler guard | Implemented as standalone abstraction and vLLM hooks, but weak E2E gain |
| vLLM E2E harness | Implemented in `benchmarks/vllm_flowprefill_workload.py` |
| SGLang harness | Implemented in `benchmarks/sglang_flowprefill_workload.py` |
| Torch microbench | Implemented in `benchmarks/compositional_cuda_graph_microbench.py` |
| Unit tests | `tests/test_staticity_control_plane.py`, `tests/test_graphadmit_cli.py` |

### Still Prototype / Do Not Overclaim

| Area | Current limitation |
|---|---|
| vLLM dynamic graph capture | vLLM capture sizes are still fixed at engine init; online-learned residual templates usually need a next engine start unless pre-captured exploration templates exist |
| Same-request shadow fallback | Same-engine graph/fallback validation exists for single-request probes, and isolated child-process probing catches crashes; full always-on production shadowing with low overhead is still prototype work |
| MoE dispatch templates | Routed-expert metadata is profiled and abstracted, but fused-MoE/all-to-all dispatch is not yet selected by GraphAdmit templates |
| Function-level partial graph | Runtime abstraction exists; real hybrid attention/diffusion branch integration is not complete |
| Scheduler | Guarded scheduler exists; positive tail-safe E2E benefit is not yet proven |
| Qwen3.5 whole-run correctness | Strict mode blacklists unsafe extra templates; one small/default-path output difference remains a backend nondeterminism caveat |
| SGLang comparison | Strong but workload-dependent; GraphAdmit should not claim universal dominance |

## 8. Main Commands Used

### 8.1 Package / CLI Checks

```bash
python -m graphadmit.cli --help
python -m graphadmit.cli doctor
python -m graphadmit.cli make-policy \
  --bucket-preset sglang-pcg \
  --max-tokens 1024 \
  --output /tmp/graphadmit_policy_smoke.json
python -m graphadmit.cli vllm-patch --target external/FlowPrefill
python -m pip install --target /tmp/graphadmit_install_smoke2 .
PYTHONPATH=/tmp/graphadmit_install_smoke2 \
  /tmp/graphadmit_install_smoke2/bin/graphadmit vllm-env \
  --policy /tmp/graphadmit_policy_smoke.json --json
```

### 8.2 Unit / Compile Checks

```bash
python -m py_compile $(git ls-files '*.py') graphadmit/*.py benchmarks/__init__.py tests/test_graphadmit_cli.py

source /home/zhujianian/miniconda3/etc/profile.d/conda.sh
conda activate crossstage
pytest -q tests/test_staticity_control_plane.py tests/test_graphadmit_cli.py
```

### 8.3 Qwen3.5 Policy Generation

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

Equivalent public CLI:

```bash
graphadmit make-policy \
  --bucket-preset sglang-pcg \
  --max-tokens 4096 \
  --live-min-samples 2 \
  --live-min-useful-rate 0.67 \
  --live-min-saving-ms 0.5 \
  --live-max-p95-regression-ms 5.0 \
  --output results/runtime_policy_vllm_qwen35_35b_generic_pcg_live.json
```

### 8.4 Qwen3.5 vLLM / GraphAdmit Strict Isolated Probe

```bash
conda activate crossstage
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmarks/vllm_flowprefill_workload.py \
  --workload results/qwentrace_morspec_qwen_16_4096.json \
  --model /mnt/models/Qwen3.5-35B-A3B \
  --tp-size 4 --max-model-len 4096 --gpu-memory-utilization 0.72 \
  --max-tokens 1 --planner-mode hybrid --our-max 4096 \
  --skip-eager --configs cp,runtime_cp \
  --runtime-policy results/runtime_policy_vllm_qwen35_35b_generic_pcg_live.json \
  --runtime-base-capture-size 512 \
  --cudagraph-mode FULL_AND_PIECEWISE \
  --live-admission --live-admission-explore \
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

### 8.5 Qwen3.5 Legacy vLLM / GraphAdmit Shadow-Baseline

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

### 8.6 Qwen3.5 vLLM Eager Reference

```bash
conda activate crossstage
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmarks/vllm_flowprefill_workload.py \
  --workload results/qwentrace_morspec_qwen_16_4096.json \
  --model /mnt/models/Qwen3.5-35B-A3B \
  --tp-size 4 --max-model-len 4096 --gpu-memory-utilization 0.72 \
  --max-tokens 1 --planner-mode hybrid --our-max 4096 \
  --configs eager \
  --output results/vllm_qwen35_35b_16_eager_ref.json \
  --profile-prefix results/vllm_qwen35_35b_16_eager_ref \
  --max-num-seqs 8
```

### 8.7 Qwen3.5 SGLang Baselines

```bash
source /home/zhujianian/miniconda3/etc/profile.d/conda.sh
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

### 8.8 Qwen3-235B GraphAdmit Strict Same-Engine

```bash
conda activate crossstage
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

### 8.9 Qwen3-32B GraphAdmit Strict Same-Engine bs64

```bash
conda activate crossstage
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmarks/vllm_flowprefill_workload.py \
  --workload results/qwentrace_morspec_qwen_64_4096.json \
  --model /mnt/models/Qwen3-32B \
  --tp-size 4 --max-model-len 4096 --gpu-memory-utilization 0.8 \
  --max-tokens 1 --planner-mode hybrid --our-max 4096 \
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

### 8.10 Qwen3-32B GraphAdmit Strict Same-Engine bs128

```bash
conda activate crossstage
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

### 8.11 Qwen3 SGLang PCG Baselines

```bash
source /home/zhujianian/miniconda3/etc/profile.d/conda.sh
conda activate sglang-bench
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmarks/sglang_flowprefill_workload.py \
  --workload results/qwentrace_morspec_qwen_64_4096.json \
  --model /mnt/models/Qwen3-32B \
  --tp-size 4 \
  --context-length 4096 \
  --mem-fraction-static 0.8 \
  --max-new-tokens 1 \
  --configs piecewise \
  --piecewise-cuda-graph-max-tokens 4096 \
  --warmup-count 3 \
  --output results/sglang_qwen3_32b_64_piecewise_e2e_0510post1_warm3.json

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python benchmarks/sglang_flowprefill_workload.py \
  --workload results/qwentrace_morspec_qwen_16_4096.json \
  --model /mnt/models/Qwen3-235B-A22B-Instruct-2507 \
  --tp-size 8 \
  --context-length 4096 \
  --mem-fraction-static 0.8 \
  --max-new-tokens 1 \
  --configs piecewise \
  --piecewise-cuda-graph-max-tokens 4096 \
  --moe-runner-backend auto \
  --moe-a2a-backend none \
  --warmup-count 3 \
  --output results/sglang_qwen3_235b_16_piecewise_e2e_0510post1_warm3.json
```

### 8.12 Residual Capture Microbenchmark

```bash
source /home/zhujianian/miniconda3/etc/profile.d/conda.sh
conda activate crossstage
python benchmarks/compositional_cuda_graph_microbench.py \
  --output results/compositional_cuda_graph_microbench.json \
  --repeat 30 --warmups 3
```

### 8.13 dInfer Admission

```bash
conda activate crossstage
python benchmarks/dinfer_qwentrace_offline.py \
  --model /mnt/models/LLaDA2.0-mini \
  --workload results/qwentrace_morspec_llada_32_2048.json \
  --limit 32 \
  --gen-length 64 \
  --block-length 16 \
  --threshold 0.99 \
  --warmups 1 \
  --validate-replay \
  --validation-mode decoded \
  --validation-atol 0.0 \
  --validation-rtol 0.0 \
  --admit-after 1 \
  --validation-interval 16 \
  --graph-max-templates 8 \
  --graph-min-free-memory-mb 4096 \
  --cleanup-policy memory_guard \
  --cleanup-min-free-memory-mb 40960 \
  --cleanup-max-interval 4 \
  --model-warmup-tokens 128 \
  --output results/dinfer_qwentrace_llada2_mini_32_runtime_admission_memguard_latest.json
```

### 8.14 Failure / Useful-Coverage Analysis

```bash
python benchmarks/analyze_cuda_graph_failure_modes.py \
  --output-dir results/cg_failure_analysis_latest
```

### 8.15 Key-Collapse Analysis

```bash
python benchmarks/analyze_vllm_keycollapse_runtime.py \
  --dispatcher <dispatcher.jsonl> \
  --runner <runner.jsonl> \
  --attention <attention.jsonl> \
  --serving-only \
  --output <analysis.json>
```

## 9. Paper Claim That Is Safe Today

The safe claim is:

> GraphAdmit makes CUDA Graph useful for dynamic LLM serving by classifying dynamicity, canonicalizing recoverable metadata/address dynamics into graph-stable templates, and executing templates only when online admission proves them token-correct and faster than fallback. Dynamic workloads always run; staticity is recovered only when safe and profitable.

Concrete supported claims:

- vLLM/Qwen3-235B: 2.69x average speedup over vLLM max512 CP with strict same-engine validation; mean/P95 slightly beat local SGLang PCG, P99 slightly loses.
- vLLM/Qwen3-32B: 1.06x-1.08x average speedup over vLLM max512 CP with strict same-engine validation and whole-run token identity.
- vLLM/Qwen3.5 hybrid-MoE: strict isolated probing detects unsafe extra-template replay and blacklists it; the fail-closed measured path is 1.38x faster than vLLM CP and faster on average than local SGLang default CG, with one small/default-path nondeterminism caveat.
- Legacy Qwen3.5 shadow-baseline: 5.95x over vLLM CP and 2.72x over SGLang default, useful as potential evidence but not the primary correctness claim.
- dInfer/LLaDA2.0-mini: 1.58x to 1.71x speedup with decoded-token correctness.
- Torch microbench: exact-shape naive CG is weak; compositional packing/padding/residual capture explain why dynamic workloads need more than a fixed graph list.

Claims to avoid:

- Do not claim universal dominance over SGLang.
- Do not claim whole-run token identity or admitted extra-template replay for Qwen3.5 strict mode yet.
- Do not claim scheduler-induced tail wins are proven.
- Do not claim MoE fused-kernel/all-to-all dispatch template selection is complete.
- Do not claim arbitrary dynamic functions are automatically graphable.

## 10. Bottom Line

The current system is already a strong artifact for the paper story:

- It is better than vLLM's default/static CUDA Graph path because it recovers useful templates beyond the fixed graph family and rejects bad graph paths.
- It can match or beat SGLang PCG in important dynamic cases, especially Qwen3-235B strict same-engine.  On Qwen3.5, the strongest claim is safer: GraphAdmit refuses unsafe hybrid/MoE extra graphs and still beats local SGLang default on average, while SGLang remains a serious baseline.
- The Torch microbenchmark should be framed as mechanism evidence, not an end-to-end serving win.
- The most defensible framing is "safe online staticity recovery", not "all dynamicity goes into CUDA Graph."
