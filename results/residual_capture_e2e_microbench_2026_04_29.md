# Residual Capture and Compositional CUDA Graph Replay Results

Date: 2026-04-29

## Implemented Pieces

- Runtime planner: `prefill_graph/runtime/residual_capture.py`
- CLI policy builder: `benchmarks/plan_residual_capture_policy.py`
- Microbenchmark: `benchmarks/compositional_cuda_graph_microbench.py`
- Runtime export: `prefill_graph/runtime/__init__.py`

The residual-capture path is intentionally fail-closed:

- It requires explicit output correctness evidence from the measured E2E run.
- It can require `candidate_graph_allowed` evidence from the candidate policy, so fallback-only rows cannot seed a new graph template.
- It admits only segments that pass useful-rate, average-saving, p95-regression, and max-regression guards.
- It emits fallback rules for unprofitable or unsafe ranges.
- In the current vLLM integration this is `next_round_residual_capture`: the engine explores and validates, then the next engine start uses the learned capture set. vLLM still fixes CUDA graph capture sizes at initialization.

## Compositional Microbenchmark

Command:

```bash
source /home/zhujianian/miniconda3/etc/profile.d/conda.sh
conda activate crossstage
python benchmarks/compositional_cuda_graph_microbench.py \
  --output results/compositional_cuda_graph_microbench.json \
  --repeat 30 --warmups 3
```

Workload: token-local MLP, 20 dynamic lengths, BF16, CUDA graph correctness checked against eager.

| Plan | Total ms | Speedup | Graph replays | Fallbacks | Padding waste | Correct |
|---|---:|---:|---:|---:|---:|---|
| eager_dynamic | 9.5457 | 1.00x | 0 | 20 | 0.00% | yes |
| exact_only_graph_32_64 | 8.5355 | 1.12x | 2 | 18 | 0.00% | yes |
| pad_to_next_template | 4.9662 | 1.92x | 20 | 0 | 23.08% | yes |
| tile_32_composition | 9.5572 | 1.00x | 39 | 0 | 23.08% | yes |
| pack_adjacent_requests | 2.6136 | 3.65x | 10 | 0 | 0.00% | yes |
| residual_learn_then_replay | 5.1191 | 1.86x | 19 | 0 | 46.99% | yes |

Interpretation:

- Exact-shape CUDA Graph is brittle: it only hits 32 and 64, so dynamic sizes such as 55/41 mostly fall back.
- Padding is useful when one replay replaces many launches and the padded work is modest.
- Fixed tiling is universal but can lose when the launch count grows.
- Packing adjacent independent token-local requests is the best case: examples like 55+41 can use one 96-token replay, with no padding waste and fewer launches.
- Residual capture is useful as an online adaptation mechanism, but the admitted templates must be cost-aware because over-capturing can create high padding/capture overhead.

## Qwen3-32B Dense E2E

Exploration learned policy:

- `(734,758] -> template 768`, n=14, useful_rate=1.00, avg_saving=9.185 ms
- `(782,789] -> template 832`, n=3, useful_rate=1.00, avg_saving=1.742 ms
- `(795,1012] -> template 1024`, n=11, useful_rate=1.00, avg_saving=10.699 ms
- All other ranges fall back to CP.

Final command used the learned policy with extra capture sizes `768,832,1024`:

```bash
python benchmarks/vllm_flowprefill_workload.py \
  --workload results/qwentrace_morspec_qwen_64_4096.json \
  --model /mnt/models/Qwen3-32B \
  --tp-size 4 --max-model-len 4096 --gpu-memory-utilization 0.8 \
  --max-tokens 1 --planner-mode hybrid --our-max 1024 \
  --extra-capture-sizes 768,832,1024 --max-extra-capture-size 1024 \
  --skip-eager --configs cp,runtime_cp \
  --runtime-policy results/runtime_policy_vllm_qwen3_32b_64_residual_learned_from_explore.json \
  --runtime-base-capture-size 512 --cudagraph-mode FULL_AND_PIECEWISE \
  --fixed-metadata-arena --fixed-metadata-arena-max-reqs 8 \
  --fixed-metadata-arena-min-tokens 734 --fixed-metadata-arena-max-tokens 1012 \
  --full-key-collapse \
  --output results/vllm_qwen3_32b_64_residual_learned_v2_e2e.json \
  --profile-prefix results/vllm_qwen3_32b_64_residual_learned_v2_e2e
```

| Config | Avg ms | P50 ms | P95 ms | P99 ms | Init s | Speedup | Correct |
|---|---:|---:|---:|---:|---:|---:|---|
| vLLM graph max512 CP FULL_AND_PIECEWISE | 43.5699 | 44.6669 | 113.7160 | 127.2549 | 74.44 | 1.00x | yes |
| Single-engine runtime hybrid max1024 CP | 39.7108 | 34.8097 | 115.6242 | 127.8034 | 88.39 | 1.10x | yes |

Profiling diagnosis:

- `key_collapse_opportunity=true`
- `tokens_with_multiple_layouts=49`
- `policy_window_arena_active=true`
- `policy_window_request_metadata_shapes_fixed=true`
- `policy_window_request_metadata_ptrs_fixed=true`
- Remaining blocker: multi-request correctness guard needs integrated metadata arena.

32B conclusion:

- Residual capture improves average latency by about 1.10x.
- The win is narrow and workload-specific; broad exploration shows many candidate ranges are not worth graphing.
- The sweet spot is mid-token windows around 735-758 and 795-1012 where graph replay consistently beats CP.
- P95 is slightly worse by about 1.9 ms, so a production admission rule should keep the current tail guard or require more samples before admitting narrow low-saving windows like 782-789.

## Qwen3-235B MoE E2E

Exploration learned policy:

- `(92,886] -> template 896`, n=12, useful_rate=1.00, avg_saving=93.014 ms
- `<=92` and `>886` fall back to CP.

Final command used only the learned 896-token residual template:

```bash
python benchmarks/vllm_flowprefill_workload.py \
  --workload results/qwentrace_morspec_qwen_16_4096.json \
  --model /mnt/models/Qwen3-235B-A22B-Instruct-2507 \
  --tp-size 8 --max-model-len 4096 --gpu-memory-utilization 0.8 \
  --max-tokens 1 --planner-mode hybrid --our-max 1024 \
  --extra-capture-sizes 896 --max-extra-capture-size 1024 \
  --skip-eager --configs cp,runtime_cp \
  --runtime-policy results/runtime_policy_vllm_qwen3_235b_16_residual_learned_from_explore.json \
  --runtime-base-capture-size 512 --cudagraph-mode FULL_AND_PIECEWISE \
  --fixed-metadata-arena --fixed-metadata-arena-max-reqs 8 \
  --fixed-metadata-arena-min-tokens 92 --fixed-metadata-arena-max-tokens 886 \
  --full-key-collapse --enable-return-routed-experts \
  --output results/vllm_qwen3_235b_16_residual_learned_896_e2e.json \
  --profile-prefix results/vllm_qwen3_235b_16_residual_learned_896_e2e
```

| Config | Avg ms | P50 ms | P95 ms | P99 ms | Init s | Speedup | Correct |
|---|---:|---:|---:|---:|---:|---:|---|
| vLLM graph max512 CP FULL_AND_PIECEWISE | 139.0360 | 211.6763 | 232.7588 | 241.3604 | 142.08 | 1.00x | yes |
| Single-engine runtime hybrid max1024 CP | 73.0310 | 55.1368 | 232.6869 | 238.0120 | 166.49 | 1.90x | yes |

Profiling diagnosis:

- `key_collapse_opportunity=true`
- `tokens_with_multiple_layouts=47`
- `policy_window_arena_active=true`
- `policy_window_request_metadata_shapes_fixed=true`
- `policy_window_request_metadata_ptrs_fixed=true`
- `policy_window_token_shape_dynamic=false`
- Remaining blocker: MoE expert-routing templates and an effective scheduler.

235B conclusion:

- One extra 896-token template covers most of the mid-token workload and gives 1.90x average speedup.
- Tail is not worse: P95 is essentially unchanged and P99 improves slightly.
- The graph pool cost increased from about 1.19 GiB to about 1.30 GiB in this run, so the single residual template has a good speed/memory tradeoff.

## Sweet Spots and Boundaries

Residual capture helps when:

- misses recur in a stable token interval;
- metadata address and shape can be canonicalized in that interval;
- graph replay wins by enough to amortize capture, warmup, and extra graph-pool memory;
- padding waste is modest or can be eliminated by request packing;
- the scheduler can accumulate compatible requests without hurting P95/P99.

Residual capture should reject when:

- candidate rows are fallback-only rather than real graph replay evidence;
- semantic dynamic behavior is present, such as attention branch, sampling branch, diffusion step, or MoE routing choice that is not guarded;
- regression tails exceed the guard even if the average is faster;
- fixed tiling causes too many graph launches;
- the admitted range has too few samples or only wins because of noise.

## Paper Takeaway

The stronger claim is not "any dynamic workload can enter CUDA Graph".

The defensible claim is:

> The serving engine can accept arbitrary dynamic workloads, identify which dynamicity is recoverable, admit only token-correct and faster graph templates, and route the rest to fallback without being forced into incorrect graph replay.

The implemented residual-capture loop supports that story:

1. Explore dynamic misses with candidate templates.
2. Validate token correctness against fallback.
3. Admit only useful coverage.
4. Emit fixed metadata arena ranges and capture sizes for the next runtime.
5. Blacklist/fallback rejected regions.

The current production gap is live in-engine capture/eviction inside an already-running vLLM engine. The current implementation is next-round online adaptation because vLLM capture sizes are fixed during initialization.
