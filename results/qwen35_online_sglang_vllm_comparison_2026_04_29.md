# Qwen3.5-35B Hybrid/MoE Online Admission Results
Date: 2026-04-29. Model: `/mnt/models/Qwen3.5-35B-A3B` (MoE + hybrid linear/full attention). Workload: `results/qwentrace_morspec_qwen_16_4096.json`, 16 requests, max_new_tokens=1, TP=4, max_model_len=4096.
## End-to-End Latency
| System | Avg ms | P50 ms | P95 ms | P99 ms | Init s |
|---|---:|---:|---:|---:|---:|
| vLLM eager/no-CG | 314.66 | 137.22 | 904.00 | 991.73 | 74.0 |
| vLLM max512 CP | 254.59 | 115.63 | 759.81 | 811.99 | 82.7 |
| Blind extended CG | 64.25 | 39.82 | 143.41 | 351.57 | 82.0 |
| GraphAdmit live runtime | 42.82 | 40.25 | 56.68 | 60.02 | 82.8 |
| SGLang default CG | 116.47 | 114.61 | 132.17 | 135.07 | 50.6 |
| SGLang piecewise CG | 286.58 | 249.53 | 702.29 | 790.08 | 57.8 |

GraphAdmit live runtime is 5.95x faster than vLLM max512 CP on average and 13.41x faster at P95. It is 2.72x faster than SGLang default CG on average and 2.33x faster at P95.

## Online Admission Evidence
- Live observation file: `results/live_obs_qwen35_35b_16.jsonl`, 10 graph-template observations, 10/10 token-correct.
- Templates observed: `tokens=3072, tokens=768, tokens=832, tokens=896`.
- Runtime dispatcher profile: `results/vllm_qwen35_35b_16_live_compare_7__single_engine_runtime_hybrid_max4096_cp_dispatcher.jsonl`. It contains both `explore_until_min_samples` and `live_admitted` decisions, so the run is not simply replaying an offline learned policy.
- Policy: `results/runtime_policy_vllm_qwen35_35b_generic_pcg_live.json`, a generic PCG-family exploration policy, not learned from Qwen3.5 trace data.

## Correctness Notes
- The 10 extra-template GraphAdmit observations are token-correct against the shadow CP fallback.
- Whole-run `all_same_outputs_vs_first` is false for GraphAdmit because request index 7 (93 tokens) differs from CP baseline, but that request is a small/default path, not an extra GraphAdmit template.
- Eager/no-CG also differs from CP on request index 5 (494 tokens), so this new hybrid model has backend-level one-token nondeterminism across vLLM modes. Do not claim whole-run token identity for Qwen3.5 until this is fixed or isolated.

## SGLang Notes
- SGLang logs: `Disabling overlap schedule since mamba no_buffer is not compatible with overlap schedule`. This likely explains why SGLang is weaker on this hybrid attention model than on Qwen3/Qwen3-dense.
- SGLang piecewise CG is worse than SGLang default CG here: 286.58 ms vs 116.47 ms average.

## Torch Naive CG Micro-Bench
Raw `torch.cuda.CUDAGraph` is not a fair end-to-end baseline for the 35B serving run because it lacks the tensor-parallel serving/runtime stack used by vLLM/SGLang. I used a token-local MLP micro-bench to isolate naive exact-shape CG vs padding/tiling/packing/residual capture.

| Plan | Total ms | Speedup vs eager | Graph replays | Fallbacks | Padding waste | Correct |
|---|---:|---:|---:|---:|---:|---|
| eager_dynamic | 9.297 | 1.00x | 0 | 20 | 0.0% | True |
| exact_only_graph_32_64 | 8.429 | 1.10x | 2 | 18 | 0.0% | True |
| pad_to_next_template | 4.944 | 1.88x | 20 | 0 | 23.1% | True |
| tile_32_composition | 9.537 | 0.97x | 39 | 0 | 23.1% | True |
| pack_adjacent_requests | 2.608 | 3.56x | 10 | 0 | 0.0% | True |
| residual_learn_then_replay | 5.166 | 1.80x | 19 | 0 | 47.0% | True |

The weak baseline is `exact_only_graph_32_64`: it only gives 1.10x because most dynamic sizes miss. Compositional packing gives 3.56x by packing 55+41 into a 96-token graph with zero padding waste.
