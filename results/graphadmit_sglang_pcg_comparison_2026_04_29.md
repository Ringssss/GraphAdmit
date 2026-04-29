# GraphAdmit vs SGLang PCG Follow-up

Date: 2026-04-29 19:34:03 CST

## What Changed

- Added SGLang-style residual bucket presets to GraphAdmit's residual capture planner.
- Added exact-template evidence guards so a large template can only be admitted from measurements that actually ran that template.
- Added tail-aware admission thresholds: tail templates require stricter useful-rate, saving, and no-regression evidence.
- Added demand-filtered exploration policy generation from workload traces.
- Updated the vLLM E2E harness to derive runtime capture sizes from admitted graph actions only, avoiding accidental capture of fallback-only buckets.

## End-to-End Results

All rows below use the same Qwen trace replay harness. Output checks passed with `all_same_outputs_vs_reference=True`.

| Model / trace | System | Avg ms | P95 ms | P99 ms | Init s | Notes |
|---|---:|---:|---:|---:|---:|---|
| Qwen3-235B, n=16 | vLLM max512 CP | 141.29 | 239.79 | 246.08 | 140.93 | baseline |
| Qwen3-235B, n=16 | old GraphAdmit 896 | 73.03 | 232.69 | 238.01 | 166.49 | old residual policy |
| Qwen3-235B, n=16 | SGLang piecewise CG | 58.37 | 127.29 | 132.81 | 201.36 | broad PCG family |
| Qwen3-235B, n=16 | GraphAdmit PCG-tail | 55.48 | 125.22 | 131.12 | 166.06 | 4 extra templates: 768, 832, 896, 3072 |
| Qwen3-32B, n=64 | vLLM max512 CP | 43.37 | 113.69 | 126.89 | 71.40 | baseline |
| Qwen3-32B, n=64 | old GraphAdmit 1024 | 39.71 | 115.62 | 127.80 | 88.39 | old residual policy |
| Qwen3-32B, n=64 | SGLang piecewise CG | 43.29 | 111.27 | 123.17 | 91.38 | broad PCG family |
| Qwen3-32B, n=64 | GraphAdmit PCG-tail | 38.99 | 115.85 | 127.24 | 86.18 | 6 extra templates: 768, 832, 896, 960, 1536, 2304 |

## Main Takeaways

- Qwen3-235B: the new policy is better than vLLM max512 CP by 2.55x avg, 1.91x P95, and 1.88x P99. It also slightly beats SGLang piecewise CG: 1.05x avg, 1.02x P95, 1.01x P99.
- Qwen3-32B: the new policy is better than vLLM max512 CP by 1.11x avg, but tail is roughly neutral/slightly worse than vLLM. It is 1.11x faster than SGLang on average, but SGLang has better P95/P99 by about 4 ms.
- The 32B result suggests GraphAdmit's admission is correctly conservative for large templates, but the scheduler/tail path still needs work if the paper wants a strict "tail never regresses" claim on dense 32B.
- The 235B result is the strongest evidence: GraphAdmit can match or beat SGLang-style piecewise coverage with fewer admitted extra templates and fail-closed admission.

## SGLang CUDA Graph Mechanism Observed

Local source checked under `/home/zhujianian/crossstage/prism-research/python/sglang`.

- `sglang/srt/model_executor/cuda_graph_runner.py` captures decode batch-size buckets: `[1,2,3,4] + [8,16,...,160]` when padding is enabled.
- Replay pads a raw batch to the next captured batch size, copies request metadata into persistent buffers, updates attention metadata, and replays the captured graph.
- `sglang/srt/layers/attention/flashinfer_backend.py` preallocates CUDA-graph KV index buffers and stores per-batch-size FlashInfer decode wrappers.
- This is strong engineering for decode/padded batch-size staticity, but it is not online admission: it does not learn whether a template is correct and faster for the current workload before admitting it.

## Verification

- `python -m py_compile prefill_graph/runtime/residual_capture.py prefill_graph/runtime/__init__.py benchmarks/plan_residual_capture_policy.py benchmarks/vllm_flowprefill_workload.py`
- `pytest -q tests/test_staticity_control_plane.py` after installing pytest into `crossstage` with the Tsinghua PyPI mirror: `4 passed`.
- Qwen3-235B E2E: `results/vllm_qwen3_235b_16_pcg_tail_learned_e2e.json`.
- Qwen3-32B E2E: `results/vllm_qwen3_32b_64_pcg_tail_learned_e2e.json`.
