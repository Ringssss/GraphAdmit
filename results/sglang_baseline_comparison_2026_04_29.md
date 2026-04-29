# SGLang Baseline Comparison

Date: 2026-04-29

## Setup

- Machine: 8x NVIDIA H100 80GB.
- Workloads:
  - Qwen3-32B: `results/qwentrace_morspec_qwen_64_4096.json`, 64 requests.
  - Qwen3-235B-A22B: `results/qwentrace_morspec_qwen_16_4096.json`, 16 requests.
- SGLang environment: conda env `sglang-bench`, installed with Tsinghua PyPI mirror.
- Final SGLang version: `0.5.10.post1`, `torch 2.9.1+cu128`, CUDA 12.8.
- Main SGLang baseline: Piecewise CUDA Graph (PCG), `piecewise_cuda_graph_max_tokens=4096`, `warmup_count=3`.

SGLang PCG is the right baseline to compare against. Their current documentation says PCG is enabled by default, captures prefill/extend pieces for predefined token lengths, pads runtime token counts up to the nearest captured size, and falls back when a token count exceeds the largest capture size:
https://sgl-project.github.io/advanced_features/piecewise_cuda_graph.html

## Commands

Qwen3-32B latest SGLang PCG:

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
```

Qwen3-235B latest SGLang PCG:

```bash
source /home/zhujianian/miniconda3/etc/profile.d/conda.sh
conda activate sglang-bench
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

## Main Results

| Workload | System | Avg ms | P50 ms | P95 ms | P99 ms | Init s | Correct |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-32B | vLLM CP baseline | 43.57 | 44.67 | 113.72 | 127.25 | 74.4 | true |
| Qwen3-32B | GraphAdmit current | 39.71 | 34.81 | 115.62 | 127.80 | 88.4 | true |
| Qwen3-32B | SGLang PCG 0.5.10 warm3 | 43.29 | 40.74 | 111.27 | 123.17 | 91.4 | true |
| Qwen3-235B | vLLM CP baseline | 139.04 | 211.68 | 232.76 | 241.36 | 142.1 | true |
| Qwen3-235B | GraphAdmit current | 73.03 | 55.14 | 232.69 | 238.01 | 166.5 | true |
| Qwen3-235B | SGLang PCG 0.5.10 warm3 | 58.37 | 55.51 | 127.29 | 132.81 | 201.4 | true |

Raw result files:

- `results/vllm_qwen3_32b_64_residual_learned_v2_e2e.json`
- `results/vllm_qwen3_235b_16_residual_learned_896_e2e.json`
- `results/sglang_qwen3_32b_64_piecewise_e2e_0510post1_warm3.json`
- `results/sglang_qwen3_235b_16_piecewise_e2e_0510post1_warm3.json`

Older SGLang 0.5.9 raw files are kept for reproducibility:

- `results/sglang_qwen3_32b_64_piecewise_e2e.json`
- `results/sglang_qwen3_235b_16_piecewise_e2e.json`

## Interpretation

Qwen3-32B:

- GraphAdmit current is still better on average: `39.71 ms` vs latest SGLang PCG `43.29 ms`, about `1.09x` faster on mean latency.
- SGLang PCG has better P95/P99: `111.27/123.17 ms` vs GraphAdmit `115.62/127.80 ms`.
- This means the current 32B story is not "dominates SGLang"; it is "better mean useful coverage, but tail policy still needs refinement."

Qwen3-235B:

- Latest SGLang PCG with sufficient warmup is stronger than GraphAdmit current: `58.37 ms` vs `73.03 ms` average, `127.29 ms` vs `232.69 ms` P95.
- GraphAdmit still gives a large win over vLLM CP: `73.03 ms` vs `139.04 ms`, but it is not enough against latest SGLang PCG.
- The main weakness is visible in long requests: GraphAdmit falls back or runs near fallback for 2829/3023-token requests, while SGLang PCG serves them with token-level PCG templates around 125-134 ms.

## Implications For The Paper

The paper should not claim a blanket win over SGLang. The stronger and more defensible claim is:

> Static CUDA Graph policies in vLLM/PyTorch can be incorrect or unprofitable under dynamic serving; GraphAdmit adds online admission, correctness guards, and workload-aware template selection. Compared with vLLM, it substantially improves useful coverage. Compared with SGLang PCG, GraphAdmit exposes a complementary design point: fail-closed admission and dynamicity classification, but it must close the steady-state tail gap on long-prefill MoE traces.

This changes the positioning:

- Keep vLLM/Torch blind CG as the primary "unsafe/unprofitable static graph" motivation.
- Use SGLang PCG as the strongest modern baseline, not as a strawman.
- The differentiator against SGLang should be correctness/admission/general dynamicity, not just latency, unless we add token-axis PCG-equivalent coverage for long prefill.

## Concrete Next Steps

1. Add token-axis residual templates for long prefill.

   Current residual capture admits `(92,886] -> 896` for 235B. This misses 2829/3023-token requests, exactly where SGLang PCG wins. Add candidate templates around `1280, 2048, 3072, 3328, 4096`, but admit them only if shadow fallback timing proves positive ROI.

2. Add SGLang-style token piecewise fallback inside GraphAdmit.

   SGLang's PCG advantage is not just bucket count; it splits the model around graph-hostile ops and replays capturable pieces. Our partial-graph component should target the same long-prefill path: attention/MoE split points run guarded fallback or eager, dense blocks replay.

3. Make admission tail-aware.

   Current policy optimizes mean useful coverage. The 235B comparison shows this can preserve average speedup while leaving P95/P99 bad. Admission should include a tail guard:

   - admit template only if `p95_graph <= p95_fallback * (1 - eps)` for that bucket, or
   - cap fallback exposure for high-token bins by forced exploration of long-token candidates.

4. Compare template cost, not only latency.

   SGLang PCG prepares many templates: 52 batch-size graphs plus 50 token-count PCG captures in these runs. GraphAdmit should report:

   - number of admitted templates,
   - initialization/capture time,
   - graph memory overhead,
   - negative graph rate,
   - useful coverage.

5. Treat SGLang compatibility limits as opportunity.

   SGLang PCG docs list auto-disable cases such as MoE A2A backend, LoRA, multimodal/VLM, DLLM, deterministic inference, and disaggregation. GraphAdmit can be positioned as a more general fail-closed admission layer if we demonstrate at least two cases where PCG is disabled or unsafe, while GraphAdmit still recovers safe subgraphs.

## Bottom Line

The current version is solid versus vLLM CP and old SGLang, but latest SGLang PCG is a very strong baseline. To make the paper robust, we should not frame the claim as "we beat SGLang everywhere." The right target is:

- beat vLLM/Torch blind CG on correctness and useful coverage,
- match or beat SGLang PCG on average for dense 32B,
- close the long-prefill MoE tail gap for 235B,
- show broader applicability where SGLang PCG is disabled or not safe to force.
