# GraphAdmit E2E Evidence Update, 2026-05-01

This update records the post-drift-action code state and fresh vLLM/GraphAdmit
end-to-end runs on idle H100 GPUs.  The main conclusion is narrower and stronger
than "graph every dynamic workload": GraphAdmit keeps dynamic workloads running,
admits only validated useful templates, and falls back when a template is unsafe
or unprofitable.

## Code Delta

- `SameEngineLiveCaptureManager.apply_drift_decision(...)` now connects workload
  drift decisions to live template state.
- `correctness_drift` blacklists recent templates.
- `negative_graph_rate_drift` increases shadow validation frequency.
- `token_distribution_shift` and `refresh_admission` move admitted templates
  back to shadow validation.
- `simulate_live_capture_drift.py` now applies drift decisions to the live
  manager, instead of only logging detector output.
- The runtime component validator now emits explicit drift-action evidence.

## Validation

Commands completed:

```bash
conda run -n crossstage python -m py_compile benchmarks/simulate_live_capture_drift.py benchmarks/validate_staticity_runtime_components.py prefill_graph/runtime/live_capture.py tests/test_staticity_control_plane.py
conda run -n crossstage pytest -q tests/test_staticity_control_plane.py tests/test_graphadmit_cli.py
conda run -n crossstage python benchmarks/validate_staticity_runtime_components.py --output results/staticity_runtime_components_validation_20260501_drift_action_v2.json
conda run -n crossstage python benchmarks/simulate_live_capture_drift.py --phase-a results/qwentrace_morspec_qwen_64_4096.json --phase-b results/qwentrace_morspec_qwen_128_4096.json --output results/live_capture_drift_qwen64_to_qwen128_20260501_v2.json --limit-a 64 --limit-b 128
```

Results:

- Unit tests: `12 passed`.
- Drift simulation with live manager actions:
  - static policy avg: `56.75 ms`
  - same-engine live capture avg: `47.13 ms`
  - speedup: `1.20x`
  - useful graph replays: `75 -> 84`
  - drift actions: `2`

## Fresh vLLM vs GraphAdmit E2E

All fresh runs used the same vLLM harness:
`benchmarks/vllm_flowprefill_workload.py`, `max_new_tokens=1`,
`disable_prefix_caching`, and idle GPUs.

| Model / workload | vLLM CP avg ms | GraphAdmit deploy avg ms | Speedup vs vLLM | P95 ms, vLLM -> GA | P99 ms, vLLM -> GA | Correctness evidence |
|---|---:|---:|---:|---:|---:|---|
| Qwen3-32B, bs64 trace, TP4 | 43.89 | 40.02 | 1.10x | 114.28 -> 111.71 | 126.97 -> 126.35 | whole-run identical, 31/31 live obs correct |
| Qwen3-32B, bs128 trace, TP4 | 41.59 | 39.19 | 1.06x | 113.00 -> 111.66 | 126.33 -> 124.96 | whole-run identical, 62/62 live obs correct |
| Qwen3-235B-A22B, bs16 trace, TP8 | 151.45 | 55.61 | 2.72x | 240.25 -> 124.63 | 247.94 -> 129.93 | whole-run identical, 10/10 live obs correct/useful |
| Qwen3.5-35B-A3B, bs16 trace, TP4 | 118.28 | 91.98 | 1.29x | 137.37 -> 147.42 | 143.58 -> 194.18 | 25/28 probe obs correct, 3 crash templates blacklisted; whole-run default-path diff remains |

Interpretation:

- Qwen3-235B is the strongest paper-grade result: large MoE, full output match,
  all admitted observations correct and useful, and better average/P95/P99 than
  vLLM CP.
- Qwen3-32B is a clean dense result: modest but repeatable average improvement,
  whole-run token identity, and no live correctness failures.
- Qwen3.5 is the safety stress case: GraphAdmit improves average latency in
  deployment mode, but the whole-run output is not identical because of a
  default-path backend difference.  The defensible claim is that GraphAdmit
  probes and blacklists unsafe extra templates, not that every Qwen3.5 mode is
  globally deterministic.

## SGLang Reference Baselines

These are existing local SGLang result files from the same repo and hardware
setup; they were not modified by this patch.

| Model / workload | SGLang mode | Avg ms | P95 ms | P99 ms | GraphAdmit relation |
|---|---|---:|---:|---:|---|
| Qwen3-32B, bs64 | piecewise CUDA Graph | 43.29 | 111.27 | 123.17 | GraphAdmit deploy is faster on avg, similar P95, slightly worse P99 |
| Qwen3-235B, bs16 | piecewise CUDA Graph | 58.37 | 127.29 | 132.81 | GraphAdmit deploy is faster on avg/P95/P99 |
| Qwen3.5-35B, bs16 | default CUDA Graph | 116.47 | 132.17 | 135.07 | GraphAdmit deploy is faster on avg, worse tail |
| Qwen3.5-35B, bs16 | piecewise CUDA Graph | 286.58 | 702.29 | 790.08 | GraphAdmit deploy is much faster |

## Strict GraphAdmit-only Ablation

`--graphadmit-only` forces vLLM native/default graph hits to fallback unless the
runtime action is GraphAdmit-owned.  This is useful for attribution and safety,
but it is not the deployment mode because it disables vLLM's already-profitable
small graphs.

| Run | vLLM CP avg ms | Strict GraphAdmit-only avg ms | Observation |
|---|---:|---:|---|
| Qwen3-32B bs64 | 43.73 | 49.02 | whole-run identical, 31/31 obs correct, but avg slower |
| Qwen3.5 bs16 | 117.29 | 124.45 | 25/28 probe obs correct, 3 crash blacklists, avg slower |

This ablation is important because it separates two claims:

- Deployment claim: GraphAdmit is additive on top of good existing vLLM graphs.
- Attribution/safety claim: unowned dynamic graph paths can be disabled, but
  doing that for all native small graphs is too conservative for performance.

## Output Files

- `results/vllm_qwen3_32b_deploy_bs64_20260501.json`
- `results/live_obs_qwen3_32b_deploy_bs64_20260501.jsonl`
- `results/vllm_qwen3_32b_deploy_bs128_20260501.json`
- `results/live_obs_qwen3_32b_deploy_bs128_20260501.jsonl`
- `results/vllm_qwen3_235b_deploy_16_20260501.json`
- `results/live_obs_qwen3_235b_deploy_16_20260501.jsonl`
- `results/vllm_qwen35_deploy_probe_16_20260501.json`
- `results/live_obs_qwen35_deploy_probe_16_20260501.jsonl`
- `results/vllm_qwen35_strict_graphadmit_only_16_20260501.json`
- `results/live_obs_qwen35_strict_graphadmit_only_16_20260501.jsonl`
- `results/staticity_runtime_components_validation_20260501_drift_action_v2.json`
- `results/live_capture_drift_qwen64_to_qwen128_20260501_v2.json`

## Bottom Line

The current evidence supports a strong, credible systems claim:

> GraphAdmit does not promise to graph arbitrary dynamic behavior.  It lets
> arbitrary dynamic workloads enter the system, recovers staticity only for
> representational/address/template dynamics that can be validated, and refuses
> unsafe or slower templates online.

For OSDI/SOSP framing, the cleanest primary results are Qwen3-235B and Qwen3-32B.
Qwen3.5 should be presented as the hybrid/MoE safety stress case: useful average
gain in deployment mode, but with explicit fail-closed blacklisting and a
remaining whole-run nondeterminism caveat.
