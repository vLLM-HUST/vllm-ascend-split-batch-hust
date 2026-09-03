# Ascend Split-Batch and Full-Graph Parallel

Owner-led research carrier for split-batch planning, dual-stream replay, and full-graph parallel execution. Core owns graph lifecycle seams; this repository owns policy, runners, validation, and evidence.

**Status: default-off.** Two independently gated capabilities are installed:

1. **Split-batch dual-pad planner** — pure planning + prechecks (`import_only`,
   no vLLM behavior change without the host contract).
2. **Cascade two-stage decode** (`vllm.general_plugins` entry
   `cascade-attention`) — eager + graph-mode two-stage cascade attention for
   vllm-ascend, patched entirely from the plugin; official vllm /
   vllm-ascend sources are NOT modified.

Technical ownership belongs to @Raing5Days, @ilnnfover. Source extraction must
preserve exact authorship, license, tests, constraints, and evidence before
activation is considered.

See [MAINTAINERS.md](MAINTAINERS.md) and [PROVENANCE.md](PROVENANCE.md).

## Cascade attention plugin (default-off)

Entry point: `vllm.general_plugins` ->
`cascade-attention = vllm_ascend_split_batch.cascade_plugin:load`.

Environment gates (all default off):

| Variable | Meaning |
|---|---|
| `VLLM_ASCEND_ENABLE_CASCADE_DECODE=1` | enable the two-stage cascade decode |
| `VLLM_ASCEND_ENABLE_CASCADE_GRAPH=1` | additionally enable the graph-mode (aclgraph) cascade twin |
| `VLLM_ASCEND_CASCADE_MIN_PREFIX` (8192) | min shared prefix tokens to trigger |
| `VLLM_ASCEND_CASCADE_MIN_REQS` (32) | min batch size to trigger |
| `VLLM_ASCEND_CASCADE_PRECISION` | `bf16` (Tier-0, default) or `fp32` (Tier-1, eager-only) |
| `VLLM_ASCEND_CASCADE_STRICT=1` | force the standard full-KV path (bit-exact reference) |
| `VLLM_ASCEND_CASCADE_TRACE=1` | per-step plugin trace to stdout |
| `VLLM_ASCEND_CASCADE_UPDATE_SKIP_STABLE` (1) | skip the stage-1 graph re-bind when its inputs are step-invariant (0 restores always-rebind) |

With every gate unset the patched methods are no-ops and the serving path is
bit-identical to stock vllm-ascend.

Microbatching: the plugin mirrors the official vllm core gate and keeps the
two-stage path off under ANY microbatching (`use_ubatching`, i.e. DBO or
`ubatch_size > 1`).  On the current vllm-ascend the platform layer resets both
`enable_dbo` and `ubatch_size`, so this gate is dormant there by construction.

Verified evidence (Qwen2.5-Coder-14B-Instruct, 910B2, CANN 9.0.1):

- eager off/on/on_fp32 token-identical; graph off/on/on_fp32 token-identical;
- C3-shaped run (B=64, shared ~4.1k tokens, gen128): 0/64 divergent arms;
- graph cascade at that shape: `VLLM_ASCEND_CASCADE_UPDATE_SKIP_STABLE` on/off
  token-identical (64/64), wall 8.51s vs 9.38s (off baseline 6.49s);
- no fail-open events during the verified runs;
- `pytest -q` + `ruff check .` green.

## Extension framework

Extension ID: `org.vllm-hust.split-batch-full-graph`

The Manifest 0.2 descriptor stays `import_only`: installation alone changes no
vLLM behavior; the cascade capability is env-gated at runtime.

```bash
python -m pip install -e ".[test]"
vllm-hust-ext extension inspect org.vllm-hust.split-batch-full-graph
pytest -q && ruff check .
```
