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

## Documentation

Normative knowledge (architecture & contracts, coding rules for humans and
agents, release process, pitfall history, progress log) lives in
[docs/](docs/README.md) — start there. Repo-level agent rules:
[AGENTS.md](AGENTS.md).

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
bit-identical to stock vllm-ascend. On load the plugin emits one INFO line,
`cascade plugin loaded (gate=0, graph_gate=0, kernel_wheel=ok)`, so a
default-off serve log proves `load()` actually ran (vLLM otherwise swallows
general-plugin load errors).

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

## fi_sampling plugin (default-off, evidence-only)

Entry point: `vllm.general_plugins` ->
`fi-sampling = vllm_ascend_split_batch.fi_sampling_plugin:load`.

Routes the sampling step to the vendored `fi_sampling` triton-ascend kernels
(flashinfer-semantics port, W2 package) by replacing the fork's
`AscendTopKTopPSampler` with a subclass at plugin-load time. With
`VLLM_HUST_FI_SAMPLING` unset nothing is imported or patched.

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_HUST_FI_SAMPLING` | `0` | master switch |
| `VLLM_HUST_FI_SAMPLING_SEED` | torch initial seed | fixed base seed (per-call advancing) |
| `VLLM_HUST_FI_SAMPLING_JOINT_MIN_BATCH` | `256` | joint-cell batch floor (W2 bench knee) |
| `VLLM_HUST_FI_SAMPLING_JOINT_MIN_K` | `32` | joint-cell top-k floor |
| `VLLM_HUST_FI_SAMPLING_K1_ARGMAX` | `1` | `0` keeps k=1 on the fork chain (tie-exact) |
| `VLLM_HUST_FI_SAMPLING_TRACE` | `0` | per-step route trace + histogram |

Routing: no top-k/top-p -> FI api1 kernel; k=1 on every row -> argmax;
`B >= 256` with `k >= 32` and top-p -> FI joint kernel; everything else (and
all five mandatory fallbacks: per-request generators, processed
logits/logprobs modes, batch-invariant, reduce-sample, async-exponential)
stays on the fork chain with no device sync.

Evidence (`docs/evidence/w2b-fi-sampling/REPORT.md`, Qwen2.5-Coder-14B, 910B2):

- default-off is zero-diff (no host/kernel import, no patch) and NPU smoke is
  6/6 PASS;
- untruncated e2e (4k ctx, B=64, 2 repeats): median TPOT off 112.67 ms vs on
  112.52 ms (**-0.14%**, within noise) with the FI api1 route confirmed on
  1300/1300 steps -- sampling is only ~1% of TPOT at that batch size;
- joint route is unreachable in the 4k-context single-card shape (B_max ~= 122
  < the 256 floor); it only fires with a ~512-token context, so the joint
  branch is a safe, currently dormant branch in normal deployments;
- `pytest -q` 73 passed + `ruff check .` green.

## Extension framework

Extension ID: `org.vllm-hust.split-batch-full-graph`

The Manifest 0.2 descriptor marks the two cascade carriers `active` (the
planner stays `import_only`): `vllm-hust-ext extension enable` injects the
`activation.environment` flags, and installation alone changes no vLLM
behavior because the cascade capability is env-gated at runtime. Verified
host range: `vllm-ascend>=0.23.0rc1,<0.24` (see `docs/release.md` §0.1).

```bash
python -m pip install -e ".[test]"
vllm-hust-ext extension inspect org.vllm-hust.split-batch-full-graph
pytest -q && ruff check .
```

## Kernel wheel dependency (soft, fail-open)

Both cascade tiers consume the `ascend_kernel` CCE op wheel
(`fa_fp32_stage1` / `lse_merge`) as a SOFT dependency: it is declared only in
the `kernels` extra — base dependencies stay empty. When the wheel is absent
or its ops fail to register, the whole cascade feature is disabled
(fail-open to the standard full-KV path) with a single warning; the
default-off semantics are unchanged. Validated pairing:

| plugin | kernel wheel | torch_npu | CANN |
|---|---|---|---|
| `0.1.0.dev0` | `ascend-kernel==2026.3.9` | `2.10.0.post2` | `9.0.1` |

The wheel is built in the kernel repo (`cascade-merge-op`, git), not
published to PyPI:

```bash
pip install ".[kernels]" --find-links /vllm-workspace/cascade-merge-op/ascend-kernel/output
python -c "import ascend_kernel, torch; assert hasattr(torch.ops.npu, 'fa_fp32_stage1')"
```

Compat tuple & red lines for the kernel side live in the kernel repo README
§4-§5. After bumping the kernel wheel: reinstall, then rerun the plugin
smoke (see [docs/release.md](docs/release.md) §2).
