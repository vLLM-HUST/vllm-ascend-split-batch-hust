# Split-batch host contract proposal

The extracted planner is pure and default-off. Runtime activation requires:

1. `vllm.graph.runtime-key.v1`: validated, bounded opaque graph metadata;
2. `vllm.forward.split-context.v1`: per-forward split mode and stream identity;
3. `vllm.ascend.graph-pool.v1`: separately owned main/parallel graph pools;
4. `vllm.worker.split-executor.v1`: execute a committed plan and restore all
   buffers/context on success, error, or cancellation.

The provider must reject speculative decoding, LoRA, MLA, M-RoPE, non-uniform
decode, unsupported graph modes, and unknown capture sizes. It must never lazily
capture an unbounded key set or modify global forward context outside a scoped
context manager.

## Cascade attention components (implemented, default-off)

The cascade plugin (`cascade_plugin.py`, `cascade_graph_plugin.py`,
`cascade_runner_patch.py`) rides ONLY the official public surface:

- `vllm.general_plugins` entry point (`load()`).
- `vllm.v1.cudagraph_dispatcher.add_cudagraph_key` / `dispatch` for graph keys.
- Runner methods `_capture_cudagraphs`, `_warmup_and_capture`,
  `_determine_batch_execution_and_padding`, `_model_forward`,
  `_update_full_graph_params_if_needed` (monkeypatched, env-gated).
- vllm-ascend `update_full_graph_params` / `get_graph_params` /
  `GraphParams` for capture/replay parameter plumbing.
- `ACLGraphWrapper` variant-entry table: the cascade twin graph is stored in a
  per-instance side table keyed by the SAME standard `BatchDescriptor`
  (the official frozen descriptor has no cascade field; the table is swapped
  in for the duration of a cascade capture/replay call).

Default-off semantics: with `VLLM_ASCEND_ENABLE_CASCADE_DECODE` unset the
patched callables delegate to the originals unchanged. `BatchDescriptor` and
all other host dataclasses are never modified.

### Anchor map on the current baseline (verified 2026-09-09)

Host = vllm-hust v1 (`0.28.1.post1.dev143+gf18cf803c`) + vllm-ascend-hust
main (`0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`), CANN 9.1.0.

| Anchor | Old location (0.23.0rc1) | Current location | Status |
| --- | --- | --- | --- |
| `AscendTopKTopPSampler` / `AscendSampler` | `vllm_ascend/sample/sampler.py` | unchanged | OK |
| runner five methods | `vllm_ascend/v1/worker/gpu_model_runner.py` | `NPUModelRunner` (`vllm_ascend/worker/model_runner_v1.py`) inherits `GPUModelRunner` from `vllm/v1/worker/gpu_model_runner.py`, where `_model_forward` / `_determine_batch_execution_and_padding` / `_warmup_and_capture` / `_capture_cudagraphs` are defined | OK (wrappers forward via `*args`) |
| `update_full_graph_params` / `GraphParams` / `get_graph_params` / `ACLGraphWrapper` | `vllm_ascend/compilation/acl_graph.py` | same module (:279 / :306 / :334 / :60) | OK |
| `BatchDescriptor` | `vllm/forward_context.py` | unchanged | OK |
| `spec_decode` / `spec_decode.ngram_proposer` / `eplb.core.policy.policy_factory` | present | present (runtime needs `scipy` + `decorator` installed) | OK |

**Carrier status (2026-09-09)**: the W4 `active` flip was reverted to
`import_only` — the three acceptance evidences were gathered on 0.23.0rc1
and do NOT transfer across a host change. Path back to `active`: re-run the
release.md 三项证据 on this baseline (default-off 零回归冒烟 / 正确性对齐 /
性能对比), then flip and keep the pin. `host.version_range` is pinned to the
exact verified build (`==0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`);
packaging rejects local-version labels in ordered comparators, hence the
point-`==` form.

## fi_sampling component (implemented, default-off)

The fi_sampling plugin (`fi_sampling_plugin.py`, `fi_sampling_route.py`) rides
ONLY:

- `vllm.general_plugins` entry point `fi-sampling` (`load()`).
- vllm-ascend module attribute `vllm_ascend.sample.sampler.AscendTopKTopPSampler`
  replaced by a subclass (`FiSamplingTopKTopPSampler`) whose `forward_native`
  overrides the base; `AscendSampler.__init__` resolves that module attribute at
  construction time, so no other host object is touched. Same variant-entry
  pattern as the fork's own `vllm_ascend/_310p/sample/sampler.py`.
- Read-only probes of `vllm.envs.VLLM_BATCH_INVARIANT` and
  `vllm_ascend.ascend_config.get_ascend_config()` (`enable_reduce_sample`,
  `enable_async_exponential`) for the fallback matrix.

Default-off semantics: with `VLLM_HUST_FI_SAMPLING` unset `load()` returns
before importing `vllm_ascend` or the vendored kernel package, so nothing is
patched and `triton` is never imported. The host source trees are untouched and
no `vllm.platform_plugins` entry is registered. Host-upgrade checklist item
(see `docs/release.md` §4): `AscendTopKTopPSampler` name/`forward_native`
signature and the `AscendSampler` construction site.
