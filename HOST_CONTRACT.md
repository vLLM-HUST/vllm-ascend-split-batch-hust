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

### Anchor map on the current baseline (parameter-level, corrected 2026-09-09)

Host = vllm-hust v1 (`0.28.1.post1.dev143+gf18cf803c`) + vllm-ascend-hust
main (`0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`), CANN 9.1.0.

> **Correction (2026-09-09, post-review).** The first version of this table
> checked symbol *existence* only and asserted "wrappers forward via `*args`".
> That assertion was **false**, and it hid four blocking signature drifts
> (D1–D4 below) — exactly the failure mode `docs/pitfalls.md` §2.1 warns about.
> Existence is not compatibility: a monkeypatched method must match the host's
> *parameter list and call convention*. Use the AST audit
> (`knowledge/evidence/cascade/logs/sig_audit2.py` semantics) before
> any `host.version_range` change.

| Anchor | Host signature (current) | Plugin wrapper | Status |
| --- | --- | --- | --- |
| `_capture_cudagraphs` | `(self, batch_descriptors, cudagraph_runtime_mode, profiler=None)` | was `(self, batch_descriptors, cudagraph_runtime_mode)` | **D1 drift → fixed** |
| `_update_full_graph_params_if_needed` | `(self, forward_context, num_tokens_padded)` | was called with 3 positional (`+positions`) | **D2 drift → fixed** |
| `impl_cls.update_graph_params` | `(update_stream, forward_context, num_tokens, vllm_config, speculative_config=None, draft_attn_metadatas=None)` | had extra `num_dcp_pcp_tokens` and forwarded 7 positional | **D3 drift → fixed** |
| `_model_forward` | `(self, input_ids=None, positions=None, intermediate_tensors=None, inputs_embeds=None, **model_kwargs)`; host calls **keyword-only** | had required leading `num_tokens_padded` | **D4 drift → fixed** |
| `_determine_batch_execution_and_padding` | 12 params (see host `gpu_model_runner.py:4040`) | identical | OK |
| `_warmup_and_capture` | `(self, desc, cudagraph_runtime_mode, profile_seq_lens=None, allow_microbatching=False, num_warmups=None, profiler=None)` | not wrapped (called via `orig`) | OK |
| `AscendTopKTopPSampler` / `AscendSampler` | `vllm_ascend/sample/sampler.py` | subclass replacement | OK |
| `update_full_graph_params` / `GraphParams` / `get_graph_params` / `ACLGraphWrapper` | `vllm_ascend/compilation/acl_graph.py` (:279/:306/:334/:60) | read-only use | OK |
| `BatchDescriptor` | `vllm/forward_context.py` | unchanged | OK |
| `spec_decode` / `spec_decode.ngram_proposer` / `eplb.core.policy.policy_factory` | present (needs `scipy` + `decorator`) | shims | OK |

**Design rule this establishes**: a wrapper that raises out of a patched host
method kills the engine, which violates the documented fail-open contract. All
cascade wrappers must (a) match the host call convention and (b) delegate to the
original on any internal failure with a single warning.

**Carrier status (2026-09-09)**: carriers remain `import_only`. The three
acceptance evidences were gathered on 0.23.0rc1 and do NOT transfer across a
host change; the new-baseline re-run
(`knowledge/evidence/cascade/EVIDENCE-V1-BASELINE.md`) **failed
evidence #2 and partially failed #3** because of D1–D4. Path back to `active`:
fix D1–D4 (done in a follow-up commit), then re-run release.md 三项证据 on this
baseline and keep the pin. `host.version_range` stays at the exact verified
build (`==0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`); packaging rejects
local-version labels in ordered comparators, hence the point-`==` form.

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
