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
