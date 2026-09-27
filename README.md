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

## Install

```bash
pip install vllm-ascend-split-batch
```

Published on PyPI (`0.1.3`; wheel + sdist, sha256 and PEP 740 attestations on the
project page). Release receipt: [docs/release.md](docs/release.md) §11.9. This version widens
`host.version_range` from a point pin to the bounded window
`>=0.25.1rc2.dev125,<0.25.2` (rationale §0.3) and carries the UpdatableGraph replay fix;
it was verified on the published bytes, not only on the work tree.

**What you get:** 8 `vllm.general_plugins` carriers and 4 extension bundles,
all **default-off** — installing changes nothing until you enable a capability
via its env var (see the table below). Cascade attention additionally needs the
`ascend_kernel` CCE op wheel, which is **not on any package index** (the
`kernels` extra pins a version that only exists in the kernel repository, so the
extra resolves only with `--find-links`):

```bash
# cascade-only soft dependency; skip it if you do not need the cascade path
pip install "vllm-ascend-split-batch[kernels]" \
  --find-links https://github.com/Raing5Days/vllm-hust-cascade-kernel/releases/expanded_assets/v2026.9.26
```

⚠️ **Host pin.** The manifest pins one verified `vllm-ascend` build
(`host.version_range`; see [docs/release.md](docs/release.md) §0.1). On any other
build the manager reports the extension as `incompatible` and
`vllm-hust-ext run` refuses to launch it — `extension enable` still works, and
the env-var route below is unaffected. Widening the pin requires re-running the
host-upgrade checklist on the target build (§4).

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
`cascade plugin loaded (gate=0, graph_gate=0, kernel_wheel=not-probed)`, so a
default-off serve log proves `load()` actually ran (vLLM otherwise swallows
general-plugin load errors). **Disabled discovery touches nothing else**: with
`VLLM_ASCEND_ENABLE_CASCADE_DECODE` unset the plugin injects no env entries,
imports no host module, installs no fail-open shim, imports no kernel wheel and
replaces no attention/graph entry (contract: `docs/release.md` §6).

## Speculative decoding boundary

Cascade admits **one query row per request**. The two-stage path flattens the
shared prefix once and re-derives the per-request suffix from the query-row
count, so MTP verification and chunked prefill (which feed `k + 1` rows) are
rejected by the dispatch gate and skip the cascade twin capture; those steps
keep native attention and the standard FULL graph. `speculative_config` being
set at all is treated as fail-closed, even when a step happens to carry single
rows, because the `k > 1` query layout is neither supported nor measured yet.
This is a guard, **not** MTP support: Split-Batch/dual-pad fails closed the
same way (`planner.precheck_reason` -> `speculative_decode_conflict`). Enabling
Cascade with speculative decoding therefore yields no cascade speedup — it is
a compatibility boundary, not an enabled-Cascade result.


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

The full evidence set travels with this repository: see
[docs/evidence/cascade/README.md](docs/evidence/cascade/README.md) for the
per-section reports (`section2-correctness.md`, `section3-performance.md`,
`section4-active-enablement.md`), the reproduction scripts, the raw result
JSONs and the sha256 manifest (including the list of large sweep logs left in
the development workspace). Read its §1 before quoting any number: the Coder
model is a **performance stand-in** and must not be compared with the real
`Qwen2.5-14B-Instruct` on behavior.

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
behavior because the cascade capability is env-gated at runtime. Declared
host range: **`vllm-ascend >=0.25.1rc2.dev125,<0.25.2`** — a bounded
*compatibility window*, not a point pin: both endpoints were verified on
910B2 hardware (`dev125` historically; `dev605` = `fbe4911bb` on 2026-09-27),
and a failure inside the window is a host-side finding by design (rationale
and the core-pairing caveat: `docs/release.md` §0.1/§0.3).

```bash
python -m pip install -e ".[test]"
vllm-hust-ext extension inspect org.vllm-hust.split-batch-full-graph
pytest -q && ruff check .
```

⚠️ **`vllm-hust-ext` is not on PyPI** (see the org's own note on
[vllm-hust-website](https://github.com/vLLM-HUST/vllm-hust-website): *"No public
vllm-hust-ext PyPI alpha exists yet; install the current source"*). It is the
org's extension-manager package, maintained in
[`vLLM-HUST/extension-manager`](https://github.com/vLLM-HUST/extension-manager) —
this repository only consumes it.

- **Installing this repository's extras never needs it.** Since 2026-09-26 the
  `test` extra no longer pins it, so `pip install "vllm-ascend-split-batch[test]"`
  resolves on its own. (Released 0.1.0/0.1.1 metadata still carries the pin —
  `pip install ...==0.1.1[test]` fails with `Could not find a version that
  satisfies the requirement vllm-hust-ext==0.2.0.dev0`; use the git step below
  for those versions. PyPI files are immutable, hence the next release.)
- **Running the manifest tests does need it** (`tests/test_manifest.py` imports
  `vllm_hust_ext.manifest`). Install the manager from git first, which is exactly
  what CI and `publish.yml` do:

```bash
python -m pip install "vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git@main"
```

The `kernels` extra is the one end users need, and it is
resolvable (`--find-links` to the kernel release, see "Install" above).

## Kernel wheel dependency (soft, fail-open)

Both cascade tiers consume the `ascend_kernel` CCE op wheel
(`fa_fp32_stage1` / `lse_merge`) as a SOFT dependency: it is declared only in
the `kernels` extra — base dependencies stay empty. When the wheel is absent
or its ops fail to register, the whole cascade feature is disabled
(fail-open to the standard full-KV path) with a single warning; the
default-off semantics are unchanged. Validated pairing:

| plugin | kernel wheel | torch_npu | CANN |
|---|---|---|---|
| `0.1.3` | `ascend-kernel==2026.9.26` | `2.13.0rc1` | `9.1.0` |
| `0.1.2` | `ascend-kernel==2026.9.26` | `2.13.0rc1` | `9.1.0` |
| `0.1.1` | `ascend-kernel==2026.9.26` | `2.13.0rc1` | `9.1.0` |

The kernel wheel is not on any package index (it is platform-tagged and must be
rebuilt whenever the CANN / torch_npu / soc tuple changes), so it ships as a
**GitHub Release asset**:

```bash
pip install "vllm-ascend-split-batch[kernels]" \
  --find-links https://github.com/Raing5Days/vllm-hust-cascade-kernel/releases/expanded_assets/v2026.9.26
python -c "import ascend_kernel, torch; assert hasattr(torch.ops.npu, 'fa_fp32_stage1')"
```

Verify the installed kernel build by its library fingerprint (the kernel repo's
rule is "read the md5, not the file name"): `libascend_kernel.so` must hash to
`ca8de2d70fe0504d` for `2026.9.26`. Building it from source is documented in the
kernel repo README (`./build.sh`, ~2 min).

Compat tuple & red lines for the kernel side live in the kernel repo README
§4-§5. After bumping the kernel wheel: reinstall, then rerun the plugin
smoke (see [docs/release.md](docs/release.md) §2).
