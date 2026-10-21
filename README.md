# Ascend Split-Batch and Full-Graph Parallel

Owner-led research carrier for split-batch planning, dual-stream replay, and full-graph parallel execution. Core owns graph lifecycle seams; this repository owns policy, runners, validation, and evidence.

**Status: cascade retired from automatic integration.** The split-batch planner
is an inert `import_only` descriptor. The remaining general plugins retain their
existing host bounds, activation gates and registration contracts.

Technical ownership belongs to @Raing5Days, @ilnnfover. Source extraction must
preserve exact authorship, license, tests, constraints, and evidence before
activation is considered.

See [MAINTAINERS.md](MAINTAINERS.md) and [PROVENANCE.md](PROVENANCE.md).

## Install

```bash
pip install vllm-ascend-split-batch
```

The published PyPI `0.1.3` release predates cascade source removal. Installing
that published release does not apply this checkout's deletion. Build and install
this checkout's wheel for the current code; publication requires separate authorization.
Historical release receipt: [docs/release.md](docs/release.md) §11.9.

**What you get:** 9 `vllm.general_plugins` carriers and 6 extension bundles.
The root split-batch bundle is planner-only and cannot be activated. There is
no `cascade-attention` entry point, no cascade environment injection, and no
`[kernels]` extra or dependency on the external cascade kernel repository.
Other plugin gates are unchanged; see [docs/support-matrix.md](docs/support-matrix.md).

Upgrade using a wheel built from this checkout, then restart all engine/worker
processes and recapture graphs. Removing registration cannot undo wrappers in an
already running process. Do not reuse an older installed wheel's entry-point
metadata. External kernel packages are not automatically uninstalled.

Cascade implementation modules and their dedicated tests are removed. Historical
evidence and provenance remain; no cascade installer can be imported from the package.
Rollback: revert the source-removal commit, rebuild/reinstall, and restart/recapture.
This restores source only; the earlier entry-point/dependency retirement remains.
Re-enabling cascade requires restoring those registrations and separate validation.

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

## GDN QKV norm (tier-2, default-off)

The accepted kernel and reversible host proposal live in
`host-proposals/GDN-qkv-norm`. There is no `gdn-qkv-norm` general plugin or
bundle entry point: registration is withdrawn while the target unpack/norm
combination remains inside an opaque core without a proven narrow dispatch.
Do not treat a packaged kernel as plug-and-play callsite replacement.

Scope: Qwen3.5-35B-A3B bf16/ND, TP2, pure non-spec decode T16, FULL graph.
Existing tier-2 activation requires the reviewed host patch plus
`VLLM_HUST_GDN_QKV_NORM=1` and `USE_DEFAULT_FLA_NORM=0` before startup;
unsupported inputs retain native fallback. No automatic patching is provided.
Rollback requires disabling the env and restarting/recapturing; reverse the
host patch only through its separately reviewed procedure.
Acceptance evidence: `host-proposals/GDN-qkv-norm/validation/`.
Registration decision: `host-proposals/GDN-qkv-norm/registration-closure.json`.
Delivery gates: [docs/release.md](docs/release.md#算子-bundle-最短交付路径).

## Removed cascade capability

Cascade eager/graph integration, Runner wrappers and adaptive-gate subprocess
are absent from the package. Neither manual imports nor old cascade environment
variables can restore them. Historical material remains under `docs/evidence/`
and `provenance/`, not as an executable plugin implementation.

Claim boundaries remain in [docs/cohort-and-ownership.md](docs/cohort-and-ownership.md).
The inert planner's speculative-decoding guards are unchanged.

## fi_gelu plugin (default-off, OOT registration)

`fi-gelu` uses `vllm.general_plugins` for loading and
`CustomOp.register_oot(name="GeluAndMul")` for execution selection. Enable with
`VLLM_HUST_FI_GELU=1` before the first activation lookup/model construction;
host factories may cache instances. The original host class is not
modified; its constructor selects the provider subclass for new instances.
The provider retains the host's `gelu_and_mul` compilation configuration name.

Only NPU bf16 inputs with at least two dimensions, positive even last dimension
and `approximate="none"` use the existing exact-erf kernel API. Other inputs and
kernel failures delegate to the host. An occupied OOT key is never overwritten;
unavailable registration or kernel dependencies leave the host path in place.
Unset the environment variable and restart workers/recapture graphs to disable.
`fi_gelu_plugin.uninstall()` removes only its own registration for future
instances; it does not change existing instances or captured graphs.

Registration checks and remaining-plugin migration boundaries are in
[HOST_CONTRACT.md](HOST_CONTRACT.md). Device gates (real host import, real
constructor dispatch, kernel numeric gate, ACL graph replay bit-exactness,
fallback delegation) are closed by `tools/probe_gelu_device.py` and
`tests/test_npu_w31_gelu_plugin.py`; results:
`docs/evidence/gelu-device-gate-20261020.json`. The op-level microbench in that
evidence is directional only (the fused kernel measured slower than the host
native chain at the probed standalone shapes), so default-off remains correct.
Serving-level L3 + compare and light L4 remain pending; historical kernel
evidence does not validate this integration.

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

Evidence (`docs/evidence/w2b-fi-sampling/REPORT.md`, Qwen2.5-Coder-14B, 910B2 —
**historical Qwen2.5/0.25 domain**, see the cohort scope above):

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

The root descriptor contains only the `import_only` planner and no activation
environment. Other bundles retain their own activation status and host bounds.
The bounded host range remains `>=0.25.1rc2.dev125,<0.25.2`; see
[docs/release.md](docs/release.md) for its verified scope.

```bash
python -m pip install -e ".[test]"
vllm-hust-ext extension inspect org.vllm-hust.split-batch-full-graph
pytest -q tests/test_planner.py
ruff check src/vllm_ascend_split_batch/planner.py tests/test_planner.py
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

## Operator validation workflow

The repository owns the executable ladder at `tools/bench-ladder.sh`; `opt bench`
dispatches to this checked-in source rather than mutating archived evidence scripts.
Select checks by change type using [the development contract](docs/agent-guide.md#2-开发工作流).
Documentation/results/pointers require format, reference, hash and Git-scope checks,
not model smoke or full pytest. Development runs affected CPU tests and static
checks; full pytest needs an explicit shared-impact, dependency/packaging,
release/merge or insufficient-targeted-coverage reason. Share full-suite results
for the same relevant source snapshot and environment across parallel sessions;
evidence, pointer, commit-hash or wording updates alone do not invalidate them.
Release admission remains unchanged.

For operator/weight/execution-order changes:

1. affected static checks and CPU tests, then necessary L0;
2. plain L3 + `opt bench compare`; numerical failure blocks promotion independently;
3. once `ready`/`validating`, bounded light L4 through the registered operator gate
   (`--candidate <ID> --trace-tier light --collection-reason candidate-ab
   --gap <question> --reuse-evidence <path>`); see `/root/opt-ops/AGENTS.md`;
4. deep only with `--collection-reason mechanism-gap` for an explicit mechanism gap;
5. parse offline and slim after delivery; never recollect historical data for cadence.

L1/L2 screen is explicit for an independent end-to-end question, not an operator
prerequisite. Graph/system collection retains the complete plain screen gate
(`L1×3 + L2×3 + L3×1`, then `--screen-report <screen.json>`); never relabel a route
to bypass it. Service smoke and package-loading smoke each need an independent
question; without packaging/registration/loading changes, do not repeat loading
smoke. Flow-only repairs do not rerun model L3 unless loading, execution branches,
input/output protocol or comparison semantics change. `opt bench run` is an
explicit end-to-end convenience command, not a per-commit gate.

`light` defaults to the host's `max_iterations=32` bounded window, disables stack/memory/shape collection,
stamps the capture scope into the opstat, and removes profiler databases after the
CSV/all-kernel result is delivered. `deep` defaults to a full window and retains
the shape database. Plain and profiler-enabled servers cannot be reused across
modes; stop the active server before switching.

`opt bench serve` requires its captured startup PID to remain live and match the
PID file before accepting `/health` and publishing the serve mode. A dead or
replaced PID fails immediately, even if another server is healthy. Standalone
`opt bench wait [timeout_s]` remains an endpoint-only readiness check.

### Carrier status is recorded, not gated

`screen.json` carries a `carriers` block (`opt-ops-carriers.v2`), and
`opt bench serve` prints the same summary right after `/health` is ready
(`tools/bench-ladder.sh carriers [serve_log]` prints it as one JSON line). It records the
three things that actually decide whether the plugin carrier is live:

1. `find_spec("vllm_ascend_split_batch")` under the **serving interpreter**
   (`/usr/local/python3.12.13/bin/python` on this host; `/usr/bin/python3` cannot see
   the package and will report a false absence);
2. the husk entry points enumerable from the `vllm.general_plugins` group
   (name + value; an install that failed to register is equivalent to no carrier);
3. functional markers counted from that run's serve log.

Plugin switches are read from the **serve side only**: `cmd_serve` writes the exported
`VLLM_HUST_*` values to `<serve log>.env.json` (`opt-ops-serve-env.v1`, atomic write, never
overwritten) before the server starts, and the probe reads them through `env_source` /
`env_scope` (`serve-process`). The shell that later runs `opt bench screen` does **not**
necessarily inherit those exports, so it is never used as the source. If the snapshot is
missing or does not match the serve log, every value is `null` with
`env_scope`/`env_source` = `unknown` and a `probe_errors` entry — an unset switch is never
reported as `""` or as `0`.

Attribution is **split per carrier** under `attribution` and is never merged into one boolean:

| field | states | meaning |
|---|---|---|
| `husk.cascade` | `effective` / `present-not-observed` / `absent` / `unknown` | `cascade plugin loaded` seen per process |
| `husk.gate_fp32` | same | `Gate fp32 cache populated` seen per worker |
| `husk.gdn_merge_inproj` | `present-but-fail-closed` / `present-not-observed` / `absent` / `unknown` | the in-repo carrier refuses to merge (no safe public seam), model stays stock |
| `runtime_tree.w3f2` | `applied` / `not-applied` / `unknown` | the deployment-tree patch merged `in_proj` (`W3F2: GDN in_proj merge applied`) |
| `runtime_tree.w3f3` | `enabled` / `not-enabled` / `failed` / `unknown` | shared scalar-gate pack patch (`W3F3: …`) |

`husk.gdn_merge_inproj = present-but-fail-closed` and `runtime_tree.w3f2 = applied` routinely
coexist: the fail-closed path is the in-repo plugin declining a merge it cannot consume safely
(see [HOST_CONTRACT.md](HOST_CONTRACT.md), GDN in_proj merge carrier), while W3F2 is a separate
runtime-tree patch. **Never** attribute the W3F2 merge to this plugin, and never treat the
fail-closed warning as a failure — it asserts the opposite (no double merge).

`Unknown vLLM environment variable detected: VLLM_HUST_*` is **not** an absence signal —
it shows up whether or not the carrier is installed, so it is recorded separately under
`unreliable_absence_signals` with `is_absence_signal: false` (see
[docs/pitfalls.md](docs/pitfalls.md) §1.5). Carrier state is a **recording axis**: the probe is
fail-closed (absence, an entry-point enumeration error, missing log, or a missing env snapshot
records `unknown`/`null`/`false`) and never changes `screen`/`serve` exit codes; only
cross-window absolute comparisons require two windows to share a carrier state.

## CommTrace stop safety

`tools/collect_comm_mechanism.sh` requires an isolated CPU-only native profiler
stop/finalize preflight before starting vLLM. A crash, timeout, or incomplete
preflight exits with code 3 and retains its evidence; shapes, stack, modules,
FULL graph, and `LADDER_SLIM=0` must not be disabled to pass it.

The optional point-pinned torch_npu source repair and apply/rollback commands are
in [`host-proposals/comm-trace-stop-gil/proposal_meta.json`](host-proposals/comm-trace-stop-gil/proposal_meta.json).
Source patching and the isolated native clear regression are not proof that the
installed library is repaired. Preflight must use the same Python interpreter
as `tools/bench-ladder.sh` serve, not the offline parsing environment. The selected
runtime library must pass
`/usr/local/python3.12.13/bin/python tools/check_comm_profiler_stop.py --output <new-evidence-directory>`;
a CPU preflight pass does not accept dual-rank pairing, callsites, graph device
boundaries, numerical correctness, or performance.

## Canonical MOD metadata

Repository identity, directly responsible maintainers, advisor status, default-off
activation, rollback, scope, and evidence qualification are recorded in
[`MOD_METADATA.json`](MOD_METADATA.json). `advisor_status: unknown` is not the
same as confirmed `none`. Performance statements remain limited to the workloads
and evidence labels recorded there; they are not general online claims.
