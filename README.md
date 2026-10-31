# Ascend Split-Batch and Full-Graph Parallel

Owner-led research carrier for split-batch planning, dual-stream replay, and full-graph parallel execution. Core owns graph lifecycle seams; this repository owns policy, runners, validation, and evidence.

**Status:** this repository publishes extension carriers and selected operator implementations; it is not an accepted Split-Batch, dual-stream or full-graph serving optimization. The root split-batch planner is `import_only` and cannot be activated. Cascade implementation and registration have been removed. Current component status and evidence are in [the delivery inventory](docs/delivery-inventory.md).

Technical ownership belongs to @Raing5Days, @ilnnfover. Source extraction must
preserve exact authorship, license, tests, constraints, and evidence before
activation is considered.

See [.github/MAINTAINERS.md](.github/MAINTAINERS.md) and [docs/PROVENANCE.md](docs/PROVENANCE.md).

## Install

```bash
pip install vllm-ascend-split-batch
```

Published wheels are immutable and may differ from this checkout. `0.1.3` predates the removal; `0.2.0` records the published removal state. Release receipts and verification limits are listed in [docs/release.md](docs/release.md#1110-已发布安装资产).

**What you get:** 9 `vllm.general_plugins` carriers and 6 extension bundles.
The root split-batch bundle is planner-only and cannot be activated. There is
no `cascade-attention` entry point, no cascade environment injection, and no
`[kernels]` extra or dependency on the external cascade kernel repository.
Other plugin gates are unchanged; see [docs/support-matrix.md](docs/support-matrix.md).

Upgrade using a wheel built from this checkout, then restart all engine/worker
processes and recapture graphs. Removing registration cannot undo wrappers in an
already running process. Do not reuse an older installed wheel's entry-point
metadata. External kernel packages are not automatically uninstalled. Cascade
implementation and registration are retired; historical reports are not current
support claims.

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
Acceptance evidence: [host-proposals/GDN-qkv-norm/validation/](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git/snapshot/host-proposals/GDN-qkv-norm/validation/).
Registration decision: `host-proposals/GDN-qkv-norm/registration-closure.json`.
Delivery gates: [docs/release.md](docs/release.md#算子-bundle-最短交付路径).

## Capability and evidence status

The published root planner remains `import_only` and cannot be activated. Current per-component discovery, activation, invocation and evidence status is recorded in [docs/delivery-inventory.md](docs/delivery-inventory.md); historical reports remain historical and are not current support claims. Unsupported configurations and invalid speculative requests must remain fail-closed and must not be crossed.

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
[docs/HOST_CONTRACT.md](docs/HOST_CONTRACT.md). Device gates (real host import, real
constructor dispatch, kernel numeric gate, ACL graph replay bit-exactness,
fallback delegation) are closed by `tools/probe_gelu_device.py` and
`tests/test_npu_w31_gelu_plugin.py`; results:
[gelu-device-gate-20261020.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/411e14c5a67dac52b3bba51ac3f25e4c593fcc28/archive/dev-hub-20261022/split-batch-direction/manifest.json)
(frozen custody original; see docs/evidence/README.md). The op-level microbench in that
evidence is directional only (the fused kernel measured slower than the host
native chain at the probed standalone shapes), so default-off remains correct.
Serving-level L3 + compare and light L4 remain pending; historical kernel
evidence does not validate this integration.

### Cohort scope (what our evidence does and does not cover)

> Authority: [`docs/cohort-and-ownership.md`](docs/cohort-and-ownership.md) — the cohort boundary plus the four
> ownership items returned to this repository; binding rule: [`AGENTS.md`](AGENTS.md) hard constraint 6.

Stated here so the numbers below are not over-extended. **2026-10-13 owner ruling: the baseline moved to the
0.28 image + Qwen3.5-35B-A3B** ("we no longer work on Qwen2.5"). The upstream audit quotes in
[`docs/cohort-and-ownership.md`](docs/cohort-and-ownership.md) §1 are kept verbatim as historical receipts;
the table below is the current claim surface. The Qwen2.5 / 0.25 evidence is retained as an **independent,
historical** set and must not be crossed with the numbers below.

| This repository's evidence covers (0.28 baseline) | This repository's evidence does **not** cover |
|---|---|
| 0.28 image stack: vllm `0.28.1.post1.dev143+gf18cf803c` + vllm-ascend `0.25.1rc2.dev125+hust…`, CANN 9.1.0, 2×910B2 TP=2 | other host builds outside the measured 0.28 image |
| Qwen3.5-35B-A3B (40L: 30 GDN + 10 MoE) hybrid **TP2** | native **MTP2** verification rows / any speculative decoding |
| chunked prefill **on**, APC **off**, `cudagraph_mode=NONE` | `async` scheduling combinations we did not run |
| bf16 | the retired Qwen2.5 / 0.25 line (historical set only) |

⇒ **Do not cross-extrapolate**: historical Qwen2.5/0.25 numbers do not carry over to the 0.28/Qwen3.5 baseline, and vice versa. The split-batch planner is not an executable optimization; its current status is `import_only`.


Historical Qwen2.5/0.25 split-batch/Cascade results are archived outside this
repository and are not current support or performance claims. See
[docs/EVIDENCE.md](docs/EVIDENCE.md) for evidence custody and cohort boundaries. They must
not be combined with the current 0.28/Qwen3.5 evidence.


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

Evidence ([docs/evidence/w2b-fi-sampling/REPORT.md](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git/snapshot/docs/evidence/w2b-fi-sampling/REPORT.md), Qwen2.5-Coder-14B, 910B2 —
**historical Qwen2.5/0.25 domain**, see the cohort scope above):

[Original sampling measurements and route observations](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/README.md#L193).
The reported end-to-end difference was within noise. The joint route was not
reachable in the long-context single-card shape. Neither observation is a new
qualification; routing guards and the historical cohort boundary above remain.

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

## 性能验证与证据

CPU 回归使用 `pytest -q -m 'not host_tree'`；仓内保留的算子基准按各自
`tools/bench_*.py` 用法运行。`tools/bench-ladder.sh` 是特定 serving 主机的
L0/L1/L2/L3/L4 流程，依赖模型、解释器、解析工具及仓外结果目录；它不是
clean-checkout CPU 测试，不是安装包依赖，也不能替代缺失的 NPU/服务验证。
性能主张须遵守[交付清单](docs/delivery-inventory.md#性能证据最小字段)：算子与
服务收益分开记录，保留负结果、失败及适用边界。具体开发和发布门见
[发布检查表](docs/release.md)。

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
