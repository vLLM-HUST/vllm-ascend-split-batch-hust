# Historical evidence custody

Raw measurements, reports, failed runs and their original context are now held
by [vllm-hust-benchmark](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git).
The exact source and per-file digests are pinned in [EVIDENCE_SOURCES.json](../EVIDENCE_SOURCES.json).
This is a storage/reference migration, not new performance or correctness
qualification. Historical eager/graph cohorts and negative results remain
separate. Runtime code, default-off behavior and host integration are unchanged.

## Retrieve originals

Download the frozen source tree to a new directory outside this checkout:

```bash
python tools/benchmark_evidence.py --all --destination /tmp/split-batch-evidence
```

The downloader checks the immutable manifest and every file's length/SHA256,
rejects unsafe paths, and refuses to overwrite a corrupt cache. Empty/invalid
historical JSON is preserved, not repaired. It never executes downloaded code.
Use a separate external tree when investigating historical scripts: their
original relative paths refer to the full archived tree, not this trimmed MOD.

The remaining source scripts in historical evidence directories are read-only
provenance, not local result stores. Their exact hashes are covered by the
custody test; the frozen fi-sampling source/hash test remains unchanged.

## Future run output

The active `bench/bench_*.py` entry points require an absolute external output
directory, including when launched through the shell wrappers:

```bash
export SPLIT_BATCH_RESULTS_DIR=/tmp/split-batch-new-run
```

Do not reuse historical output directories or commit generated results here.
This setting does not authorize a device run, alter cache policy, or remove any
existing experiment approval requirement. Publish reviewed raw evidence to
benchmark and link it from the MOD. CI verifies archive retrieval and rejects
new tracked data in the cleaned directories; the trusted organization policy
continues to check pull requests.

## Installation assets are retained

Historical source-package evidence is archived under
[split-history-release-*](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022).
Published distributions are not rewritten: the existing Release assets match
PyPI and remain referenced by [release instructions](release.md).
Archiving their evidence is not proof that their installation consumers have
disappeared. The sdist allowlist and distribution verifier prevent evidence from
entering future packages. No published installation asset is deleted here.

## Mixed reports and local-only probes

Historical prose and report-only JSON now link to immutable benchmark originals.
See [EVIDENCE_AVAILABILITY.json](../EVIDENCE_AVAILABILITY.json) for scope and the
central availability audit. Six reference occurrences resolve to five archived
originals: two aliases by declared SHA256 and four references by original-path
provenance mappings plus SHA256. CI downloads and checks the recovery mappings
and original bytes. The other 33 occurrences remain unrecovered; null
references and archived summaries must not be reported as successful raw checks.
Failures, rejected methods, registration withdrawal and cohort boundaries are
retained. Numeric routing/shape constants and installation identities are not
measurements and remain in their implementation contracts.

## Direction branch batch

分支 `0.28/comm-overlap-free-attribution` 的窗口证据与交付语料（365 + 66 + 18 件（三批）：
算子 A/B 与 L0 地板 JSON、L3 捕获、opstat 副本、窗口回执、hazard 源码 vintage、
host-proposals 提案目录（patches/orig/OWNERSHIP）、撤回套组含配对取证
(`host-proposals/GDN-qkv-norm/validation/l4-multibuffer-20261019`)、
通信机制和 OPT-truth 的历史采集）已按 custody 政策冻结于：

- 批次分支: `direction-evidence-20261022` @ commit `02add2a27be84b106dd5e8c589b0a5bc3f928bff`
- 快照: `archive/dev-hub-20261022/split-batch-direction/snapshot/`（镜像原始相对路径）
- 内层 manifest: `archive/dev-hub-20261022/split-batch-direction/manifest.json`
  （schema mod-git-custody/v1，449 项，逐文件 bytes/sha256/git_blob，
  source_commit `03b0fa51627b6f0d580abc991421efdc5070439b`）
- 原始取件 URL 形态:
  `https://raw.githubusercontent.com/vLLM-HUST/vllm-hust-benchmark/02add2a27be84b106dd5e8c589b0a5bc3f928bff/archive/dev-hub-20261022/split-batch-direction/snapshot/<原始路径>`

登记约定（C 政策）：活跃窗口证据留方向分支；**窗口收尾批量追加到归档分支并在本文件登记**；
合入 main 前清零。GDN-qkv-norm 撤回套组的恒量（配对 forensics sha256
`13212eccec831dbb7ba8c0cc2dafb43967118a890f5b02f01e35b6e534f06ce3`；rank_pair
all_comm 2624 / allreduce 2592）由 tests/test_attribute_comm_free.py 内嵌守护。
