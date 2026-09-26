# Provenance

Source archives:

- [vLLM-HUST core legacy](https://github.com/intellistream/vllm-hust-legacy-20260831)
- [vLLM Ascend HUST legacy](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831)

Primary history:

- [Core #273 MERGED](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/273)
- [Core #263 CLOSED_UNMERGED](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/263)
- [Ascend #280 OPEN 1/4](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831/pull/280)
- [Ascend #281 OPEN 2/4](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831/pull/281)
- [Ascend #282 OPEN 3/4](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831/pull/282)
- [Ascend #283 OPEN 4/4](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831/pull/283)

Original commit patches are preserved under `provenance/legacy-patches/`.

Closed does not mean merged, and open does not mean accepted. These references are migration evidence, not a release receipt. Exact commits, files, authors, licenses, tests, constraints, and benchmark receipts must be recorded before implementation code is accepted.

## Vendored fi_sampling host API (`src/vllm_ascend_split_batch/fi_sampling/`)

- Upstream algorithm anchor: flashinfer `main@3a4e7052`
  (`include/flashinfer/sampling.cuh`, `flashinfer/sampling.py`).
- Source package (worktree, authority for statistical-equivalence evidence):
  `/vllm-workspace/flashinfer-migration/sampling/fi_sampling/`, W2 package
  reviewed and approved 2026-09-09; report
  `flashinfer-migration/sampling/报告-fi_sampling-w2-w3-2026-09-09.md`
  (191 PASS / 0 FAIL; micro-bench V=152064).
- Vendored files and sha256 of the source at vendoring time (2026-09-09):

| vendored file | source sha256 |
|---|---|
| `api.py` | `902884d6cd64c78b8b917f6404f51305a1f23a37760b80a005466b770918a03c` |
| `kernels.py` | `8d0b20cca2559a2c4d893a96bc6c2398569eb2805258848f54ccde114b17d27e` |
| `npu_env.py` | `04a3a122f66a74b7176cfaf7f3b1b701871bfe0e55e252fcf58f8a85b352a77a` |
| `pure.py` | `923d7db26a704e778dfafb4bd74a2742364e6a5b8de250518bc08dfd6954ab08` |

- Vendoring changes: `from fi_sampling.x import ...` -> `from .x import ...`
  (relative imports only); `__init__.py` written for this package. No
  algorithmic change. The vendored files are byte-identical to the source
  apart from those import lines, so `diff` against the source package stays
  meaningful; `src/vllm_ascend_split_batch/fi_sampling` is therefore excluded
  from `ruff` line-length enforcement (the source package lints with ruff's
  default line length, not 88).
- `reference.py` and the test/bench harnesses are NOT vendored; the source
  package remains the authority for simulator/statistical evidence.
- The source package changed once during vendoring (`kernels.py`, 2026-09-09
  04:53, review errata on the `kSCALE` provenance comment); the table above is
  the re-synced state. Re-check the hashes when bumping.
