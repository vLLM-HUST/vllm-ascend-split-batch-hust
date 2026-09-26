# Attribution — vendored `fi_sampling` host API

This package is a **verbatim copy** of four host modules from the W2 `fi_sampling`
package (a flashinfer-semantics port to triton-ascend written in this project's
development workspace). It is vendored so the plugin wheel is self-contained.

## What is vendored

| file here | source file | vendored sha256 |
|---|---|---|
| `api.py` | `fi_sampling/api.py` | `ca8dc05cfc8906d7ef7992aa6f872e732277c89432b58a6471e6ffd647dc7446` |
| `kernels.py` | `fi_sampling/kernels.py` | `8d0b20cca2559a2c4d893a96bc6c2398569eb2805258848f54ccde114b17d27e` |
| `npu_env.py` | `fi_sampling/npu_env.py` | `04a3a122f66a74b7176cfaf7f3b1b701871bfe0e55e252fcf58f8a85b352a77a` |
| `pure.py` | `fi_sampling/pure.py` | `923d7db26a704e778dfafb4bd74a2742364e6a5b8de250518bc08dfd6954ab08` |

`__init__.py` is **not** vendored — it was written for this package (it carries the
project's license header) and only re-exports the four entry points with relative
imports.

## License and origin

- Algorithmic origin: **flashinfer** `main@3a4e7052`
  (`include/flashinfer/sampling.cuh`, `flashinfer/sampling.py`), Apache-2.0,
  Copyright (c) 2023-2025 by the FlashInfer team.
- W2 package (immediate source of the vendored bytes): authored in this project's
  development workspace, W2 review approved 2026-09-09; report
  `报告-fi_sampling-w2-w3-2026-09-09.md` (191 PASS / 0 FAIL; micro-bench V=152064).
- This repository as a whole is distributed under Apache-2.0; see the root
  [`LICENSE`](../../../LICENSE).

## Why there is no license header inside the vendored files

The four vendored files carry **no in-file copyright header**, because the W2 source
files carry none either, and byte-identity with the reviewed snapshot is a hard
requirement (`PROVENANCE.md`): the hashes above are the provenance receipt, and an
in-file header would invalidate them. Attribution therefore lives in this file and in
[`PROVENANCE.md`](../../../PROVENANCE.md) instead of in the copied bytes.

> If the W2 source later gains headers, re-sync the copy and update the hashes rather
> than editing the vendored files in place.

## Revision anchor caveat (2026-09-26)

The W2 source package is a **live worktree** (`knowledge/surveys/sampling/fi_sampling`
in the development workspace) and has since gained the W3 renorm/mask family, so a
`diff` against today's source no longer isolates the import rewrite. The vendored
bytes correspond to the **W2 snapshot**, and `PROVENANCE.md` §4.1 records both the
vendored hashes and the source-side hashes.

Re-verification for `api.py` does **not** need that snapshot: it is the only vendored
file whose import lines changed, and the change is known and reversible — rewriting
its two relative imports back to absolute ones reproduces the recorded source-side
sha256 exactly (2026-09-26 measurement; enforced by
`tests/test_provenance_hashes.py::test_api_py_source_hash_is_recoverable`).
