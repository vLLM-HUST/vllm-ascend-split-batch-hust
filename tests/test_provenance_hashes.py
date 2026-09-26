# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PROVENANCE 的机械守卫（CPU-only，无需设备栈）。

`PROVENANCE.md` 与 `fi_sampling/ATTRIBUTION.md` 里记的 sha256 是溯源收据；只要有人
在原地改一个字节，收据就失效而**不会有人发现**。本文件把三件事变成可执行判据：

1. 四个 vendored 文件的 sha256 与 `ATTRIBUTION.md` 的表一致（改字节 ⇒ 红）；
2. 除这四个 vendored 文件外，`src/**/*.py` 都有许可声明（新文件漏头 ⇒ 红）；
3. 归档 patch 的 commit sha 仍在 `PROVENANCE.md` 里被引用（搬运时漏记 ⇒ 红）。

第 3 条只做**文本级**核对（不联网）：`PROVENANCE.md` 必须至少列出全部 14 个归档 sha
的前 12 位。真实性与上游状态（PR merged/open、作者）由
`docs/provenance-verification.md` 的核对记录负责，需联网时人工重跑那里的命令。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG = REPO_ROOT / "src" / "vllm_ascend_split_batch"
FIS = PKG / "fi_sampling"

#: 与 `fi_sampling/ATTRIBUTION.md` / `PROVENANCE.md` 的表一致。改 vendored 字节时
#: 必须同时更新这两处（收据刷新），否则本测试红 —— 这是设计意图。
VENDORED_SHA256 = {
    "api.py": "ca8dc05cfc8906d7ef7992aa6f872e732277c89432b58a6471e6ffd647dc7446",
    "kernels.py": "8d0b20cca2559a2c4d893a96bc6c2398569eb2805258848f54ccde114b17d27e",
    "npu_env.py": "04a3a122f66a74b7176cfaf7f3b1b701871bfe0e55e252fcf58f8a85b352a77a",
    "pure.py": "923d7db26a704e778dfafb4bd74a2742364e6a5b8de250518bc08dfd6954ab08",
}

#: `api.py` 的**源侧** sha256（W2 快照）。它是四个文件里唯一被动过导入行的，所以不能直接
#: 比对源文件；把 vendored 的两条相对导入还原成绝对导入后必须命中此值 —— 该命中同时证明
#: "记录值为真" 与 "搬运只改了那两行"（PROVENANCE.md §4.1）。
API_PY_SOURCE_SHA256 = (
    "902884d6cd64c78b8b917f6404f51305a1f23a37760b80a005466b770918a03c"
)
#: vendoring 时的导入改写（源码 → vendored 的相对导入；逆变换即据此还原）。
VENDORED_IMPORT_REWRITES = (
    ("from fi_sampling.kernels import", "from .kernels import"),
    ("from fi_sampling.npu_env import", "from .npu_env import"),
)

LICENSE_RE = re.compile(r"Apache License|SPDX-License-Identifier:\s*Apache-2\.0")


def test_vendored_files_match_the_recorded_hashes() -> None:
    for name, expected in VENDORED_SHA256.items():
        path = FIS / name
        assert path.is_file(), f"vendored file missing: {name}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, (
            f"{name} changed: recorded {expected}, actual {actual}. "
            "If the vendored copy was intentionally re-synced, update "
            "PROVENANCE.md and fi_sampling/ATTRIBUTION.md in the same commit."
        )


def test_api_py_source_hash_is_recoverable() -> None:
    """`api.py` 的源侧 hash 用逆变换复核（源包已漂移，不能直接比对）。

    搬运只改了导入行，所以把相对导入还原成绝对导入后，字节必须等于 W2 源快照，
    其 sha256 必须命中 `PROVENANCE.md` §4.1 记录的值。命中即同时证明：记录为真、
    且搬运没有夹带其它编辑（例如把 W3 代码带进来）。
    """
    text = (FIS / "api.py").read_text(encoding="utf-8")
    for source_form, vendored_form in VENDORED_IMPORT_REWRITES:
        assert vendored_form in text, (
            f"expected the vendored relative import {vendored_form!r} in api.py; "
            "if the vendoring transform changed, update VENDORED_IMPORT_REWRITES "
            "and PROVENANCE.md §4.1 together"
        )
        text = text.replace(vendored_form, source_form)
    actual = hashlib.sha256(text.encode()).hexdigest()
    assert actual == API_PY_SOURCE_SHA256, (
        "api.py no longer reproduces the recorded W2 source hash "
        f"(expected {API_PY_SOURCE_SHA256}, got {actual}). Either the vendored "
        "bytes drifted, or the recorded source hash is wrong."
    )


def test_every_python_module_declares_a_license() -> None:
    missing = []
    for path in sorted(PKG.rglob("*.py")):
        if path.parent == FIS:
            # Vendored bytes are pinned by hash instead; see the note in
            # fi_sampling/ATTRIBUTION.md ("Why there is no license header").
            continue
        if not LICENSE_RE.search(path.read_text(encoding="utf-8")[:600]):
            missing.append(str(path.relative_to(REPO_ROOT)))
    assert not missing, f"module without a license declaration: {missing}"


def test_provenance_lists_every_archived_commit() -> None:
    provenance = (REPO_ROOT / "PROVENANCE.md").read_text(encoding="utf-8")
    patches = sorted((REPO_ROOT / "provenance" / "legacy-patches").rglob("*.patch"))
    assert patches, "legacy patch archive is empty"
    missing = []
    for patch in patches:
        head = patch.read_text(encoding="utf-8", errors="replace")[:400]
        sha = re.search(r"^From ([0-9a-f]{40}) ", head, re.M).group(1)
        if sha[:12] not in provenance:
            missing.append(f"{patch.name} -> {sha[:12]}")
    assert not missing, (
        "archived commit not referenced in PROVENANCE.md "
        f"(add it with its verification state): {missing}"
    )


def test_vendored_attribution_notes_the_source_drift() -> None:
    """源包已漂移这件事必须在文档里写明，不能沉默；且核对方法要写清。"""
    text = (FIS / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert "Revision anchor caveat" in text
    provenance = (REPO_ROOT / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "逆变换复核命中" in provenance, (
        "PROVENANCE.md must record that api.py's source hash is verified by the "
        "inverse import transform (not by a plain sha256 of the live source)"
    )
    assert "订正记录" in provenance, (
        "PROVENANCE.md must keep the correction trail for the api.py verdict"
    )
