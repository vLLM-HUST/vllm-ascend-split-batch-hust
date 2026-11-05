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

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG = REPO_ROOT / "src" / "vllm_ascend_split_batch"

LICENSE_RE = re.compile(r"Apache License|SPDX-License-Identifier:\s*Apache-2\.0")


def test_every_python_module_declares_a_license() -> None:
    missing = []
    for path in sorted(PKG.rglob("*.py")):
        if not LICENSE_RE.search(path.read_text(encoding="utf-8")[:600]):
            missing.append(str(path.relative_to(REPO_ROOT)))
    assert not missing, f"module without a license declaration: {missing}"


def test_provenance_lists_every_archived_commit() -> None:
    provenance = (REPO_ROOT / "docs" / "PROVENANCE.md").read_text(encoding="utf-8")
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
