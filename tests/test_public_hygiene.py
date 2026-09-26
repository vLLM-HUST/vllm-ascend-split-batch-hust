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

"""发布前卫生检查：仓库文本里不得出现内网地址或凭据样式串。

为什么要有这个测试：本仓是**公开**仓库，并已发布到 PyPI（发行物永久不可覆盖）。
证据目录里存的是真实运行日志，2026-09-26 首发前扫描时**真的**在
`docs/evidence/w2b-fi-sampling/logs/` 找到 10 处未遮蔽的 `192.168.0.5`
（torch distributed init 打印）—— cascade 目录当时已遮蔽，这批早先提交的漏了。
靠人眼复核一次不够，所以把判据固定下来。

作用域与豁免：

- 只看 `git ls-files` 里的文本文件（`.git` 与二进制跳过）；
- `192.168.x.x` 这类**占位符**不匹配（正则要求四段都是数字）；
- 历史 patch 存档（`provenance/legacy-patches/`）里的上游 diff 若含内网地址，
  属上游原文，用显式豁免清单放行（当前为空——存档里没有命中）。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: RFC1918 私网地址（四段全为数字，因此 `192.168.x.x` 占位符不会被匹配）。
PRIVATE_ADDRESS_RE = re.compile(
    r"\b(?:192\.168|10)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
    r"|\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b"
)

#: 凭据样式串（只认高置信度形态，避免把文档里的说明文字当命中）。
SECRET_PATTERNS = (
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{22,}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"pypi-[A-Za-z0-9_-]{50,}"),
)

#: 显式豁免：上游存档原文里若含私网地址，记在这里（附理由），不要放宽正则。
EXEMPT_PATHS: tuple[str, ...] = ()

TEXT_SUFFIXES = (".py", ".md", ".txt", ".json", ".sh", ".toml", ".yml", ".yaml", ".log")
SKIP_DIRS = (".git", "__pycache__")


def _tracked_text_files() -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    files = []
    for rel in out.splitlines():
        if not rel.endswith(TEXT_SUFFIXES):
            continue
        if any(part in SKIP_DIRS for part in Path(rel).parts):
            continue
        if rel in EXEMPT_PATHS:
            continue
        path = REPO_ROOT / rel
        if path.is_file():
            files.append(path)
    return files


@pytest.mark.parametrize("pattern_name", ["private_address", "secret"])
def test_repo_text_carries_no_private_address_or_secret(pattern_name: str) -> None:
    """公开仓库（含 PyPI 发行物）里不得出现内网地址或凭据样式串。"""
    patterns = (
        (PRIVATE_ADDRESS_RE,) if pattern_name == "private_address" else SECRET_PATTERNS
    )
    hits = []
    for path in _tracked_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in patterns:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append(
                    f"{path.relative_to(REPO_ROOT)}:{line} -> {match.group(0)[:60]}"
                )
    assert not hits, (
        f"{pattern_name} leak(s) in tracked text files (mask them, e.g. "
        f"private addresses become 192.168.x.x): " + "; ".join(hits[:10])
    )
