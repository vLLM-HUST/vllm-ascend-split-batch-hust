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

"""守卫的守卫：证明 `tools/seam_diff.py` 抓得住一类已知漂移。

`tools/seam_diff.py` 是宿主升级核对清单的机械前半程（`docs/release.md` §4）。它跑出
"全一致"时有两种可能：**真的没漂移**，或**它根本没在检查**。所以它自带 `--selftest`：
用合成小树注入一次 D1 类漂移（删掉调用点的 `profiler` 关键字），断言工具报 1。

本测试只是把那步自证纳入 CPU 门槛 —— 不读真实宿主树，因此在干净 runner 上也能跑。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "seam_diff.py"


def test_seam_diff_selftest_passes() -> None:
    """工具必须能抓到注入的漂移（自身 exit 0 = 两项断言都成立）。"""
    assert TOOL.is_file(), f"缺失升级对拍工具: {TOOL}"
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--selftest"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"seam_diff --selftest 失败（rc={proc.returncode}）\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    # 输出里必须同时出现"未漂移 rc=0"与"漂移 rc=1"两个判据，避免自证退化成空跑。
    assert "rc=0" in proc.stdout and "rc=1" in proc.stdout, proc.stdout


def test_seam_diff_reports_differences_with_nonzero_exit(tmp_path: Path) -> None:
    """反向断言：给一对**相同**的合成树应报 0，删一个关键字后必须报 1。

    直接调工具而不是调 selftest，验证的是"退出码语义"这一外部契约（可作 CI 判据）。
    """
    body = (
        "def call(self, descs, mode, profiler):\n"
        "    self._capture_cudagraphs(\n"
        "        batch_descriptors=descs,\n"
        "        cudagraph_runtime_mode=mode,\n"
        "        profiler=profiler,\n"
        "    )\n"
    )
    for name, text in (
        ("old", body),
        ("same", body),
        ("drift", body.replace("        profiler=profiler,\n", "")),
    ):
        target = tmp_path / name / "vllm" / "v1" / "worker"
        target.mkdir(parents=True)
        (target / "gpu_model_runner.py").write_text(text, encoding="utf-8")

    def run(left: str, right: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(TOOL),
                str(tmp_path / left),
                str(tmp_path / right),
                "--repo",
                "core",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

    assert run("old", "same").returncode == 0
    drifted = run("old", "drift")
    assert drifted.returncode == 1
    assert "profiler" in drifted.stdout
