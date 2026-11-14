"""把文档里记过的踩坑落成机械守卫（纯文本，无设备、无宿主树依赖）。

每条判据都对应文档里的一句约束，改动判据前先改对应文档：

- manifest 的 `host.version_range` 两端有界、`activation.environment` 是真实
  注入值：docs/pitfalls.md（打包与变更）、docs/release.md §3 §6；
- 根 bundle 必须 `import_only` 且不注入环境变量：docs/architecture.md §3；
- manifest `extension_version` 与包版本一致：docs/release.md §6；
- sdist 只含运行时、元数据、README、许可与溯源：docs/release.md §6；
- 仓内不得出现原始数据文件、超大文件、缓存垃圾：
  docs/container-and-plugin-boundary-spec.md §2.4；
- 对外行文不得硬编码宿主工作区绝对路径：docs/pitfalls.md（工作区与宿主）、
  该规范 §2.3 的 CWD 隔离；
- `plugin_carrier` 的门控变量必须在 `src/` 里真实存在：docs/pitfalls.md
  （工作区与宿主 / 插件与证据）、docs/outer-loop-test-plan.md §2.0；
- 索引里的路径必须是仓内相对路径：docs/container-and-plugin-boundary-spec.md §2.3。

无法机械判定的部分（只能人工核对）见 docs/pitfalls.md 末尾"仍靠人工"小节。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "src/vllm_ascend_split_batch"
ROOT_BUNDLE_ID = "org.vllm-hust.split-batch-full-graph"

#: `activation.environment` 的键必须是环境变量名；值必须是可注入的短标记。
ENV_KEY_RE = re.compile(r"^VLLM_[A-Z0-9_]+$")

#: 宿主兼容声明必须两端有界，且不得带本地构建后缀。
VERSION_RANGE_MIN_RE = re.compile(r">=\d")
VERSION_RANGE_MAX_RE = re.compile(r"<[=]?\d")

#: sdist 允许携带的内容（`docs/release.md` §6：运行时、元数据、README、许可、溯源）。
SDIST_REQUIRED = {"/pyproject.toml", "/README.md", "/LICENSE"}
SDIST_ALLOWED_PREFIXES = ("/src/", "/docs/PROVENANCE.md")
SDIST_FORBIDDEN_HINTS = (
    "workbench",
    "host-proposals",
    "ops/",
    "evidence",
    "results",
    "profile",
    "weights",
    ".db",
    ".csv",
)

#: 被当作原始数据、不得进 Git 的后缀。
DATA_SUFFIXES = (
    ".csv",
    ".tsv",
    ".jsonl",
    ".ndjson",
    ".parquet",
    ".arrow",
    ".prom",
    ".log",
    ".log.gz",
    ".db",
    ".sqlite",
    ".sqlite3",
)

#: 单个被追踪文件的上限（当前最大约 0.3 MB）。
MAX_TRACKED_BYTES = 2 * 1024 * 1024

#: 缓存与构建垃圾的路径形态。
JUNK_RE = re.compile(
    r"(^|/)(__pycache__|\.pytest_cache|\.ruff_cache)/|\.pyc$|\.egg-info/|^dist/"
)

#: 对外规范行文（与 tests/test_doc_consistency.py 的 PROSE_SURFACE 保持同步）。
PROSE_SURFACE = (
    "README.md",
    "AGENTS.md",
    "GLOSSARY.md",
    "docs/README.md",
    "docs/architecture.md",
    "docs/agent-guide.md",
    "docs/support-matrix.md",
    "docs/operator-benchmark-standard.md",
    "docs/npu-benchmark-execution-guide.md",
    "docs/op-agent-harness-spec.md",
    "docs/container-and-plugin-boundary-spec.md",
    "docs/HOST_CONTRACT.md",
    "docs/cohort-and-ownership.md",
    "docs/release.md",
    "docs/delivery-inventory.md",
    "docs/pitfalls.md",
    "ops/README.md",
)

#: 宿主工作区路径：写进对外文档就等于把环境写死。
HOST_PATH_RE = re.compile(
    r"/vllm-workspace/|(?<![\w<`\"'])/root/[A-Za-z0-9_\u4e00-\u9fff]+/"
)

#: `plugin_carrier` 但 `src/` 里还没有门控变量的存量债：算子 id -> 现状。
#: 接线完成或索引改判后删除对应条目，不要留过期豁免。
WIRING_EXEMPT = {
    "gdn_qkv_norm": (
        "公开注册已撤回，能力只在宿主提案 host-proposals/GDN-qkv-norm/；"
        "索引保留的是历史交付记录（README 的 GDN QKV norm 段）"
    ),
    "moe_down_residual_norm": (
        "目前只有调优 Harness 用例（tools/op_agent/cases/），未注册 general plugin 入口"
    ),
    "ffn_activation": (
        "目前只有调优 Harness 用例（tools/op_agent/cases/），未注册 general plugin 入口"
    ),
}


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True
    )
    return [name for name in out.stdout.decode().split("\0") if name]


def _manifests() -> list[tuple[Path, dict]]:
    return [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted((ROOT / "src").rglob("vllm-hust-extension-v0.2.json"))
    ]


def test_manifests_declare_bounded_host_range_and_real_activation_values():
    """宿主范围两端有界、不带本地后缀；activation 是真实注入值不是说明文本。"""
    violations = []
    for path, manifest in _manifests():
        name = path.relative_to(ROOT).as_posix()
        version_range = manifest["host"]["version_range"]
        if not VERSION_RANGE_MIN_RE.search(version_range):
            violations.append(f"{name}: version_range 缺下界 {version_range!r}")
        if not VERSION_RANGE_MAX_RE.search(version_range):
            violations.append(f"{name}: version_range 缺上界 {version_range!r}")
        if "+local" in version_range:
            violations.append(f"{name}: version_range 带本地构建后缀 {version_range!r}")
        environment = manifest["activation"]["environment"]
        if "configuration" in environment:
            violations.append(f"{name}: configuration 不是 env 门控的替代入口")
        for key, value in environment.items():
            if not ENV_KEY_RE.match(key):
                violations.append(f"{name}: 环境变量名不合法 {key!r}")
            plain_value = isinstance(value, str) and value and " " not in value
            if not plain_value or len(value) > 32:
                violations.append(f"{name}: {key} 的值不像注入标记：{value!r}")
    assert not violations, "manifest 边界声明不合规: " + "; ".join(violations)


def test_root_bundle_stays_import_only_without_activation_environment():
    """根 bundle 只做规划描述，不可激活、不注入环境变量。"""
    found = False
    for path, manifest in _manifests():
        if manifest.get("extension_id") != ROOT_BUNDLE_ID:
            continue
        found = True
        name = path.relative_to(ROOT).as_posix()
        statuses = [item.get("status") for item in manifest["implementation"]]
        assert statuses and all(status == "import_only" for status in statuses), (
            f"{name}: 根 bundle 只允许 import_only，实际 {statuses}"
        )
        assert manifest["activation"]["environment"] == {}, (
            f"{name}: 根 bundle 的 activation.environment 必须为空"
        )
    assert found, f"未找到根 bundle {ROOT_BUNDLE_ID}"


def test_manifest_version_matches_package_version():
    """manifest 的 extension_version 必须与包版本一致。"""
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]
    mismatched = [
        f"{path.relative_to(ROOT).as_posix()}={manifest['extension_version']}"
        for path, manifest in _manifests()
        if manifest.get("extension_version") != version
    ]
    assert not mismatched, f"extension_version 与包版本 {version} 不一致: " + "; ".join(
        mismatched
    )


def test_sdist_ships_only_runtime_metadata_and_provenance():
    """sdist 只含运行时、元数据、README、许可与溯源，不得夹带工作台或数据。"""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    include = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    missing = SDIST_REQUIRED - set(include)
    assert not missing, f"sdist 缺少必备条目: {sorted(missing)}"
    offenders = [
        entry
        for entry in include
        if entry not in SDIST_REQUIRED and not entry.startswith(SDIST_ALLOWED_PREFIXES)
    ]
    assert not offenders, f"sdist 夹带了非发布内容: {offenders}"
    hinted = [
        entry
        for entry in include
        if any(hint in entry.lower() for hint in SDIST_FORBIDDEN_HINTS)
    ]
    assert not hinted, f"sdist 命中禁止携带的内容: {hinted}"


def test_no_raw_data_files_are_tracked():
    """原始数据、Trace 与数据库一律留在仓外，不得进 Git。"""
    offenders = [
        name for name in _tracked_files() if name.lower().endswith(DATA_SUFFIXES)
    ]
    assert not offenders, "仓内出现原始数据文件: " + "; ".join(offenders[:10])


def test_no_oversized_files_are_tracked():
    """单个被追踪文件不得超限，避免把 Trace 或权重误提交。"""
    offenders = []
    for name in _tracked_files():
        path = ROOT / name
        if not path.is_file():
            continue
        size = os.path.getsize(path)
        if size > MAX_TRACKED_BYTES:
            offenders.append(f"{name} ({size / 1024 / 1024:.1f} MB)")
    assert not offenders, "被追踪文件超过上限: " + "; ".join(offenders)


def test_no_cache_or_build_junk_is_tracked():
    """缓存与构建垃圾不得进版本库。"""
    offenders = [name for name in _tracked_files() if JUNK_RE.search(name)]
    assert not offenders, "追踪了缓存/构建垃圾: " + "; ".join(offenders[:10])


def test_prose_does_not_hardcode_host_workspace_paths():
    """对外行文不得写死宿主工作区绝对路径（含 `/vllm-workspace`）。"""
    violations = []
    for name in PROSE_SURFACE:
        path = ROOT / name
        assert path.is_file(), f"PROSE_SURFACE 里的文件不存在: {name}"
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "<" in line:  # 允许 `<仓外结果目录>/` 这类占位写法
                continue
            if HOST_PATH_RE.search(line):
                violations.append(f"{name}:{lineno}")
    assert not violations, "对外行文写死了宿主路径: " + "; ".join(violations)


def _gate_references() -> str:
    """`src/` 下所有 Python 源码的文本拼接口，用于查门控变量是否真实存在。"""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in PACKAGE_DIR.rglob("*.py")
    )


def test_plugin_carrier_gates_exist_in_source_or_are_declared_debt():
    """`plugin_carrier` 声称的门控变量必须真的出现在 `src/` 里。"""
    index = json.loads((ROOT / "DELIVERY_INDEX.json").read_text(encoding="utf-8"))
    source = _gate_references()
    violations = []
    seen = set()
    for entry in index["operators"]:
        if entry["layer"] != "plugin_carrier":
            continue
        gate_env = entry["gate_env"]
        if gate_env == "unexposed":
            continue
        variable = gate_env.split("=")[0]
        if variable in source:
            continue
        if entry["id"] in WIRING_EXEMPT:
            seen.add(entry["id"])
            continue
        violations.append(f"{entry['id']}: {variable} 在 src/ 里找不到")
    assert not violations, (
        "plugin_carrier 的门控变量没有接线（源码模块/工具不等于公开入口）: "
        + "; ".join(violations)
    )
    stale = sorted(set(WIRING_EXEMPT) - seen)
    assert not stale, f"WIRING_EXEMPT 里的条目已不再需要豁免，请删除: {stale}"


def test_delivery_index_paths_are_repo_relative():
    """索引里的目录与文件路径必须是仓内相对路径。"""
    index = json.loads((ROOT / "DELIVERY_INDEX.json").read_text(encoding="utf-8"))
    violations = []
    for entry in index["operators"]:
        for key in ("directory", "readme", "report", "evidence"):
            value = entry.get(key)
            if not isinstance(value, str):
                continue
            if value.startswith(("/", "~", "file:", "computer:")) or "://" in value:
                violations.append(f"{entry['id']}.{key}={value!r}")
    assert not violations, "索引路径不是仓内相对路径: " + "; ".join(violations)
