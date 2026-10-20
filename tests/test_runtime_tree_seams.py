"""CPU-only guards for the runtime-tree deployment scripts canonical copy.

The deployed copies live under /root (mutable PVC); this repo holds the
canonical originals.  Tests here never touch /vllm-workspace and never
import vllm.
"""

import json
import pathlib
import py_compile
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
RT = HERE.parent / "tools" / "runtime-tree"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def test_canonical_copies_exist():
    for name in (
        "apply-runtime-patches.sh",
        "check-tree-seams.py",
        "tree-seam-assertions.json",
    ):
        assert (RT / name).is_file(), name


def test_apply_script_parses():
    proc = _run(["bash", "-n", str(RT / "apply-runtime-patches.sh")])
    assert proc.returncode == 0, proc.stderr


def test_seam_checker_compiles_and_selftests():
    checker = RT / "check-tree-seams.py"
    py_compile.compile(str(checker), doraise=True)
    assert checker.stat().st_mode & 0o111, "部署脚本须保留可执行位"
    proc = _run([str(checker), "--selftest"])
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_seam_table_schema():
    table = json.loads((RT / "tree-seam-assertions.json").read_text())
    entries = table["seams"]
    assert isinstance(entries, list) and entries
    for e in entries:
        assert e["kind"] in ("symbol", "text"), e
        assert e.get("consumer") and e.get("call_token"), e
        if e["kind"] == "symbol":
            assert e.get("definer") and e.get("symbol"), e


def test_manifest_layer_chain_is_well_formed():
    text = (RT / "apply-runtime-patches.sh").read_text()
    rows = []
    in_manifest = False
    for line in text.splitlines():
        if line.startswith('MANIFEST="'):
            in_manifest = True
            continue
        if in_manifest:
            if line.strip() == '"':
                break
            if "|" in line:
                rows.append(line.strip().strip('"').split("|"))
    assert rows, "MANIFEST 未解析到条目"
    seen_rel_order: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        assert len(row) == 4, f"条目 {i} 字段数 != 4: {row}"
        seen_rel_order.setdefault(row[1], []).append(i)
    for rel, idxs in seen_rel_order.items():
        assert idxs == sorted(idxs)
        if len(idxs) > 1:
            for prev, cur in zip(idxs, idxs[1:], strict=False):
                orig_prev = rows[prev][2]
                orig_cur = rows[cur][2]
                assert orig_cur != orig_prev, (
                    f"{rel} 相邻层共用同一 orig 快照: {orig_cur}"
                )
