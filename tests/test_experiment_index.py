import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.operator_tools import experiment_index as index  # noqa: E402


@pytest.mark.parametrize("size", [0, 1, 2, 5, 16, 31, 32, 59])
def test_tree_is_ordered_complete_and_balanced(size):
    rows = iter(index.balanced(range(size), 0, size))
    node = next(rows, None)

    def visit(depth, low, high):
        nonlocal node
        if node is None or not low <= node[1] < high:
            return 0, 0
        level, key = node
        assert level == depth
        node = next(rows, None)
        lh, ln = visit(depth + 1, low, key)
        rh, rn = visit(depth + 1, key + 1, high)
        assert abs(lh - rh) <= 1
        return max(lh, rh) + 1, ln + rn + 1

    assert visit(0, 0, size)[1] == size and node is None


def test_index_current_and_plan_links_resolve():
    rendered = index.render(ROOT)
    assert rendered == (ROOT / "INDEX.md").read_text(), (
        "regenerate the experiment index"
    )
    for text in (rendered, (ROOT / "实验计划.md").read_text()):
        for path in re.findall(r"\]\(([^)]+)\)", text):
            assert (ROOT / unquote(path)).exists(), path
    assert "history/gdn-chunk-prep" in rendered and "op/gdn_split_ba" in rendered


def test_find_reads_only_the_rendered_index(tmp_path, monkeypatch, capsys):
    (tmp_path / "INDEX.md").write_text("  - [`history/demo`](demo.json)\n")
    monkeypatch.setattr(index, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["experiment_index", "--find", "Demo"])
    index.main()
    assert "demo.json" in capsys.readouterr().out
    sys.argv[-1] = "missing"
    with pytest.raises(SystemExit) as exc:
        index.main()
    assert exc.value.code == 1


def test_retired_chunk_patch_is_only_an_archive():
    record = json.loads(
        (ROOT / "docs/optimization-history/gdn-chunk-prep.json").read_bytes()
    )
    patch = ROOT / record["patch"]
    assert patch.suffix == ".patch" and patch.is_relative_to(ROOT / "experiments")
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == record["patch_sha256"]
    assert {
        line.split()[2][2:]
        for line in patch.read_text().splitlines()
        if line.startswith("diff --git ")
    } == set(record["source_sha256"])
    assert record["deployed"] is False and record["service_ab"] is None
