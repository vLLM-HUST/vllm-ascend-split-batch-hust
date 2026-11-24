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


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    for directory in ("docs/evidence", "docs/optimization-history", "experiments/demo"):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "DELIVERY_INDEX.json").write_text('{"operators": []}')
    monkeypatch.setattr(index, "ROOT", tmp_path)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["experiment_index", *args])
        index.main()

    return tmp_path, run


def test_changes_require_refresh_but_conclusions_are_not_copied(catalog):
    root, run = catalog
    output = root / "INDEX.md"
    record = root / "docs/optimization-history/demo.json"
    run("--write")
    empty = output.read_bytes()
    record.write_text('{"decision":"first result"}')
    source = record.read_bytes()
    with pytest.raises(SystemExit):
        run()
    run("--write")
    run()
    linked = output.read_bytes()
    assert (
        linked != empty and b"history/demo" in linked and b"first result" not in linked
    )
    assert record.read_bytes() == source
    record.write_text('{"decision":"revised result"}')
    revised = record.read_bytes()
    run()
    run("--write")
    assert output.read_bytes() == linked and record.read_bytes() == revised
    output.write_bytes(linked + b"handwritten verdict\n")
    with pytest.raises(SystemExit):
        run()
    run("--write")
    assert output.read_bytes() == linked
    record.unlink()
    with pytest.raises(SystemExit):
        run()
    run("--write")
    run()
    assert output.read_bytes() == empty


@pytest.mark.parametrize(
    "record", [{}, {"decision": " "}, {"verdict": {}}, {"verdict": {"service": None}}]
)
def test_history_without_a_conclusion_cannot_be_indexed(catalog, record):
    root, run = catalog
    (root / "docs/optimization-history/demo.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="demo.json"):
        run("--write")
    assert not (root / "INDEX.md").exists()


def test_orphan_or_changed_patch_is_not_silently_resealed(catalog):
    root, run = catalog
    path = "experiments/demo/candidate.patch"
    patch = root / path
    patch.write_bytes(b"archived diff")
    with pytest.raises(ValueError, match="candidate.patch"):
        run("--write")
    source = root / "docs/optimization-history/demo.json"
    source.write_text(
        json.dumps(
            {
                "decision": "not measured",
                "patch": path,
                "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
            }
        )
    )
    run("--write")
    before = (source.read_bytes(), (root / "INDEX.md").read_bytes())
    patch.write_bytes(b"altered diff")
    with pytest.raises(ValueError, match="candidate.patch"):
        run("--write")
    assert (source.read_bytes(), (root / "INDEX.md").read_bytes()) == before
    patch.unlink()
    with pytest.raises(ValueError, match="candidate.patch"):
        run()


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


def test_group_renders_human_readable_sections_without_writing_index(catalog, capsys):
    root, run = catalog
    (root / "ops/demo_op").mkdir(parents=True)
    (root / "ops/demo_op/evidence.json").write_text("{}")
    (root / "DELIVERY_INDEX.json").write_text(
        json.dumps(
            {"operators": [{"id": "demo_op", "evidence": "ops/demo_op/evidence.json"}]}
        )
    )
    (root / "experiments/demo/EXPERIMENT_PLAN.md").write_text("# plan\n")
    (root / "docs/optimization-history/demo.json").write_text('{"decision":"done"}')
    (root / "host-proposals/demo-prop").mkdir(parents=True)
    (root / "host-proposals/demo-prop/proposal_meta.json").write_text("{}")
    (root / "docs/evidence/demo-arch").mkdir(parents=True)
    (root / "provenance/legacy-patches/demo-pr").mkdir(parents=True)
    (root / "provenance/legacy-patches/demo-pr/a.patch").write_text("diff\n")

    run("--group")
    out = capsys.readouterr().out
    assert not (root / "INDEX.md").exists()
    expected_order = [
        "## 实验计划（进行中或参考）",
        "- [`plan/demo`](experiments/demo/EXPERIMENT_PLAN.md)",
        "## 算子交付证据（以 DELIVERY_INDEX.json 为准）",
        "- [`op/demo_op`](ops/demo_op/evidence.json)",
        "## 研发记录（阶段性结论）",
        "- [`history/demo`](docs/optimization-history/demo.json)",
        "## 宿主修改提案",
        "- [`proposal/demo-prop`](host-proposals/demo-prop/proposal_meta.json)",
        "## 历史证据归档（只读溯源）",
        "- [`archive/demo-arch`](docs/evidence/demo-arch)",
        "## 早期上游 PR 留痕（只读溯源）",
        "- [`legacy/demo-pr`](provenance/legacy-patches/demo-pr)",
    ]
    positions = [out.index(token) for token in expected_order]
    assert positions == sorted(positions)
