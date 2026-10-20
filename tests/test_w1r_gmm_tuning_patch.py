# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only gates for the W1R GMM ``tuning_config`` / counts-dump patch.

Tier-2 host proposal (``host-proposals/W1R-gmm-tuning-config/``): the lever only
touches two private host call sites, so the patch original lives in this
repository and the runtime tree receives a deployed copy.

Nothing here needs ``torch_npu``: the injected helper block is executed as
plain Python with fake tensors, and every structural claim is checked by AST.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import pathlib
import subprocess

import pytest

PROPOSAL = (
    pathlib.Path(__file__).resolve().parents[1]
    / "host-proposals"
    / "W1R-gmm-tuning-config"
)
HOST_ORIG = PROPOSAL / "host_orig" / "routed_experts.py.orig"
PATCH_DIFF = PROPOSAL / "patch.diff"
META = PROPOSAL / "proposal_meta.json"
GEN = PROPOSAL / "gen_patch.py"

BEGIN = "# --- vllm-hust W1R tier-2 host proposal (default-off) BEGIN"
END = "# --- vllm-hust W1R tier-2 host proposal (default-off) END"


def _load_gen():
    spec = importlib.util.spec_from_file_location("w1r_gen_patch", GEN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _meta() -> dict:
    return json.loads(META.read_text(encoding="utf-8"))


@pytest.fixture()
def patched_text() -> str:
    return _load_gen().build_patched(HOST_ORIG.read_text(encoding="utf-8"))


class CountingLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, message: str, *args) -> None:
        self.messages.append(message % args if args else message)


@pytest.fixture()
def helper_ns(patched_text):
    """Execute the injected helper block in a fresh namespace (no torch)."""
    start = patched_text.index(BEGIN)
    end = patched_text.index(END) + len(END)
    block = patched_text[start:end]
    logger = CountingLogger()
    namespace: dict = {"__name__": "w1r_helpers", "logger": logger}
    exec(compile(block, "<w1r-helpers>", "exec"), namespace)
    namespace["_test_logger"] = logger
    return namespace


class FakeTensor:
    """Minimal stand-in: only ``shape`` and ``tolist`` are read by the hook."""

    def __init__(self, shape, values=None, explode_shape=False):
        self.shape = tuple(shape)
        self._values = list(values) if values is not None else None
        self._explode_shape = explode_shape
        self.tolist_calls = 0

    def tolist(self):
        self.tolist_calls += 1
        if self._values is None:
            raise RuntimeError("not a device tensor")
        return list(self._values)


class ExplodingShapeTensor:
    """Any shape read proves the disabled path did not early-return."""

    def __init__(self, shape):
        self._shape = tuple(shape)

    @property
    def shape(self):
        raise AssertionError("shape must not be read when the dump is disabled")

    def tolist(self):
        raise AssertionError("tolist must not run when the dump is disabled")


def test_host_orig_sha_matches_generator(patched_text):
    gen = _load_gen()
    text = HOST_ORIG.read_text(encoding="utf-8")
    assert _sha(text) == gen.EXPECTED_ORIG_SHA256
    assert _sha(text) == _meta()["hashes"]["host_sha256"]


def test_generated_patch_is_byte_identical(patched_text, tmp_path):
    gen = _load_gen()
    out = tmp_path / "patch.diff"
    assert gen.main(["--out", str(out)]) == 0
    assert out.read_bytes() == PATCH_DIFF.read_bytes()
    assert _sha(out.read_text(encoding="utf-8")) == _meta()["hashes"]["patch_sha256"]
    assert _sha(patched_text) == _meta()["hashes"]["patched_sha256"]


def test_generator_fail_closed_on_host_drift(tmp_path):
    gen = _load_gen()
    drifted = tmp_path / "routed_experts.py"
    drifted.write_text(
        HOST_ORIG.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8"
    )
    assert gen.main(["--host", str(drifted), "--check"]) == gen.EXIT_DRIFT


def test_check_mode_flags_a_stale_patch(tmp_path):
    gen = _load_gen()
    stale = tmp_path / "patch.diff"
    stale.write_text("--- stale ---\n", encoding="utf-8")
    assert gen.main(["--check", "--out", str(stale)]) == gen.EXIT_STALE
    fresh = tmp_path / "fresh.diff"
    fresh.write_bytes(PATCH_DIFF.read_bytes())
    assert gen.main(["--check", "--out", str(fresh)]) == 0


def test_patch_applies_and_reverts(patched_text, tmp_path):
    target = tmp_path / "routed_experts.py"
    target.write_bytes(HOST_ORIG.read_bytes())
    orig_sha = _sha(target.read_text(encoding="utf-8"))

    applied = subprocess.run(["patch", "-s", str(target), str(PATCH_DIFF)], check=False)
    assert applied.returncode == 0, applied.stderr
    applied_text = target.read_text(encoding="utf-8")
    assert _sha(applied_text) == _meta()["hashes"]["patched_sha256"]
    assert applied_text == patched_text
    compile(applied_text, str(target), "exec")

    reverted = subprocess.run(
        ["patch", "-R", "-s", str(target), str(PATCH_DIFF)], check=False
    )
    assert reverted.returncode == 0, reverted.stderr
    assert _sha(target.read_text(encoding="utf-8")) == orig_sha


def test_no_new_module_level_import_or_io(patched_text):
    orig = HOST_ORIG.read_text(encoding="utf-8")

    def module_level_imports(text: str) -> set[str]:
        found = set()
        for node in ast.parse(text).body:
            if isinstance(node, ast.Import):
                found.update(f"import {alias.name}" for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names = ",".join(alias.name for alias in node.names)
                found.add(f"from {node.module} import {names}")
        return found

    assert module_level_imports(patched_text) == module_level_imports(orig)

    stack = list(ast.parse(patched_text).body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "open", "module level open() is forbidden"
        stack.extend(ast.iter_child_nodes(node))


def test_tuning_config_is_single_value(helper_ns):
    parse = helper_ns["_hust_w1r_parse_tuning_config"]
    assert parse(None) is None
    assert parse("   ") is None
    for raw, expected in (("1", [1]), ("8", [8]), ("32", [32])):
        assert parse(raw) == expected
    for bad in ("0", "3", "-1", "abc", "1.5", "0x10", "1,2", "8, 32"):
        with pytest.raises(ValueError):
            parse(bad)


def test_tuning_kwargs_zero_diff_and_fail_fast(helper_ns, monkeypatch):
    state = helper_ns["_HUST_W1R_STATE"]
    monkeypatch.delenv("VLLM_HUST_GMM_TUNING_CONFIG", raising=False)
    state.update({"tuning": None, "tuning_ready": False, "tuning_error": None})
    assert helper_ns["_hust_w1r_tuning_kwargs"]() == {}

    monkeypatch.setenv("VLLM_HUST_GMM_TUNING_CONFIG", "8")
    state.update({"tuning": None, "tuning_ready": False, "tuning_error": None})
    assert helper_ns["_hust_w1r_tuning_kwargs"]() == {"tuning_config": [8]}

    monkeypatch.setenv("VLLM_HUST_GMM_TUNING_CONFIG", "5")
    state.update({"tuning": None, "tuning_ready": False, "tuning_error": None})
    with pytest.raises(ValueError):
        helper_ns["_hust_w1r_tuning_kwargs"]()
    # the cached error keeps failing loudly instead of silently proceeding
    with pytest.raises(ValueError):
        helper_ns["_hust_w1r_tuning_kwargs"]()


def test_dump_disabled_returns_before_any_shape_read(helper_ns, monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_HUST_GMM_COUNTS_DUMP", raising=False)
    helper_ns["_HUST_W1R_STATE"].update({"dump_path": None, "dump_path_ready": False})
    result = helper_ns["_hust_w1r_maybe_dump"](
        ExplodingShapeTensor([128, 2048]),
        ExplodingShapeTensor([256, 2048, 512]),
        None,
        1,
        "apply_gmm1",
        False,
    )
    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_dump_eager_sample_schema(helper_ns, monkeypatch, tmp_path):
    out = tmp_path / "dump.jsonl"
    monkeypatch.setenv("VLLM_HUST_GMM_COUNTS_DUMP", str(out))
    monkeypatch.setenv("VLLM_HUST_GMM_COUNTS_DUMP_MAX_PER_KEY", "1")
    monkeypatch.setenv("VLLM_HUST_COHORT_ID", "028-qwen35-35b-a3b-tp2-fullgraph")
    monkeypatch.setenv(
        "VLLM_HUST_LOAD_CALIBER_ID", "mmlu-pro-gdn-high-concurrency-longctx"
    )
    helper_ns["_HUST_W1R_STATE"].update({"dump_path": None, "dump_path_ready": False})
    helper_ns["_hust_w1r_capture_state"] = lambda: False
    counts = [1] * 64 + [0] * 192
    helper_ns["_hust_w1r_maybe_dump"](
        FakeTensor([128, 2048]),
        FakeTensor([256, 2048, 512]),
        FakeTensor([256], counts),
        1,
        "apply_gmm1",
        False,
    )
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [rec["kind"] for rec in lines] == ["meta", "sample"]
    meta, sample = lines
    assert meta["schema"] == "vllm-hust-gmm-counts-dump.v2"
    assert "NOT a timing" not in meta["sampling_semantics"]
    assert "replay does not execute Python" in meta["sampling_semantics"]
    assert sample["sample_source"] == "eager"
    assert sample["group_list_type"] == 1
    assert sample["m"] == 128 and sample["n"] == 512 and sample["experts"] == 256
    assert sample["counts_sum"] == 64 and sample["nonzero_experts"] == 64
    assert sample["counts_normalization"] == "counts"
    assert sample["counts"] == counts
    assert sample["cohort"] == "028-qwen35-35b-a3b-tp2-fullgraph"


def test_dump_cumsum_group_list_is_normalized(helper_ns, monkeypatch, tmp_path):
    """F1: group_list_type=0 (cumsum bounds) must become counts summing to M."""
    out = tmp_path / "dump.jsonl"
    monkeypatch.setenv("VLLM_HUST_GMM_COUNTS_DUMP", str(out))
    monkeypatch.setenv("VLLM_HUST_GMM_COUNTS_DUMP_MAX_PER_KEY", "1")
    helper_ns["_HUST_W1R_STATE"].update({"dump_path": None, "dump_path_ready": False})
    helper_ns["_hust_w1r_capture_state"] = lambda: False
    # 4 experts, 10 tokens: bounds 3,7,7,10 -> counts 3,4,0,3
    helper_ns["_hust_w1r_maybe_dump"](
        FakeTensor([10, 2048]),
        FakeTensor([4, 2048, 512]),
        FakeTensor([4], [3, 7, 7, 10]),
        0,
        "apply_gmm1",
        False,
    )
    sample = json.loads(out.read_text().splitlines()[-1])
    assert sample["group_list_type"] == 0
    assert sample["counts"] == [3, 4, 0, 3]
    assert sample["counts_normalization"] == "cumsum_to_counts"
    assert sample["group_list_raw"] == [3, 7, 7, 10]
    assert sample["counts_sum"] == sample["m"] == 10


def test_dump_weight_list_and_need_trans(helper_ns, monkeypatch, tmp_path):
    """F3: list weights use the first item; need_trans swaps K/N."""
    out = tmp_path / "dump.jsonl"
    monkeypatch.setenv("VLLM_HUST_GMM_COUNTS_DUMP", str(out))
    monkeypatch.setenv("VLLM_HUST_GMM_COUNTS_DUMP_MAX_PER_KEY", "2")
    helper_ns["_HUST_W1R_STATE"].update({"dump_path": None, "dump_path_ready": False})
    helper_ns["_hust_w1r_capture_state"] = lambda: False
    # need_trans=True -> weight laid out [E, N, K] = [256, 512, 2048]
    helper_ns["_hust_w1r_maybe_dump"](
        FakeTensor([128, 2048]),
        [FakeTensor([256, 512, 2048])],
        FakeTensor([256], [1] * 128 + [0] * 128),
        1,
        "apply_gmm1",
        True,
    )
    sample = json.loads(out.read_text().splitlines()[-1])
    assert sample["split_weight_list"] is True
    assert sample["need_trans"] is True
    assert sample["weight_shape"] == [256, 512, 2048]
    assert sample["n"] == 512
    assert sample["k"] == 2048


def test_dump_weight_list_of_2d_experts(helper_ns, monkeypatch, tmp_path):
    """N2: split expert weights -> experts = len(weight), not shape[0]."""
    out = tmp_path / "dump.jsonl"
    monkeypatch.setenv("VLLM_HUST_GMM_COUNTS_DUMP", str(out))
    monkeypatch.setenv("VLLM_HUST_GMM_COUNTS_DUMP_MAX_PER_KEY", "1")
    helper_ns["_HUST_W1R_STATE"].update({"dump_path": None, "dump_path_ready": False})
    helper_ns["_hust_w1r_capture_state"] = lambda: False
    per_expert = [FakeTensor([2048, 512]) for _ in range(4)]
    helper_ns["_hust_w1r_maybe_dump"](
        FakeTensor([8, 2048]),
        per_expert,
        FakeTensor([4], [2, 2, 2, 2]),
        1,
        "apply_gmm1",
        False,
    )
    sample = json.loads(out.read_text().splitlines()[-1])
    assert sample["split_weight_list"] is True
    assert sample["experts"] == 4
    assert sample["weight_shape"] == [2048, 512]
    assert sample["n"] == 512
    assert sample["counts_sum"] == 8


def test_dump_unsupported_group_list_type_keeps_raw_only(
    helper_ns, monkeypatch, tmp_path
):
    """N5: group_list_type outside {0,1} -> raw only, counts None."""
    out = tmp_path / "dump.jsonl"
    monkeypatch.setenv("VLLM_HUST_GMM_COUNTS_DUMP", str(out))
    monkeypatch.setenv("VLLM_HUST_GMM_COUNTS_DUMP_MAX_PER_KEY", "1")
    helper_ns["_HUST_W1R_STATE"].update({"dump_path": None, "dump_path_ready": False})
    helper_ns["_hust_w1r_capture_state"] = lambda: False
    helper_ns["_hust_w1r_maybe_dump"](
        FakeTensor([8, 2048]),
        FakeTensor([4, 2048, 512]),
        FakeTensor([4], [2, 2, 2, 2]),
        2,
        "apply_gmm1",
        False,
    )
    sample = json.loads(out.read_text().splitlines()[-1])
    assert sample["group_list_type"] == 2
    assert sample["group_list_raw"] == [2, 2, 2, 2]
    assert sample["counts"] is None
    assert sample["counts_sum"] is None
    assert sample["counts_normalization"] == "unsupported_group_list_type"


def test_dump_skips_inside_capture_without_device_read(
    helper_ns, monkeypatch, tmp_path
):
    """F2: inside capture the hook must not touch the device tensor."""
    out = tmp_path / "dump.jsonl"
    monkeypatch.setenv("VLLM_HUST_GMM_COUNTS_DUMP", str(out))
    helper_ns["_HUST_W1R_STATE"].update({"dump_path": None, "dump_path_ready": False})
    helper_ns["_hust_w1r_capture_state"] = lambda: True
    group_list = FakeTensor([256], [1] * 128 + [0] * 128)
    helper_ns["_hust_w1r_maybe_dump"](
        FakeTensor([128, 2048]),
        FakeTensor([256, 2048, 512]),
        group_list,
        1,
        "apply_gmm1",
        False,
    )
    assert group_list.tolist_calls == 0
    sample = json.loads(out.read_text().splitlines()[-1])
    assert sample["sample_source"] == "skipped_inside_capture"
    assert sample["capture_state"] is True
    assert sample["counts"] is None


def test_dump_unknown_capture_state_also_skips(helper_ns, monkeypatch, tmp_path):
    out = tmp_path / "dump.jsonl"
    monkeypatch.setenv("VLLM_HUST_GMM_COUNTS_DUMP", str(out))
    helper_ns["_HUST_W1R_STATE"].update({"dump_path": None, "dump_path_ready": False})
    helper_ns["_hust_w1r_capture_state"] = lambda: None
    group_list = FakeTensor([256], [1] * 128 + [0] * 128)
    helper_ns["_hust_w1r_maybe_dump"](
        FakeTensor([128, 2048]),
        FakeTensor([256, 2048, 512]),
        group_list,
        1,
        "apply_gmm1",
        False,
    )
    assert group_list.tolist_calls == 0
    assert (
        json.loads(out.read_text().splitlines()[-1])["sample_source"]
        == "skipped_unknown_capture_state"
    )


def test_dump_warns_once_and_only_counts_successful_writes(
    helper_ns, monkeypatch, tmp_path
):
    """F7: unusable counts -> warn once and no sample; quota only on success."""
    target = tmp_path / "sub" / "dump.jsonl"
    monkeypatch.setenv("VLLM_HUST_GMM_COUNTS_DUMP", str(target))
    monkeypatch.setenv("VLLM_HUST_GMM_COUNTS_DUMP_MAX_PER_KEY", "1")
    helper_ns["_HUST_W1R_STATE"].update({"dump_path": None, "dump_path_ready": False})
    helper_ns["_hust_w1r_capture_state"] = lambda: False
    state = helper_ns["_HUST_W1R_STATE"]
    dump = helper_ns["_hust_w1r_maybe_dump"]
    logger = helper_ns["_test_logger"]

    # 1) counts unreadable -> warn once, no file, no quota consumed
    dump(
        FakeTensor([8, 16]),
        FakeTensor([4, 16, 32]),
        FakeTensor([4]),
        1,
        "apply_gmm1",
        False,
    )
    assert len(logger.messages) == 1
    assert "sample skipped after error" in logger.messages[0]
    assert not target.exists()
    assert state["dump_per_key"] == {}

    # 2) recoverable: a readable call writes the sample and fills the quota
    dump(
        FakeTensor([8, 16]),
        FakeTensor([4, 16, 32]),
        FakeTensor([4], [2, 2, 2, 2]),
        1,
        "apply_gmm1",
        False,
    )
    assert target.exists()
    assert sum(state["dump_per_key"].values()) == 1
    assert len(logger.messages) == 1

    # 3) quota is respected for the same key
    dump(
        FakeTensor([8, 16]),
        FakeTensor([4, 16, 32]),
        FakeTensor([4], [2, 2, 2, 2]),
        1,
        "apply_gmm1",
        False,
    )
    assert sum(state["dump_per_key"].values()) == 1

    # 4) a failing writer warns nothing new and consumes no quota
    state["dump_path"] = str(tmp_path)  # a directory cannot be opened
    dump(
        FakeTensor([9, 16]),
        FakeTensor([4, 16, 32]),
        FakeTensor([4], [3, 2, 2, 2]),
        1,
        "apply_gmm2",
        False,
    )
    assert len(logger.messages) == 1
    assert not any("apply_gmm2" in key for key in state["dump_per_key"])


def test_dump_per_rank_path(helper_ns, monkeypatch, tmp_path):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("VLLM_HUST_GMM_COUNTS_DUMP", str(tmp_path / "d.jsonl"))
    helper_ns["_HUST_W1R_STATE"].update({"dump_path": None, "dump_path_ready": False})
    assert helper_ns["_hust_w1r_dump_path"]().endswith("d.jsonl.rank1")

    template = tmp_path / "d-{rank}.jsonl"
    monkeypatch.setenv("VLLM_HUST_GMM_COUNTS_DUMP", str(template))
    helper_ns["_HUST_W1R_STATE"].update({"dump_path": None, "dump_path_ready": False})
    assert helper_ns["_hust_w1r_dump_path"]().endswith("d-1.jsonl")


def test_dump_creates_parent_directory(helper_ns, monkeypatch, tmp_path):
    nested = tmp_path / "a" / "b" / "dump.jsonl"
    monkeypatch.setenv("VLLM_HUST_GMM_COUNTS_DUMP", str(nested))
    monkeypatch.delenv("RANK", raising=False)
    helper_ns["_HUST_W1R_STATE"].update({"dump_path": None, "dump_path_ready": False})
    assert helper_ns["_hust_w1r_dump_path"]() == str(nested)
    assert nested.parent.is_dir()


def test_dump_disabled_helper_has_early_return(patched_text):
    """F6: the disabled path must return before any shape/phase work."""
    start = patched_text.index(BEGIN)
    end = patched_text.index(END) + len(END)
    tree = ast.parse(patched_text[start:end])
    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_hust_w1r_maybe_dump"
    )
    body = [node for node in func.body if not isinstance(node, ast.Expr)]
    assert isinstance(body[0], ast.Assign), "path resolve must run first"
    assert isinstance(body[1], ast.If), "early return must follow immediately"
    returned = body[1].body[0]
    assert isinstance(returned, ast.Return) and returned.value.value is None


def test_call_sites_forward_output_unchanged(patched_text):
    tree = ast.parse(patched_text)
    hits = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in ("apply_gmm1", "apply_gmm2")
    }
    assert set(hits) == {"apply_gmm1", "apply_gmm2"}
    for name, func in hits.items():
        gmm_assigns = [
            node
            for node in ast.walk(func)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Call)
            and isinstance(node.value.value.func, ast.Attribute)
            and node.value.value.func.attr == "npu_grouped_matmul"
        ]
        assert len(gmm_assigns) == 1, name
        assert gmm_assigns[0].value.slice.value == 0
        dump_calls = [
            node
            for node in ast.walk(func)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_hust_w1r_maybe_dump"
        ]
        assert len(dump_calls) == 1, name
        assert len(dump_calls[0].value.args) == 6, name


def test_meta_pins_and_semantics():
    meta = _meta()
    assert meta["host_version_range"].startswith("==")
    assert "not a measurement" in meta["notes"]["dump_window_status"].lower()
    assert "replay" in meta["env"]["dump"]["sampling_semantics"]
    assert meta["notes"]["tuning_config_semantics"]
    cost = meta["notes"]["default_off_cost"]
    assert "early return" in cost.lower() or "早退" in cost
    assert "not byte-identical" in cost
    assert "two early-return Python calls" in cost
    assert meta["upstream_status"]["state"].startswith("未提交")


def test_python_syntax_of_patched_host(patched_text):
    compile(patched_text, "routed_experts.py", "exec")
