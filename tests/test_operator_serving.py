"""Contract tests for workload selection (`random`, `sharegpt`, `long-prefix`)
and the decoupled outer serving loop (`tools/operator_tools/serving_loop.py`).

Uses stdlib only so it runs in every dependency-free CI package environment.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


workloads = _load_module(
    "tools.operator_tools.workloads", "tools/operator_tools/workloads.py"
)
serving_loop = _load_module(
    "tools.operator_tools.serving_loop", "tools/operator_tools/serving_loop.py"
)
timeline_insight = _load_module(
    "tools.operator_tools.timeline_insight", "tools/operator_tools/timeline_insight.py"
)


longrun = _load_module(
    "tools.operator_tools.serving_longrun", "tools/operator_tools/serving_longrun.py"
)


operator_status = _load_module(
    "tools.operator_tools.operator_status", "tools/operator_tools/operator_status.py"
)


def test_status_infers_active_operator_without_progress(tmp_path, monkeypatch):
    (tmp_path / "plan.json").write_text(
        json.dumps({"operators": [{"id": "op", "config": {}}]})
    )
    stage = tmp_path / "op/bench_0_off"
    stage.mkdir(parents=True)
    (stage / "server.log").write_text("Loading model\n")
    monkeypatch.setattr(operator_status, "process_alive", lambda *a: True)
    report = operator_status.collect(tmp_path)
    assert report["status"] == "running"
    assert report["operators"][0]["stage"] == "bench_0_off"
    assert report["operators"][0]["latest_log"] == "Loading model"


def test_status_rejects_summary_without_guards_and_overrides_old_completion(tmp_path):
    (tmp_path / "plan.json").write_text(
        json.dumps({"operators": [{"id": "op", "config": {}}]})
    )
    directory = tmp_path / "op"
    directory.mkdir()
    (directory / "summary.json").write_text('{"random": {"tested": true}}')
    (tmp_path / "completion.json").write_text('{"status": "finished_pending_review"}')
    report = operator_status.collect(tmp_path)
    assert report["status"] == "finished_with_failures"
    assert report["operators"][0]["status"] == "incomplete_evidence"
    for name in ("serving_l3.json", "l4.json"):
        (directory / name).write_text('{"passed": true}')
    assert (
        operator_status.collect(tmp_path)["operators"][0]["status"]
        == "collected_pending_review"
    )


def test_status_displays_skip_reason_from_plan_over_stale_progress(tmp_path):
    reason = "尚未接入独立服务开关；本批未执行服务测试"
    (tmp_path / "plan.json").write_text(
        json.dumps({"operators": [{"id": "op", "reason": reason}]})
    )
    (tmp_path / "progress.json").write_text(
        json.dumps(
            [
                {
                    "id": "op",
                    "status": "not_run",
                    "reason": "Handled by preceding long run",
                }
            ]
        )
    )
    output = tmp_path / "display.json"
    text = operator_status.refresh([tmp_path], output)
    assert f"op: not_run | reason: {reason}" in text
    assert "Handled by preceding" not in text
    saved = json.loads(output.read_text())["runs"][0]["operators"][0]
    assert saved["reason"] == reason
    assert output.with_suffix(".txt").read_text() == text


def test_longrun_requires_complete_inventory():
    import pytest

    with pytest.raises(ValueError, match="every indexed operator"):
        longrun.validate({"operators": [], "timeout_seconds": 60})


def test_longrun_records_failure_and_continues(tmp_path, monkeypatch):
    def fail(config, output):
        raise ValueError("numerical guard failed")

    monkeypatch.setattr(longrun.collector, "run", fail)
    plan = {
        "timeout_seconds": 60,
        "operators": [
            {"id": "first", "config": {}},
            {"id": "second", "reason": "not integrated"},
        ],
    }
    results = longrun.execute(plan, tmp_path)
    assert results[0]["status"] == "failed"
    assert results[1]["status"] == "not_run"
    assert (tmp_path / "first/failure.json").exists()
    assert json.loads((tmp_path / "progress.json").read_text()) == results


def test_server_records_requested_shutdown_and_early_exit(tmp_path, monkeypatch):
    import pytest

    module = longrun.collector

    class Process:
        pid = 123456
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = -15

    process = Process()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(module, "request", lambda *a: None)
    monkeypatch.setattr(module.os, "killpg", lambda *a: None)
    config = {"port": 0, "gate": "VLLM_HUST_TEST", "server_command": ["unused"]}
    with module.server(config, tmp_path / "normal", "off"):
        pass
    result = json.loads((tmp_path / "normal/exit.json").read_text())
    assert result["shutdown_requested"] and result["collection_completed"]
    assert not result["forced_kill"]
    process.returncode = None
    with (
        pytest.raises(RuntimeError, match="before collection finished"),
        module.server(config, tmp_path / "early", "off"),
    ):
        process.returncode = 1
    result = json.loads((tmp_path / "early/exit.json").read_text())
    assert result["returncode_before_shutdown"] == 1
    assert not result["collection_completed"]
    assert not result["shutdown_requested"]


def test_guard_only_never_enters_performance(tmp_path, monkeypatch):
    module = longrun.collector
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]")
    monkeypatch.setattr(module, "verify_host", lambda *a: None)
    monkeypatch.setattr(module, "collect_guard", lambda *a: ["same"] * 8)

    def forbidden(*args):
        raise AssertionError("guard-only must not claim attribution or run benchmarks")

    monkeypatch.setattr(module, "verify_profiles", forbidden)
    module.run({"sharegpt_path": str(dataset), "guard_only": True}, tmp_path)
    assert json.loads((tmp_path / "serving_l3.json").read_text())["passed"]
    assert not (tmp_path / "summary.json").exists()


def test_longrun_wait_uses_exit_event(monkeypatch):
    calls = []
    monkeypatch.setattr(longrun.os, "pidfd_open", lambda pid: 123)
    monkeypatch.setattr(longrun.os, "close", calls.append)
    monkeypatch.setattr(longrun.select, "select", lambda *args: ([123], [], []))
    longrun.wait_previous(99, 60)
    assert calls == [123]


attribution = _load_module(
    "tools.operator_tools.serving_attribution",
    "tools/operator_tools/serving_attribution.py",
)


def _kernel(kind, inputs="", output="", dtype=""):
    return {
        "Type": kind,
        "Input Shapes": inputs,
        "Output Shapes": output,
        "Input Data Types": dtype,
        "Output Data Types": "FLOAT" if kind == "Cast" else dtype,
    }


def test_dedicated_attribution_rejects_noop_and_missing_paths():
    for operator in attribution.GATES:
        assert not attribution.evaluate(operator, [], [])["passed"]


def test_dedicated_attribution_conv_and_gate_cache():
    conv = [_kernel("CausalConv1d")] * 480
    transpose = [_kernel("Transpose", '"4096,4;2"', '"4,4096"')] * 480
    assert attribution.evaluate("gdn_conv_pack", conv + transpose, conv)["passed"]
    assert not attribution.evaluate(
        "gdn_conv_pack", conv + transpose, conv + transpose
    )["passed"]
    router = [_kernel("MatMulV2", "1,2048;256,2048", dtype="FLOAT;FLOAT")] * 640
    cast = [_kernel("Cast", "256,2048", "256,2048", "DT_BF16")] * 640
    assert attribution.evaluate("gate_fp32", router + cast, router)["passed"]
    assert not attribution.evaluate("gate_fp32", router + cast, [])["passed"]


def test_dedicated_attribution_projection_and_split():
    def mm(width):
        return [
            _kernel("MatMulV2", f"1,2048;{width},2048", dtype="DT_BF16;DT_BF16")
        ] * 480

    assert attribution.evaluate("gdn_merge_inproj", mm(6144) + mm(32), mm(6176))[
        "passed"
    ]
    assert not attribution.evaluate("gdn_merge_inproj", mm(6144), mm(6176))["passed"]
    conv = [_kernel("CausalConv1d")] * 480
    fused = [_kernel("_unpack_kernel"), _kernel("_rearrange_kernel")] * 30
    assert attribution.evaluate("gdn_split_fusion", conv, conv + fused)["passed"]
    assert not attribution.evaluate("gdn_split_fusion", conv, conv + fused[:-1])[
        "passed"
    ]


def test_dedicated_attribution_failure_blocks_benchmarks(tmp_path):
    import pytest

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 2048,
                "num_experts": 256,
                "layer_types": ["linear_attention"] * 30 + ["full_attention"] * 10,
            }
        )
    )
    with pytest.raises(ValueError, match="Incomplete per-device"):
        longrun.collector.verify_profiles(
            {
                "attribution": "gate_fp32",
                "gate": attribution.GATES["gate_fp32"],
                "world_size": 2,
                "model_path": str(tmp_path),
                "guard_server_args": ["--enforce-eager", "--no-enable-prefix-caching"],
            },
            tmp_path,
        )


def test_canonical_workloads_present_and_characterized():
    catalog = workloads.WORKLOAD_CATALOG
    for wid in ("random", "sharegpt", "long-prefix"):
        assert wid in catalog
        prof = catalog[wid]
        assert prof.workload_id == wid
        assert prof.batch_token_behavior
        assert prof.bottleneck_focus
        assert prof.default_num_prompts > 0
        assert prof.default_concurrency > 0

    assert "mmlu-pro" not in catalog


def test_operator_workload_selection_pairs_controlled_and_stress():
    # Decode general op -> controlled random@32 + sharegpt stress + prefill cross-check
    dec = workloads.select_workloads_for_operator(phase="decode")
    assert dec["primary_workload"] == "random"
    assert dec["primary_concurrency"] == 32
    assert dec["stress_workload"] == "sharegpt"
    assert dec["cross_check_workload"] == "long-prefix"

    # Row-gated decode op (active up to 32 rows) -> concurrency capped at 16
    gated = workloads.select_workloads_for_operator(phase="decode", max_active_rows=32)
    assert gated["primary_workload"] == "random"
    assert gated["primary_concurrency"] == 16
    assert gated["stress_workload"] == "sharegpt"
    assert gated["cross_check_workload"] == "long-prefix"

    # Prefill-heavy op -> long-prefix + sharegpt stress + decode cross-check
    pref = workloads.select_workloads_for_operator(phase="prefill")
    assert pref["primary_workload"] == "long-prefix"
    assert pref["stress_workload"] == "sharegpt"
    assert pref["cross_check_workload"] == "random"


def test_every_operator_covers_both_execution_phases():
    """A single-phase evaluation can hide a regression in the other phase."""
    for phase in ("decode", "prefill", "decode"):
        pairing = workloads.select_workloads_for_operator(phase=phase)
        ids = {
            pairing["primary_workload"],
            pairing["stress_workload"],
            pairing["cross_check_workload"],
        }
        assert ids == {"random", "sharegpt", "long-prefix"}, ids
        assert pairing["cross_check_rationale"]


def test_outer_loop_plan_emits_all_three_workload_commands():
    plan = serving_loop.build_outer_loop_plan(
        op_name="demo_op",
        gate_env="VLLM_HUST_DEMO=1",
        phase="decode",
        port=8350,
        model="qwen35-35b",
        out_dir="/tmp/plan_out",
    )
    assert plan["workload_pairing"]["primary_workload"] == "random"
    assert plan["workload_pairing"]["cross_check_workload"] == "long-prefix"
    for key in ("primary_bench_cmd", "stress_bench_cmd", "cross_check_bench_cmd"):
        assert plan[key], key
    assert (
        "long-prefix" in plan["cross_check_bench_cmd"]
        or "--random-input-len" in (plan["cross_check_bench_cmd"])
    )


def test_build_bench_serve_cmd_for_random_and_sharegpt(tmp_path):
    fake_sg = tmp_path / "sharegpt.json"
    fake_sg.write_text("[]", encoding="utf-8")

    cmd_rand = workloads.build_bench_serve_cmd(
        workload_id="random",
        port=8350,
        model="qwen35-35b",
        result_dir=str(tmp_path),
        result_filename="r.json",
    )
    assert "--dataset-name" in cmd_rand
    assert "random" in cmd_rand
    assert "--ignore-eos" in cmd_rand

    cmd_sg = workloads.build_bench_serve_cmd(
        workload_id="sharegpt",
        port=8350,
        model="qwen35-35b",
        result_dir=str(tmp_path),
        result_filename="sg.json",
        dataset_path=str(fake_sg),
    )
    assert "sharegpt" in cmd_sg
    assert "--dataset-path" in cmd_sg
    assert str(fake_sg) in cmd_sg


def test_outer_loop_greedy_l3_and_op_statistic_l4(tmp_path):
    # 1. Greedy output comparison
    ok_res = serving_loop.compare_greedy_outputs(["a", "b"], ["a", "b"])
    assert ok_res["passed"] is True
    assert ok_res["match_summary"] == "2/2"

    bad_res = serving_loop.compare_greedy_outputs(["a", "b"], ["a", "c"])
    assert bad_res["passed"] is False
    assert bad_res["mismatched_indices"] == [1]

    # 2. L4 op_statistic.csv parsing + SNR gate
    csv_file = tmp_path / "op_statistic.csv"
    csv_file.write_text(
        "OP Type,Count,Total Time(us)\n"
        "_output_norm_kernel,15360,43468.8\n"
        "MatMulV2,92160,5600000.0\n",
        encoding="utf-8",
    )
    l4 = serving_loop.parse_op_statistic_csv(
        str(csv_file),
        target_kernel_substr="_output_norm_kernel",
        expected_calls=15360,
    )
    assert l4["hit_verified"] is True
    assert l4["call_count_actual"] == 15360
    assert l4["target_avg_us"] == 2.83

    snr_small = serving_loop.evaluate_e2e_snr_gate(
        net_saved_ms=10.0, all_kernel_total_ms=10000.0, serving_noise_floor_pct=1.5
    )
    assert snr_small["recommend_full_ab"] is False

    snr_large = serving_loop.evaluate_e2e_snr_gate(
        net_saved_ms=120.0, all_kernel_total_ms=5700.0, serving_noise_floor_pct=1.5
    )
    assert snr_large["recommend_full_ab"] is True


def test_snr_gate_scales_by_device_share():
    """Kernel savings cannot reach the idle part of the wall-clock window."""
    full = serving_loop.evaluate_e2e_snr_gate(
        net_saved_ms=120.0, all_kernel_total_ms=5700.0, serving_noise_floor_pct=1.5
    )
    assert full["recommend_full_ab"] is True
    assert full["device_share"] == 1.0

    idle = serving_loop.evaluate_e2e_snr_gate(
        net_saved_ms=120.0,
        all_kernel_total_ms=5700.0,
        serving_noise_floor_pct=1.5,
        device_share=0.25,
    )
    assert (
        abs(
            idle["expected_all_kernel_delta_pct"]
            - full["expected_all_kernel_delta_pct"] * 0.25
        )
        < 1e-3
    )
    assert idle["recommend_full_ab"] is False


def test_outer_loop_ab_summary_is_read_only(tmp_path):
    op_dir = tmp_path / "demo_op"
    op_dir.mkdir()
    (op_dir / "evidence.json").write_text(
        json.dumps(
            {
                "$schema": "operator-benchmark-evidence/v1",
                "operator_name": "demo_op",
                "l3_numerical_guard": {"passed": True},
            }
        ),
        encoding="utf-8",
    )

    off1 = tmp_path / "off1.json"
    off2 = tmp_path / "off2.json"
    on1 = tmp_path / "on1.json"
    on2 = tmp_path / "on2.json"
    off1.write_text(
        json.dumps(
            {
                "completed": 128,
                "failed": 0,
                "output_throughput": 758.0,
                "mean_tpot_ms": 37.2,
            }
        ),
        encoding="utf-8",
    )
    off2.write_text(
        json.dumps(
            {
                "completed": 128,
                "failed": 0,
                "output_throughput": 762.0,
                "mean_tpot_ms": 37.0,
            }
        ),
        encoding="utf-8",
    )
    on1.write_text(
        json.dumps(
            {
                "completed": 128,
                "failed": 0,
                "output_throughput": 780.0,
                "mean_tpot_ms": 36.3,
            }
        ),
        encoding="utf-8",
    )
    on2.write_text(
        json.dumps(
            {
                "completed": 128,
                "failed": 0,
                "output_throughput": 782.0,
                "mean_tpot_ms": 36.2,
            }
        ),
        encoding="utf-8",
    )

    summary = serving_loop.summarize_serving_ab_runs(
        [str(off1), str(off2)], [str(on1), str(on2)], workload_id="random"
    )
    assert summary["status"] == "verified_faster"
    assert summary["gen_rate_delta_pct"] > 2.0

    assert not hasattr(serving_loop, "update_operator_outer_evidence")
    assert json.loads((op_dir / "evidence.json").read_text())["l3_numerical_guard"] == {
        "passed": True
    }
    assert not (op_dir / "checksum.sha256").exists()


def test_real_collection_rejects_partial_benchmarks(tmp_path):
    import pytest

    collector = _load_module(
        "tools.operator_tools.serving_collect",
        "tools/operator_tools/serving_collect.py",
    )
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"completed": 127, "failed": 1}))
    with pytest.raises(ValueError, match="Incomplete benchmark"):
        collector.validate_benchmark(path, 128)


def test_real_collection_stops_before_benchmark_on_greedy_mismatch(
    tmp_path, monkeypatch
):
    import pytest

    collector = _load_module(
        "tools.operator_tools.serving_collect",
        "tools/operator_tools/serving_collect.py",
    )
    monkeypatch.setattr(collector, "verify_host", lambda *args: None)
    monkeypatch.setattr(collector, "collect_guard", lambda config, output, arm: [arm])
    monkeypatch.setattr(
        collector,
        "verify_profiles",
        lambda *args: pytest.fail("L3 failure must stop further evaluation"),
    )
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]")
    with pytest.raises(ValueError, match="Serving L3 failed"):
        collector.run({"sharegpt_path": str(dataset)}, tmp_path)
    assert not json.loads((tmp_path / "serving_l3.json").read_text())["passed"]
    assert not (tmp_path / "summary.json").exists()


def test_real_collection_requires_off_absence_and_all_rank_on_counts(tmp_path):
    import pytest

    collector = _load_module(
        "tools.operator_tools.serving_collect",
        "tools/operator_tools/serving_collect.py",
    )
    config = {"world_size": 2, "attention_layers": 10, "kernel": "target"}
    for arm in ("off", "on"):
        for rank in (0, 1):
            path = tmp_path / f"guard_{arm}" / "profile" / str(rank)
            path.mkdir(parents=True)
            count = 160 if arm == "on" else 0
            (path / "op_statistic.csv").write_text(
                f"OP Type,Count,Total Time(us)\ntarget,{count},100\n"
            )
    collector.verify_profiles(config, tmp_path)
    path = tmp_path / "guard_off/profile/0/op_statistic.csv"
    path.write_text("OP Type,Count,Total Time(us)\ntarget,160,100\n")
    with pytest.raises(ValueError, match="L4 attribution failed"):
        collector.verify_profiles(config, tmp_path)


def test_timeline_subgraph_compressor_and_svg_renderer(tmp_path):
    csv_path = tmp_path / "kernel_details.csv"
    rows = [
        "Device_id,Name,Type,Accelerator Core,Duration(us),Wait Time(us),"
        "Input Shapes,Input Data Types,Output Shapes,"
        "aiv_mte2_ratio,aiv_mte3_ratio,aiv_vec_ratio"
    ]
    for _ in range(30):
        rows.extend(
            [
                '0,aclnnMatmul_MatMulV2,MatMulV2,AI_CORE,15.0,0.0,"16,4096",'
                'BF16,"16,6144",0.0,0.0,0.0',
                '0,aclnnSplitD,SplitD,AI_VECTOR_CORE,3.8,0.5,"16,6144",'
                'BF16,"16,2048",0.75,0.2,0.05',
                '0,aclnnRmsNorm,RmsNorm,AI_VECTOR_CORE,4.2,0.2,"256,128",'
                'BF16,"256,128",0.68,0.1,0.22',
                '0,aclnnMul,Mul,AI_VECTOR_CORE,3.0,0.1,"256,128",'
                'BF16,"256,128",0.80,0.1,0.10',
                '0,aclnnMatmul_Out,MatMulV2,AI_CORE,12.0,0.0,"16,2048",'
                'BF16,"16,4096",0.0,0.0,0.0',
            ]
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    digest = timeline_insight.compress_kernel_timeline(str(csv_path), top_k=3)
    assert digest["scanned_events"] == 150
    assert len(digest["top_fusable_chains"]) == 1
    top = digest["top_fusable_chains"][0]
    assert top["pattern_signature"] == "SplitD -> RmsNorm -> Mul"
    assert top["occurrence_count"] == 30
    assert top["anchor_before"]["type"] == "MatMulV2"
    assert top["anchor_after"]["type"] == "MatMulV2"
    assert top["operators"][0]["bottleneck"] == "memory_bound"

    md = timeline_insight.format_insight_markdown(digest)
    assert "SplitD -> RmsNorm -> Mul" in md
    assert "repeats=30x" in md

    svg_file = tmp_path / "swimlane.svg"
    timeline_insight.render_timeline_swimlane_svg(digest, str(svg_file))
    svg_text = svg_file.read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert "SplitD -&gt; RmsNorm -&gt; Mul" in svg_text


def test_summary_rejects_invalid_metrics_and_unpaired_runs(tmp_path):
    import pytest

    paths = [tmp_path / f"run{i}.json" for i in range(4)]
    valid = dict(completed=128, failed=0, output_throughput=2, mean_tpot_ms=1)
    for path in paths:
        path.write_text(json.dumps(valid))
    off, on = list(map(str, paths[:2])), list(map(str, paths[2:]))
    for patch in (
        {"completed": 0},
        {"failed": 1},
        {"output_throughput": float("nan")},
        {"mean_tpot_ms": -1},
        {"mean_tpot_ms": True},
    ):
        paths[0].write_text(json.dumps(valid | patch))
        with pytest.raises(ValueError, match="benchmark"):
            serving_loop.summarize_serving_ab_runs(off, on, "random")
    paths[0].write_text(json.dumps(valid))
    with pytest.raises(ValueError, match="paired"):
        serving_loop.summarize_serving_ab_runs(off[:1], on, "random")
    alias = tmp_path / "alias.json"
    alias.symlink_to(paths[0])
    with pytest.raises(ValueError, match="distinct"):
        serving_loop.summarize_serving_ab_runs(off, [str(alias), on[1]], "random")
    paths[0].write_text(json.dumps(valid | {"completed": 127}))
    with pytest.raises(ValueError, match="same request count"):
        serving_loop.summarize_serving_ab_runs(off, on, "random")


def test_cli_guard_and_attribution_fail_closed(tmp_path):
    base = tmp_path / "off.json"
    on = tmp_path / "on.json"
    base.write_text('["a"]')
    on.write_text('["b"]')
    assert (
        serving_loop.main(
            ["compare-greedy", "--baseline", str(base), "--candidate", str(on)]
        )
        == 1
    )
    csv = tmp_path / "op_statistic.csv"
    csv.write_text("OP Type,Count,Total Time(us)" + chr(10) + "other,1,1" + chr(10))
    assert (
        serving_loop.main(["parse-l4", "--csv", str(csv), "--kernel", "missing"]) == 1
    )


def test_real_collection_covers_catalog_and_uses_pairing(tmp_path, monkeypatch):
    from contextlib import contextmanager

    collector = _load_module(
        "tools.operator_tools.serving_collect",
        "tools/operator_tools/serving_collect.py",
    )
    monkeypatch.setattr(collector, "verify_host", lambda *args: None)
    monkeypatch.setattr(collector, "collect_guard", lambda *args: ["same"])
    monkeypatch.setattr(collector, "verify_profiles", lambda *args: None)
    monkeypatch.setattr(collector, "completion", lambda *args: None)

    @contextmanager
    def fake_server(config, output, arm):
        output.mkdir()
        yield {}

    monkeypatch.setattr(collector, "server", fake_server)
    commands = []

    def bench(cmd, **kwargs):
        commands.append(cmd)
        target = (
            Path(cmd[cmd.index("--result-dir") + 1])
            / cmd[cmd.index("--result-filename") + 1]
        )
        expected = int(cmd[cmd.index("--num-prompts") + 1])
        target.write_text(
            json.dumps(
                {
                    "completed": expected,
                    "failed": 0,
                    "output_throughput": 10,
                    "mean_tpot_ms": 1,
                }
            )
        )

    monkeypatch.setattr(collector.subprocess, "run", bench)
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]")
    collector.run(
        {
            "python": "/python",
            "model_alias": "demo",
            "model_path": "/model",
            "port": 8350,
            "sharegpt_path": str(dataset),
            "phase": "prefill",
        },
        tmp_path,
    )
    assert len(commands) == 12
    assert set(json.loads((tmp_path / "summary.json").read_text())) == {
        "random",
        "sharegpt",
        "long-prefix",
    }
    assert (
        json.loads((tmp_path / "workload_pairing.json").read_text())["primary_workload"]
        == "long-prefix"
    )
    assert any("2048" in cmd for cmd in commands)


def test_workload_configuration_rejects_invalid_values(tmp_path, monkeypatch):
    import pytest

    missing = str(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        workloads.resolve_local_dataset_path("sharegpt", missing)
    monkeypatch.setenv("VLLM_HUST_DATASET_SHAREGPT", missing)
    with pytest.raises(FileNotFoundError):
        workloads.resolve_local_dataset_path("sharegpt")
    for phase, rows in [
        ("typo", None),
        ("decode", 0),
        ("decode", -1),
        ("decode", True),
    ]:
        with pytest.raises(ValueError):
            workloads.select_workloads_for_operator(phase, rows)
    for overrides in [{"num_prompts": 0}, {"concurrency": -1}]:
        with pytest.raises(ValueError):
            workloads.build_bench_serve_cmd(
                "random", 8350, "demo", "/tmp", "r.json", **overrides
            )


def test_l4_rejects_malformed_statistics(tmp_path):
    import pytest

    path = tmp_path / "op_statistic.csv"
    for count, duration in [
        ("1.5", "1"),
        ("-1", "1"),
        ("nan", "1"),
        ("1", "nan"),
        ("1", "-1"),
        ("", "1"),
    ]:
        path.write_text(f"OP Type,Count,Total Time(us)\ntarget,{count},{duration}\n")
        with pytest.raises(ValueError, match="Invalid attribution row"):
            serving_loop.parse_op_statistic_csv(str(path), "target")


def test_collection_rejects_repository_output_before_creation(tmp_path, monkeypatch):
    import pytest

    collector = _load_module(
        "tools.operator_tools.serving_collect",
        "tools/operator_tools/serving_collect.py",
    )
    config = tmp_path / "config.json"
    config.write_text("{}")
    target = ROOT / ".openbitfun" / "invalid-collection-output"
    assert not target.exists()
    monkeypatch.setattr(
        sys, "argv", ["collect", "--config", str(config), "--output", str(target)]
    )
    with pytest.raises(SystemExit) as exc:
        collector.main()
    assert exc.value.code == 2
    assert not target.exists()


def test_collection_rejects_missing_dataset_before_server(tmp_path, monkeypatch):
    import pytest

    collector = _load_module(
        "tools.operator_tools.serving_collect",
        "tools/operator_tools/serving_collect.py",
    )
    monkeypatch.setattr(
        collector,
        "collect_guard",
        lambda *args: pytest.fail("must validate before starting server"),
    )
    with pytest.raises(FileNotFoundError):
        collector.run({"sharegpt_path": str(tmp_path / "missing")}, tmp_path)


def test_host_probe_uses_explicit_cann_environment(tmp_path, monkeypatch):
    import subprocess

    collector = _load_module(
        "tools.operator_tools.serving_collect",
        "tools/operator_tools/serving_collect.py",
    )
    monkeypatch.setenv("PYTHONPATH", "/wrong-host")
    monkeypatch.setenv("VLLM_HUST_UNUSED", "1")

    def probe(argv, **kwargs):
        assert argv[0] == "/existing-venv/bin/python"
        assert kwargs["env"]["PYTHONPATH"] == "/cann/python/site-packages"
        assert "VLLM_HUST_UNUSED" not in kwargs["env"]
        assert kwargs["cwd"] == tmp_path
        return subprocess.CompletedProcess(argv, 0, "{}", "")

    monkeypatch.setattr(collector.subprocess, "run", probe)
    collector.verify_host(
        {
            "python": "/existing-venv/bin/python",
            "server_command": ["/existing-venv/bin/python", "-m", "vllm"],
            "environment": {"PYTHONPATH": "/cann/python/site-packages"},
        },
        tmp_path,
    )
    assert json.loads((tmp_path / "host_preflight.json").read_text())["returncode"] == 0


def test_wrong_host_stops_before_server_start(tmp_path, monkeypatch):
    import pytest

    collector = _load_module(
        "tools.operator_tools.serving_collect",
        "tools/operator_tools/serving_collect.py",
    )
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]")

    def reject_host(*args):
        raise ValueError("Host mismatch")

    monkeypatch.setattr(collector, "verify_host", reject_host)
    monkeypatch.setattr(
        collector,
        "collect_guard",
        lambda *args: pytest.fail("wrong host must stop before starting server"),
    )
    with pytest.raises(ValueError, match="Host mismatch"):
        collector.run({"sharegpt_path": str(dataset)}, tmp_path)


def test_greedy_cli_rejects_non_string_json_and_preserves_blank_lines(tmp_path):
    import pytest

    baseline, candidate = tmp_path / "base.json", tmp_path / "candidate.json"
    candidate.write_text('["a"]')
    for value in [{}, [1], [None]]:
        baseline.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="array of strings"):
            serving_loop.main(
                [
                    "compare-greedy",
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                ]
            )
    baseline, candidate = tmp_path / "base.txt", tmp_path / "candidate.txt"
    baseline.write_text("a\n\nb\n")
    candidate.write_text("a\nb\n")
    assert (
        serving_loop.main(
            [
                "compare-greedy",
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
            ]
        )
        == 1
    )


def test_workload_dry_run_never_writes_fake_metrics(tmp_path):
    import os
    import subprocess

    env = dict(os.environ, DRY_RUN="1")
    env.pop("VLLM_HUST_DATASET_SHAREGPT", None)
    for workload in workloads.WORKLOAD_CATALOG:
        result = subprocess.run(
            [
                "bash",
                str(ROOT / "workloads/run_workload.sh"),
                workload,
                "8350",
                "demo",
                str(tmp_path),
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "workloads/evaluate_outer_loop.sh"),
            "demo",
            "VLLM_HUST_DEMO=1",
            "8350",
            "demo",
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.glob("*.json"))
    env["VLLM_HUST_DATASET_SHAREGPT"] = str(tmp_path / "missing")
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "workloads/run_workload.sh"),
            "sharegpt",
            "8350",
            "demo",
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_serving_sync_history_idempotence_and_custody(tmp_path, monkeypatch):
    import pytest

    from tools.operator_tools import custody, serving_sync

    root, raw = tmp_path / "repo", tmp_path / "raw"
    directory = root / "ops/demo"
    directory.mkdir(parents=True)
    raw.mkdir()
    entry = {
        "id": "demo",
        "evidence": "ops/demo/evidence.json",
        "layer": "host_fork",
        "serving_status": "pending_outer_loop",
    }
    serving_sync.atomic(root / "DELIVERY_INDEX.json", {"operators": [entry]})
    original = {
        "$schema": "operator-benchmark-evidence/v1",
        "l0_microbenchmark": {"speedup": 2},
    }
    serving_sync.atomic(directory / "evidence.json", original)
    custody.write_dir_checksum(str(directory))
    result = {
        "serving_sync": {"archive": raw.as_uri(), "completed_ns": 10},
        "serving_end_to_end": {
            "status": "within_noise",
            "tested": True,
            "speedup": 1.01,
        },
        "l3_numerical_guard": {"passed": True},
        "l4_attribution": {"verified": True},
    }
    monkeypatch.setattr(serving_sync, "validate", lambda *args: result)
    assert serving_sync.sync(raw, "demo", root) == "synced"
    evidence = serving_sync.load(directory / "evidence.json")
    assert evidence["serving_history"][0]["evidence"] == original
    assert evidence["l0_microbenchmark"] == original["l0_microbenchmark"]
    assert custody.verify_dir_checksum(str(directory))[0]
    before = (directory / "evidence.json").read_bytes()
    assert serving_sync.sync(raw, "demo", root) == "already_synced"
    assert (directory / "evidence.json").read_bytes() == before
    result["serving_sync"]["completed_ns"] = 11
    with pytest.raises(ValueError, match="raw data changed"):
        serving_sync.sync(raw, "demo", root)
    (directory / "evidence.json").write_bytes(before + b"\n")
    with pytest.raises(ValueError, match="drift"):
        serving_sync.sync(raw, "demo", root)


def test_serving_sync_rejects_failed_or_missing_collection(tmp_path):
    import pytest

    from tools.operator_tools import serving_sync

    with pytest.raises(FileNotFoundError):
        serving_sync.validate(tmp_path, {})
    (tmp_path / "failure.json").write_text("{}")
    with pytest.raises(ValueError, match="failed"):
        serving_sync.validate(tmp_path, {})


def test_status_marks_requested_stop_without_erasing_completed_guards(tmp_path):
    (tmp_path / "plan.json").write_text(
        json.dumps(
            {
                "operators": [
                    {"id": "queued", "config": {}},
                    {"id": "done", "config": {}},
                ]
            }
        )
    )
    (tmp_path / "progress.json").write_text(
        json.dumps(
            [
                {
                    "id": "done",
                    "finished": 1,
                    "status": "numerical_only_attribution_pending",
                }
            ]
        )
    )
    (tmp_path / "stop-request.json").write_text(
        json.dumps({"reason": "Switch to combination A/B"})
    )
    (tmp_path / "completion.json").write_text(
        json.dumps({"error": "KeyboardInterrupt"})
    )
    result = operator_status.collect(tmp_path)
    assert result["status"] == "stopped_by_request"
    assert result["operators"][0]["reason"] == "Switch to combination A/B"
    assert result["operators"][1]["status"] == "numerical_only_attribution_pending"


def _collector():
    return _load_module(
        "tools.operator_tools.serving_collect",
        "tools/operator_tools/serving_collect.py",
    )


def test_profile_prompts_default_to_the_original_probe():
    collector = _collector()
    assert collector.build_profile_prompts(1, "") == [collector.PROFILE_PROMPT]


def test_shared_prefix_reaches_every_profiled_request():
    collector = _collector()
    prompts = collector.build_profile_prompts(3, "shared context. ")
    assert len(prompts) == 3
    assert all(prompt.startswith("shared context. ") for prompt in prompts)
    # Distinct suffixes, otherwise the cache hit would be indistinguishable
    # from a repeated identical request.
    assert len(set(prompts)) == 3


def test_prefix_is_used_even_for_a_single_request():
    collector = _collector()
    prompts = collector.build_profile_prompts(1, "shared context. ")
    assert len(prompts) == 1
    assert prompts[0].startswith("shared context. ")


def test_shared_prefix_sizing():
    collector = _collector()
    assert collector.build_shared_prefix(0) == ""
    # A cache block is far smaller than this, so the filler must clear it.
    assert len(collector.build_shared_prefix(256)) >= 1024


def test_concurrent_profile_requests_keep_prompt_order(monkeypatch):
    collector = _collector()

    def fake_completion(config, prompt, tokens):
        return {
            "choices": [{"text": prompt}],
            "usage": {"completion_tokens": tokens},
        }

    monkeypatch.setattr(collector, "completion", fake_completion)
    out = collector.run_profile_requests({}, ["a", "b", "c"], 4, 2)
    assert [row["choices"][0]["text"] for row in out] == ["a", "b", "c"]


def test_sequential_path_is_the_default(monkeypatch):
    collector = _collector()
    order = []

    def fake_completion(config, prompt, tokens):
        order.append(prompt)
        return {"choices": [{"text": prompt}], "usage": {"completion_tokens": tokens}}

    monkeypatch.setattr(collector, "completion", fake_completion)
    collector.run_profile_requests({}, ["a", "b"], 4, 1)
    assert order == ["a", "b"]
