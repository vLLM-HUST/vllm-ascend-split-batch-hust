"""Contract tests for workload selection (`random`, `sharegpt`, `long-prefix`)
and the decoupled outer serving loop (`tools/op_agent/serving_loop.py`).

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


workloads = _load_module("tools.op_agent.workloads", "tools/op_agent/workloads.py")
custody = _load_module("tools.op_agent.custody", "tools/op_agent/custody.py")
serving_loop = _load_module(
    "tools.op_agent.serving_loop", "tools/op_agent/serving_loop.py"
)
timeline_insight = _load_module(
    "tools.op_agent.timeline_insight", "tools/op_agent/timeline_insight.py"
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


def test_outer_loop_ab_summary_and_evidence_writeback(tmp_path):
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
        json.dumps({"output_throughput": 758.0, "mean_tpot_ms": 37.2}),
        encoding="utf-8",
    )
    off2.write_text(
        json.dumps({"output_throughput": 762.0, "mean_tpot_ms": 37.0}),
        encoding="utf-8",
    )
    on1.write_text(
        json.dumps({"output_throughput": 780.0, "mean_tpot_ms": 36.3}),
        encoding="utf-8",
    )
    on2.write_text(
        json.dumps({"output_throughput": 782.0, "mean_tpot_ms": 36.2}),
        encoding="utf-8",
    )

    summary = serving_loop.summarize_serving_ab_runs(
        [str(off1), str(off2)], [str(on1), str(on2)], workload_id="random"
    )
    assert summary["status"] == "verified_faster"
    assert summary["gen_rate_delta_pct"] > 2.0

    serving_loop.update_operator_outer_evidence(
        str(op_dir),
        serving_l3={"passed": True, "match_summary": "8/8"},
        serving_e2e=summary,
    )
    ok, msg = custody.verify_dir_checksum(str(op_dir))
    assert ok is True, msg


def test_real_collection_rejects_partial_benchmarks(tmp_path):
    import pytest

    collector = _load_module(
        "tools.op_agent.serving_collect", "tools/op_agent/serving_collect.py"
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
        "tools.op_agent.serving_collect", "tools/op_agent/serving_collect.py"
    )
    monkeypatch.setattr(collector, "collect_guard", lambda config, output, arm: [arm])
    monkeypatch.setattr(
        collector,
        "verify_profiles",
        lambda *args: pytest.fail("L3 failure must stop further evaluation"),
    )
    with pytest.raises(ValueError, match="Serving L3 failed"):
        collector.run({}, tmp_path)
    assert not json.loads((tmp_path / "serving_l3.json").read_text())["passed"]
    assert not (tmp_path / "summary.json").exists()


def test_real_collection_requires_off_absence_and_all_rank_on_counts(tmp_path):
    import pytest

    collector = _load_module(
        "tools.op_agent.serving_collect", "tools/op_agent/serving_collect.py"
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
