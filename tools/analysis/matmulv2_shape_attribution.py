#!/usr/bin/env python3
"""MatMulV2 shape/callsite attribution — 只读复算（不采集、不起服、不改输入）。

输入（全部已归档，不改）:
  - records/ladder/1021-031741-opstat.json（对照总量: count 7712 / total_ms 100.03）
  - /root/prof/ladder-1021-031236/.../kernel_details.csv（rank1 light L4，含 Input Shapes）
  - /models/modelscope_cache/Qwen/Qwen3.5-35B-A3B/config.json（层结构常量）

输出: records/ladder/<stamp>-matmulv2-shape-attribution.json（新文件名，追加证据）
"""

import csv
import hashlib
import json
import os
import statistics
import sys

LADDER = "/root/包体恢复/records/ladder"
WIN = "/root/prof/ladder-1021-031236"
KD = os.path.join(
    WIN,
    "dp0_pp0_tp1_dcp0_ep1_rank1_140621_20261021031741458_ascend_pt",
    "ASCEND_PROFILER_OUTPUT",
    "kernel_details.csv",
)
OPSTAT = os.path.join(LADDER, "1021-031741-opstat.json")
CONFIG = "/models/modelscope_cache/Qwen/Qwen3.5-35B-A3B/config.json"
REPO_EV = "/root/vllm-workspace/vllm-ascend-split-batch-hust/docs/evidence/synthetic-fixed-protocol"
STAMP = sys.argv[1] if len(sys.argv) > 1 else "1021-100000"
OUT = os.path.join(LADDER, f"{STAMP}-matmulv2-shape-attribution.json")

MED_COLS = [
    "Duration(us)",
    "Wait Time(us)",
    "Block Num",
    "aicore_time(us)",
    "aicore_ratio",
    "aic_mac_time(us)",
    "aic_mac_ratio",
    "aic_mte2_time(us)",
    "aic_mte3_time(us)",
    "cube_utilization(%)",
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def med(rows, key):
    return round(statistics.median(float(r["raw"].get(key, 0)) for r in rows), 3)


def parse_shapes(s):
    a, b = s.split(";")
    ma, ka = (int(x) for x in a.strip('"').split(","))
    nb, kb = (int(x) for x in b.strip('"').split(","))
    return ma, ka, nb, kb


def main():
    cfg = json.load(open(CONFIG))["text_config"]
    opstat = json.load(open(OPSTAT))
    mm = opstat["ops"]["MatMulV2"]
    all_kernel_ms = opstat["device_ms"]["all_kernel_ms"]

    rows, names = [], set()
    with open(KD) as f:
        for r in csv.DictReader(f):
            if "MatMulV2" not in r["Name"]:
                continue
            names.add(r["Name"])
            m, k, n, kw = parse_shapes(r["Input Shapes"])
            rows.append(
                {
                    "m": m,
                    "k": k,
                    "n": n,
                    "kw": kw,
                    "w": f"{n},{kw}",
                    "start": float(r["Start Time(us)"]),
                    "dur": float(r["Duration(us)"]),
                    "raw": r,
                }
            )
    assert rows, "kernel_details 里没有 MatMulV2 行"

    # ---- 分桶: (m, weight N,K) ----
    buckets = {}
    for r in rows:
        buckets.setdefault((r["m"], r["w"]), []).append(r)

    # ---- 迭代切块: 241 次/迭代 = 7712/32 ----
    rows.sort(key=lambda r: r["start"])
    per_iter = len(rows) // 32
    assert len(rows) % 32 == 0 and per_iter * 32 == len(rows), (len(rows), per_iter)
    iters = [rows[i * per_iter : (i + 1) * per_iter] for i in range(32)]
    prefill_iters = [
        i for i, b in enumerate(iters) if b[0]["m"] > b[-1]["m"] or b[0]["m"] >= 100
    ]
    dec_ms_med = statistics.median(
        sum(r["dur"] for r in b) / 1000.0 for b in iters if b[0]["m"] < 100
    )
    periods = []
    dec_blocks = [b for b in iters if b[0]["m"] < 100]
    for x, y in zip(dec_blocks, dec_blocks[1:]):
        periods.append((y[0]["start"] - x[0]["start"]) / 1000.0)
    step_ms = round(statistics.median(periods), 3) if periods else None

    # ---- callsite 确认表（代码级核对: vllm/model_executor/models/qwen3_next.py）----
    # 次数算术: 每 decode 迭代 241 = 40 层 × (attn 2 + moe 4) + lm_head 1
    callsite_confirmed = {
        "6176,2048": "GDN in_proj_qkvz+ba 合并（F2 落地; 每 GDN 层 1 次 × 30）",
        "2048,2048": "GDN out_proj（30）+ FA o_proj（10）— 同形冲突, 各层 1 次",
        "4608,2048": "FA qkv_proj 含 attn_output_gate（q 2048×2(gate) + k 256 + v 256 = 4608/秩; × 10 FA 层）",
        "256,2048": "MoE router GateLinear（qwen3_next.py:186; num_experts=256; × 40 层）",
        "512,2048": "shared expert gate_up 合并权重（checkpoint 映射 qwen3_next.py:634-635; 中间维 512/TP2=256 ×2; × 40 层）",
        "2048,256": "shared expert down（RowParallel; × 40 层）",
        "1,2048": "shared_expert_gate 标量门 ReplicatedLinear（qwen3_next.py:192; × 40 层）",
        "124160,2048": "lm_head（vocab 248320/TP2; × 1/迭代）",
    }
    dec_iters_n = len(dec_blocks)
    pre_iters_n = 32 - dec_iters_n
    table = []
    for (m, w), rs in sorted(
        buckets.items(), key=lambda kv: -sum(x["dur"] for x in kv[1])
    ):
        tot_ms = sum(x["dur"] for x in rs) / 1000.0
        is_dec = m < 100
        per_call = len(rs) / (dec_iters_n if is_dec else pre_iters_n)
        label = callsite_confirmed.get(w)
        if w == "124160,2048":
            label = callsite_confirmed[w]
        table.append(
            {
                "m": m,
                "weight": w,
                "count": len(rs),
                "calls_per_iter": round(per_call, 2),
                "phase": "decode" if is_dec else "prefill",
                "total_ms": round(tot_ms, 3),
                "share_of_matmulv2_pct": round(100.0 * tot_ms / mm["total_ms"], 3),
                "share_of_all_kernel_pct": round(100.0 * tot_ms / all_kernel_ms, 3),
                "dur_median_us": med(rs, "Duration(us)"),
                "wait_median_us": med(rs, "Wait Time(us)"),
                "block_num_median": med(rs, "Block Num"),
                "aicore_time_us": med(rs, "aicore_time(us)"),
                "aic_mac_time_us": med(rs, "aic_mac_time(us)"),
                "aic_mac_ratio": med(rs, "aic_mac_ratio"),
                "aic_mte2_time_us": med(rs, "aic_mte2_time(us)"),
                "aic_mte3_time_us": med(rs, "aic_mte3_time(us)"),
                "cube_util_pct": med(rs, "cube_utilization(%)"),
                "callsite_hypothesis": label,
                "callsite_status": "code-confirmed" if label else None,
            }
        )

    dec_rows = [r for r in rows if r["m"] < 100]
    pre_rows = [r for r in rows if r["m"] >= 100]
    dec_ms = sum(r["dur"] for r in dec_rows) / 1000.0
    pre_ms = sum(r["dur"] for r in pre_rows) / 1000.0
    phase = {
        "prefill_m353": {
            "m": 353,
            "iters": pre_iters_n,
            "count": len(pre_rows),
            "total_ms": round(pre_ms, 3),
            "share_of_matmulv2_pct": round(100.0 * pre_ms / mm["total_ms"], 3),
        },
        "decode": {
            "m_values": sorted({r["m"] for r in dec_rows}),
            "iters": dec_iters_n,
            "count": len(dec_rows),
            "total_ms": round(dec_ms, 3),
            "share_of_matmulv2_pct": round(100.0 * dec_ms / mm["total_ms"], 3),
            "matmul_ms_per_iter_median": round(dec_ms_med, 3),
            "step_period_ms_median": step_ms,
            "matmul_pct_of_step": (
                round(100.0 * dec_ms_med / step_ms, 2) if step_ms else None
            ),
        },
    }

    checks = [
        {
            "name": "count_vs_opstat",
            "ok": len(rows) == mm["count"],
            "detail": f"kernel_details {len(rows)} vs opstat {mm['count']}",
        },
        {
            "name": "total_ms_vs_opstat",
            "ok": abs(sum(r["dur"] for r in rows) / 1000.0 - mm["total_ms"]) < 0.02,
            "detail": f"复算 {sum(r['dur'] for r in rows) / 1000.0:.3f} vs opstat {mm['total_ms']}",
        },
        {
            "name": "iterations_32_blocks_241",
            "ok": per_iter == 241,
            "detail": f"241 次/迭代 × 32 = {per_iter * 32}",
        },
        {
            "name": "kernel_name_single_variant",
            "ok": len(names) == 1,
            "detail": str(sorted(names)),
        },
        {
            "name": "decode_iters_plus_prefill",
            "ok": dec_iters_n + pre_iters_n == 32,
            "detail": f"decode {dec_iters_n} + prefill {pre_iters_n} = 32",
        },
        {
            "name": "decode_m_eq_captured_requests",
            "ok": phase["decode"]["m_values"] == [1]
            and opstat["load_caliber"]["window_coverage"]["captured_request_count"]
            == 1,
            "detail": f"decode M={phase['decode']['m_values']} vs captured_request_count=1",
        },
    ]
    assert all(c["ok"] for c in checks), [c for c in checks if not c["ok"]]

    out = {
        "schema": "opt-ops-shape-attribution.v1",
        "id": f"{STAMP}-matmulv2-shape-attribution",
        "generated_by": "vllm-ascend-split-batch-hust/tools/analysis/matmulv2_shape_attribution.py（只读复算；未起服、未采集、未跑 NPU）",
        "kind": "derived-evidence（派生证据：只依赖已归档窗口的结构化产物 + light L4 kernel_details.csv）",
        "route": "028/D7-matmulv2-decode-fusion",
        "candidate": "matmulv2-decode-shape-attribution",
        "op": "MatMulV2",
        "load_caliber": opstat["load_caliber"],
        "cohort": opstat["cohort"],
        "model_config_facts": {
            "hidden_size": cfg["hidden_size"],
            "num_hidden_layers": cfg["num_hidden_layers"],
            "full_attention_interval": cfg["full_attention_interval"],
            "num_attention_heads": cfg["num_attention_heads"],
            "num_key_value_heads": cfg["num_key_value_heads"],
            "head_dim": cfg["head_dim"],
            "linear_num_value_heads": cfg["linear_num_value_heads"],
            "linear_num_key_heads": cfg["linear_num_key_heads"],
            "linear_key_head_dim": cfg["linear_key_head_dim"],
            "linear_value_head_dim": cfg["linear_value_head_dim"],
            "linear_conv_kernel_dim": cfg["linear_conv_kernel_dim"],
            "moe_intermediate_size": cfg["moe_intermediate_size"],
            "shared_expert_intermediate_size": cfg["shared_expert_intermediate_size"],
            "vocab_size": cfg["vocab_size"],
            "note": "TP2: 线性/共享专家头与中间维按 rank 折半；gdn 层数 30、full-attention 层数 10",
        },
        "inputs": [
            {
                "path": KD,
                "sha256": sha(KD),
                "role": "rank1 light L4 kernel_details.csv（含 Input Shapes/设备计数器）",
            },
            {"path": OPSTAT, "sha256": sha(OPSTAT), "role": "对照总量"},
            {"path": CONFIG, "sha256": sha(CONFIG), "role": "模型结构常量"},
        ],
        "op_totals": {
            "op": "MatMulV2",
            "count": len(rows),
            "total_ms": round(sum(r["dur"] for r in rows) / 1000.0, 3),
            "all_kernel_ms": all_kernel_ms,
            "share_of_all_kernel_pct": round(100.0 * mm["total_ms"] / all_kernel_ms, 3),
            "rank_scope": "单 rank（TP2/rank1）解析窗；rank0 结构对称未解析",
        },
        "per_shape": table,
        "callsite_confirmation": {
            "method": "decode 次数算术（每迭代 241 = 40 层 × 6 + lm_head）× 权重形状 × 代码结构核对",
            "source": "/root/vllm-workspace/vllm-hust/vllm/model_executor/models/qwen3_next.py",
            "key_lines": {
                "router_gate_linear": 186,
                "shared_expert_gate": 192,
                "attn_output_gate_fuse": 308,
                "experts_call": 259,
                "ckpt_gate_up_mapping": "632-635",
            },
            "note": "decode 相位每桶标签全部 code-confirmed；prefill 迭代同权重形状（M=353）",
        },
        "candidate_ledger_positions": {
            "shared_gate_into_gate_up_n513": {
                "status": "closed",
                "by": "F3-shared-gate-pack rejected",
                "reason": "L3 7/8 输出分歧（bitwise 门），仅关 513direct，不外推方向但不重提",
            },
            "shared_expert_pack_into_gmm": {
                "status": "closed",
                "by": "F3-shared-gate-pack rejected + F3-alt-shared-slot blocked",
                "reason": "Python forward 不被调用（F1 机制结论: FusedMoE 定制层持有 shared_expert 引用）",
            },
            "matmul_tuning_family": {
                "status": "adjacent-rejected",
                "by": "W1R-gmm-tuning-config rejected",
                "reason": "GMM 调优族 rejected；MatMulV2 侧无直接条目, 走 L0 需先立标尺",
            },
            "router_merge_into_inproj_qkv": {
                "status": "open-novel",
                "reason": "台账无 router 方法条目; 与 F2 in_proj 合并同模式, 输入同为残差流 hidden_states",
            },
        },
        "candidate_list": [
            {
                "rank": 1,
                "id": "router-merge-into-inproj-qkv",
                "op_callsite": "MoE router [256,2048] M=1, 40/iter",
                "window_ms": 9.674,
                "expected_delta": "消除 1240 次/窗（decode 全部）+ ~9.7ms/窗; router logits 从合并 GEMM 输出 narrow 切出",
                "risks": "跨 Qwen3NextSparseMoeBlock/attention 模块边界; 图模式下 FusedMoE 消费 router_logits 的协议; bf16 GEMM 数值门 L3 bitwise",
                "gate": "L0(合并 GEMM vs 分离 pair 同形状) → L3(8/8 bitwise) → light L4(M1: MatMulV2 次数 -1240/窗)",
            },
            {
                "rank": 2,
                "id": "matmulv2-decode-m1-l0-ruler",
                "op_callsite": "in_proj [6176,2048] M=1, 26.46µs vs 权重装载下限 ~8µs",
                "window_ms": 25.069,
                "expected_delta": "无直接改动; 先立 M=1 skinny GEMM 标尺判 kernel headroom 是否真实",
                "risks": "tuning 族与 W1R rejected 相邻; 需先问授权才可进 kernel 替换类方法",
                "gate": "L0 eager 单卡 A/B（现 kernel vs 理论装载下限/换 template）",
            },
            {
                "rank": 3,
                "id": "shared-gate-family",
                "op_callsite": "[1,2048] 40/iter 6.1ms + [512,2048] + [2048,256]",
                "window_ms": 24.662,
                "expected_delta": "closed — 见 candidate_ledger_positions; 列此仅为份额完整性",
                "risks": "F3 rejected(bitwise FAIL) / F3-alt blocked",
                "gate": "不得重开, 除非用户授权且新机制假设",
            },
        ],
        "phase_split": phase,
        "cross_checks": checks,
        "gaps": [
            "[2048,2048] 桶是 GDN out（30）+ FA o（10）同形合并计数——实例级归属靠层数算术与模块结构，非逐行栈",
            "单 rank: 仅 TP2/rank1；rank0 未解析",
            "截断窗: max_iterations=32、captured_request_count=1 → decode 占比为该口径下界",
            "kernel_details 无 callsite 栈；callsite 归属由代码结构核对（callsite_confirmation）承载",
        ],
        "conclusions_scope_limited": [
            "本文件只做形状/次数/时长归因、callsite 确认与设备计数器读数，不构成任何优化收益结论",
            "decode 每迭代 241 次 MatMulV2 全部 M=1（captured_request=1），属 skinny GEMM 包络；各桶 callsite 已 code-confirmed",
            "L4 次数归零门与 L0/L3 门按 candidate_list 各 gate 字段执行；rejected/blocked 家族不得静默重开",
        ],
    }
    with open(OUT, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.makedirs(REPO_EV, exist_ok=True)
    repo_copy = os.path.join(REPO_EV, os.path.basename(OUT))
    with open(repo_copy, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote", OUT, "sha", sha(OUT))
    print("repo copy sha", sha(repo_copy))
    print(
        json.dumps(
            {
                "per_shape": [
                    {
                        k: t[k]
                        for k in (
                            "m",
                            "weight",
                            "count",
                            "calls_per_iter",
                            "total_ms",
                            "dur_median_us",
                            "cube_util_pct",
                            "callsite_hypothesis",
                        )
                    }
                    for t in table
                ],
                "phase": phase,
            },
            ensure_ascii=False,
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
