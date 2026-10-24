# 剩余载体按天花板算术收口（2026-10-24）

## 一、方法与输入

输入：`/root/包体恢复/records/ladder/1024-075552-opstat.json`
（`trace_tier=light`，`load_caliber=synthetic-fixed-protocol`，cohort `028-qwen35-35b-a3b-tp2-fullgraph`）

| 量 | 值 |
|---|---|
| `stage_ms` | 1111.1 |
| `computing_ms` | 371.3 |
| `all_kernel_ms` | 476.0 |
| `MatMulV2` 单次 | 13.0 µs（100.27 ms / 7712） |
| GDN layer-step 数 | 960（`fused_gdn_gating_kernel` count） |

天花板＝目标算子 `total_ms`；若目标是"少一次调用"，则＝少掉的调用数 × 单次耗时。
两者都取**全消上界**（假设被消部分完全免费），真实收益必然低于表值。

可判门槛：

- 墙钟（TTFT/TPOT/E2E）会话间漂移 ±3.5% → **38.9 ms** 才可见
- `computing` 跨窗稳定性 ~0.3%（372.5 / 371.3 / 372.5）→ **1.1 ms** 才可见

## 二、结果

| 载体 | 目标 | 上限 ms | %stage | %computing | 判定 |
|---|---|---|---|---|---|
| `fi_gelu` | `GeluAndMul` | 0 | 0 | 0 | **不适用**：profile 无此算子，模型走 `SwiGlu`（9.05 ms） |
| `mlp_chunk` | 激活峰值 | 0 | 0 | 0 | 非延迟收益：分块只增循环与 launch 开销 |
| `aot_cache_guard` | 编译期 AOTAutograd 缓存 | 0 | 0 | 0 | 稳态窗口不可见；只影响启动/预热 |
| `fi_sampling` | decode 采样路径 | ≤1.73 | 0.16 | 0.47 | 低于两道门槛 |
| `rope_fix` | `split_qkv_rmsnorm_mrope_kernel` | ≤2.00 | 0.18 | 0.54 | 低于两道门槛 |
| `gdn_merge_inproj` | `in_proj_ba` 的 `MatMulV2` | ≤12.48 | 1.12 | 3.36 | 低于墙钟门槛 3.1×；仅 `computing` 侧理论上可判 |

`fi_sampling` 的上界取 top-25 最低项 `Abs`（1.73 ms）：采样相关算子在 profile 里不入 top-25，整条路径全消也不超过该值。

## 三、判定

六条全部 **closed-by-ceiling**，保持 default-off，不再为其起服做 A/B。

理由：收益上界低于可测量门槛。此类改动的"通过"无法与平台噪声区分，因此不能作为可发布结论；
结案文本即为"上限 X ms = Y% stage，低于门槛"，不进入基线。

`gdn_merge_inproj` 附带两条边界：

- 墙钟侧不可见（1.12% ≪ 3.5%），只有 `computing` 侧名义可判（3.36% vs 0.3% 地板）；
- 它是六条里唯一存在真实源码缝隙的一条（0.28 的 `gdn.py` 无 in-proj 合并挂点）。
  若未来要验，单次 `serve --profile light` + 只读 `computing` 与 `MatMulV2` total 即可，不需要配对设计。

## 四、重开条件

- 目标算子占比上升（形状变化、或上游改动使其 base 成本变大）→ 用当时 opstat 重算本表；
- `fi_gelu`：仅当换用 GELU 激活的模型时重开；
- `aot_cache_guard`：仅在测量冷启动/首 token 编译延迟时重开，稳态窗口下不重开。

## 五、本轮同时确立的判据边界

1. **MMLU-Pro smoke 不作为任何逐条判据**：n=20 同配置跨 0.55 / 0.60 / 0.40，无判别力。
   仅保留 liveness 用途；定量漂移需同题集 ≥200 且逐题落盘做配对检验。
2. **L4 只比 `computing` 与目标算子 total**：`hcom_allReduce` 跨窗 72–346 ms、调用次数相同，stage/comm/free 不可比。
3. **先算天花板再决定是否测量**：上界低于门槛的改动不安排起服，直接按本文档结案。
4. 配对采集器（`tools/ab_paired.py`）**已撤出**：其设计目标是把分辨率压到 1%，与本判据 3 的"低于门槛即收口"冲突；
   且 2 对样本的中位数不足以支撑统计结论。历史实现与撤出理由记录在本段。
