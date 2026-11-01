# 算子验收报告: MoE Shared-Expert Epilogue Multiply Fusion (`AscendSharedExperts.part2`)

## 1. 评测环境与形状事实

- **硬件拓扑**: Atlas 800T A2 (2×910B2, TP=2)
- **环境版本**: CANN 9.1.0 / vllm `0.28.1.post1.dev143+gf18cf803c` / vllm-ascend `0.25.1rc2.dev125+hust`（镜像 Commit `74f0c0a27`）
- **模型与权重**: Qwen3.5-35B-A3B (bf16, ND 格式)，40 层全部 MoE（30 层 GDN + 10 层 full attention，256 experts / top-8，shared expert 512）
- **形状参数**: `gate [T, 1]`、`shared_out [T, 2048]`；decode 桶 T ∈ {1, 8, 16, 32}，prefill 桶 T ∈ {266, 1867, …}
- **图模式与采样**: `cudagraph_mode=FULL`，capture `[1,2,4,8,16,32]`，chunked prefill on，APC off，无投机、无量化；L3 使用固定 8 条 prompt + `temperature=0.0`
- **开关**: `VLLM_HUST_MOE_EPILOGUE=1`（默认 `0`）；`VLLM_HUST_MOE_EPILOGUE_MAXROWS` 默认 `32`

## 2. L0 纯算子对比 (Microbenchmark)

图内打包 50 次后摊 replay，单卡 910B2，D=2048，bf16，单位 µs/次。脚本 `bench_l0.py`，直接调用宿主实现 `moe_epilogue.gate_mul`，对照单个 aten 广播乘 `gate * out`：

| M | 原生 `gate * out` | 融合 `moe_gate_mul` | 省 | 省% | 逐位一致 |
|---|---|---|---|---|---|
| 1 | 1.17 | **1.00** | 0.17 | 14.2% | ✅ |
| 8 | 3.64 | **1.34** | **2.30** | **63.1%** | ✅ |
| 16 | 3.68 | **1.71** | 1.96 | 53.4% | ✅ |
| 32 | 3.73 | **2.43** | 1.30 | 34.7% | ✅ |

**逐位一致**（bitwise True，M ∈ {1,8,16,32}）：kernel 内部按 fp32 相乘后 round 到 bf16，与 aten 的 `mul` 走同一条舍入路径。

收益随 M 增大递减（63.1% → 34.7%），因为融合 kernel 的耗时随 M 线性增长，而原生广播乘的固定启动开销被摊薄。M=32 是门控上限，再大就交给 aclnn。

### 立项阶段的历史探针（另一形态，仅作对照）

| 腿 | M=1 | M=8 | M=16 |
|---|---|---|---|
| `stock3`（`sigmoid → mul → add`） | 3.16 | 6.30 | 6.96 |
| `fused3`（三算子合一，未落地） | 1.27 | 1.80 | 2.43 |
| `stock2`（`sigmoid → mul`） | 1.94 | 4.63 | 4.66 |
| `fused2`（`sigmoid + mul` 合一，未落地） | 1.15 | 1.47 | 1.80 |

这两条腿都不是最终交付形态。`fused3` 与 eager 三算子链的最大偏差 0.0156–0.0312（≈1 ulp），不满足逐位相等——因为当时拿 eager 三 aten op 当参考，而线上跑的是编译后已融合的图，两者本就不是同一舍入路径。**当前交付形态的 L0 已改由上表实测，逐位一致成立。**

## 3. L3 数值正确性护栏

- 方式：固定 8 条 prompt、`temperature=0`、`max_tokens=64`，开启与关闭两腿独立起服，逐条比对输出文本
- 结果：**8/8 完全一致**
- 额外的自算 sigmoid 方案（不保留宿主 `Sigmoid`）实测 4/8 匹配，已否决

判据：`metric = exact_token_match`，通过。

### 宿主源码直改后的复测（2026-11-01）

原数据是在「镜像宿主 + 运行时源码变换」形态下采集的。改成宿主源码直改（提交 `ace3d3842`）后重新对拍一次：

| 项 | 值 |
|---|---|
| 宿主代码 | `vllm_ascend` 取自宿主工作树的 N1 实现（overlay 到容器树，`PYTHONPATH` 前置） |
| 服务参数 | TP2、`max_model_len=8192`、`cudagraph_mode=FULL`、capture `[1,2,4,8,16,32]`、APC off |
| 两腿 | `VLLM_HUST_MOE_EPILOGUE` 未设 / `=1` |
| 结果 | **8/8 逐条完全一致**（输出文件长度均 4756 字节） |

产物：`records/l3host-off.json`、`records/l3host-on.json`。

## 4. L4 图内调用与链路归因

> **口径提示**：本节数据来自旧的「镜像宿主 + 运行时源码变换」形态，尚未在宿主源码直改形态下重新采集。机制层面另有证据：`tests/ut/ops/test_moe_epilogue.py::test_decode_shape_dispatches_kernel` 断言 decode 形状确实走 kernel 分支而非回退路径。

同一 cohort 下 rank0、非通信设备时间，64 个 decode step：

| 指标 | 基线 | 开启后 | 变化 |
|---|---|---|---|
| `moe_gate_mul` kernel | 0 | **2470 次 / 6.35 ms** | 融合路径命中 |
| `Mul`（decode） | 3187 次 / 22.36 ms | **617 次 / 2.67 ms** | −2570 次 |
| decode 非通信设备时间 | 515.2 ms | — | 分母 |

- 调用次数符合预期模型：40 层 × 64 step ≈ 2560 次，实测 2470 次，差值为末层与 step 边界未填满 decode 桶的部分。
- 残余 617 次 `Mul` 来自 full attention 层的 `attn_output_gate`（10 层 × 60 step ≈ 600），与本算子无关。
- 单次耗时：原生 `Mul` 22.36 ms / 3187 ≈ **7.01 µs**；融合 `moe_gate_mul` 6.35 ms / 2470 ≈ **2.57 µs**。

computing 口径（同族合计 `Mul`）：22.36 ms → 2.67 ms，净 **−13.10 ms**，占 decode 非通信 **−2.54%**，占 computing 分母 1108.9 ms 的 **−1.18%**。

## 5. 服务端到端

主口径：random `2048/256 @32`，ABBA 平衡设计，2 块 × 4 腿 = 8 腿。

| 指标 | A（stock） | B（开启） | 差异 |
|---|---|---|---|
| tok/s | 434.71 | 429.11 | **−1.29%** |
| TPOT | 66.27 ms | 66.85 ms | **+0.88%** |

第二口径（`256/512 @1`，ABBA 4 腿）：TPOT −0.58%，但同配置 A 腿之间的极差达 2.04%，差异小于腿间波动。

**结论：三种口径全部落在噪声范围内，不主张端到端收益。**

## 6. 结论与交付决策

| 判据 | 结果 |
|---|---|
| 机制（kernel 是否真的替换目标 task） | ✅ `moe_gate_mul` 稳定命中 2470 次 |
| 数值正确性（L3） | ✅ 8/8 逐字一致 |
| L4 调用归因 | ✅ `Mul` 3187 → 617，残余可解释 |
| computing 收益 | ✅ −1.18% |
| 服务端到端 | ❌ 噪声内，不成立 |

交付决策：**保持 `default_enabled: false`**。以「computing 级收益成立、端到端未证」归档，不得对外宣称服务端加速。

## 7. 证据边界

- 全部数字属于本文 §1 的 cohort，离开该 cohort 只能比较方向，不能拼接数字。
- computing 与 e2e 不得互相推导。
- 宿主提交 `ace3d3842` 尚未合入基线；合入并重建镜像后需复测 L3 一次，因为原数据是在「镜像宿主 + 运行时源码变换」条件下采集的。
