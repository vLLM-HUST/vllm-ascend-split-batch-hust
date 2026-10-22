# D7 `MatMulV2` decode 方向：现状、已判结论、待外部确认的问题（2026-10-22）

目标形状：decode 相位、`synthetic-fixed-protocol` 口径、cohort
`028-qwen35-35b-a3b-tp2-fullgraph`（Qwen3.5-35B-A3B，TP2，FULL 图档，
vllm 0.28.1.post1.dev143 / vllm_ascend 0.25.1rc2），设备 Ascend910B2。

`MatMulV2` 是该窗第一大算子：**7712 次 / 100.172ms**（同窗 `hcom_allReduce_`
388.6ms、`GroupedMatmul` 94.8ms）。每迭代 241 次 = 40 层 ×（attn 2 + moe 4）+ lm_head，
`kernel_details` 已拆成 15 桶（8 decode + 7 prefill），callsite 全部 code-confirmed。

## 1. 三个方法的终态

| 候选 | 阶段 | 结论 |
|---|---|---|
| `matmulv2-decode-shape-attribution` | inconclusive（交付物） | 桶级归因完成，验证门 100%≥95%，交叉核对 6/6 |
| `router-merge-into-inproj-qkv` | **rejected** | 合并 GEMM 的**前提不成立**（见 §3），不是接缝缺失 |
| `matmulv2-decode-m1-l0-ruler` | inconclusive | 布局面未过预声明门（in_proj 任一臂 ≥15%）；标尺判读口径已按 §2 修正 |

## 2. 标尺口径修正（ruler v2）

旧判读量用 **eager 提交地板**（实测 34.6–36.6µs）做分母，而同一批桶在 FULL 图内的
per-call 只有 **11.97–26.11µs** —— 地板大于被测对象，等于测不到；同时 `copy_` 的
读+写合计带宽被当单向用，`bw_floor` 恰好小一半（gdn_inproj 11.532µs 应为 23.064µs）。
故旧规则「>4MB 桶 headroom<20% → 无 kernel 杠杆」作废（`tools/bench_matmul_m1_ruler.py`
的 `reading.withdrawn` 显式记录）。

按「FULL 图内 per-call 有效读 / 同卡纯读 1.51TB/s」重算（窗 1021-031741，32 iters）：

| 桶 | µs/call | 有效读 GB/s | 占纯读上限 |
|---|---|---|---|
| gdn_inproj | 26.11 | 968.9 | 64.2% |
| fa_qkv | 22.12 | 853.3 | 56.5% |
| out_o | 11.97 | 700.8 | 46.4% |
| lm_head | 366.38 | 1388.1 | **91.9%** |
| shared_gate_up | 7.56 | 277.4 | 18.4% |
| router | 7.56 | 138.7 | 9.2% |
| shared_down | 6.97 | 150.4 | 10.0% |
| shared_gate | 4.77 | 0.9 | 0.1% |

读法：lm_head 已贴纯读天花板（收敛）；大桶离上限还有 36–54%，**存在未测装载空间**，
可达性取决于「A2 上有没有能持续拉满 MTE2 的 M=1 通路」；小桶 4.77–7.56µs/call 是
per-kernel 固定开销主导（地板仅 0.7–1.4µs），杠杆在少发 kernel，不在核内。

## 3. router 合并的前提为何不成立（源码级）

栈内 `vllm/model_executor/models/qwen3_next.py`（fork HEAD `5a820400f`）：

| 行 | 代码 | 含义 |
|---|---|---|
| L561 / L563 | `hidden_states = self.input_layernorm(hidden_states, residual)` | norm1 |
| L570 / L572 | `self.linear_attn(hidden_states=…)` / `self.self_attn(hidden_states=…)` | in_proj / qkv_proj 吃 norm1 |
| L598 | `hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)` | attention 输出进 norm2 |
| L600 / L605 | `hidden_states = self.mlp(hidden_states)` | MoE 吃 norm2 |
| L259–L261 | `self.experts(hidden_states=hidden_states, router_logits=hidden_states)` | router 吃 norm2 |

in_proj 与 router 的输入之间隔着 attention + residual + 一次 RMSNorm，**没有公共激活** →
`cat([W_in, W_router], dim=0)` 在真实图上不可施加。L0 的 −12.5% / −13.5% 只在
「人为给两个 GEMM 同一张 x」的基准里成立，保留为历史证据但不作可行性依据；
旧记录的「FULL 图收益上限 0.9–1.7% all-kernel」一并作废。

## 4. 目前已授权、待卡验证的方向：router + shared_expert_gate 合并

同一 MoE 块里两个门吃**同一个**输入（`qwen3_next.py` L186 `GateLinear(2048, 256)`、
L192 `ReplicatedLinear(2048, 1)`；`qwen2_moe.py` L121
`out = F.sigmoid(self.expert_gate(x)[0]) * out`）→ `cat([W_router, W_shared_gate], 0)`
= **257×2048** 一次 GEMM，语义成立，省一次 kernel 发车。

- 授权：2026-10-22 用户明确授权进入 **vllm-core `qwen3_next.py` 补丁领地**（仅限该合并对）。
- 候选：`router-shared-gate-merge`（见工作台台账）。
- **CPU 侧契约已实现并验证**（2026-10-22）：`tools/moe_gate_merge.py` +
  `tests/test_moe_gate_merge.py`（12 用例，纯 Python、不依赖 torch）：行序契约
  （前 256 行 router / 最后 1 行 shared gate）、两路切片**返回拷贝**（对应 device 侧
  `contiguous` 要求）、以及「一次 257 行 matvec」与「两次 256+1 matvec」在真实维度
  （M=1、K=2048）与稠密小维度上**逐元素精确相等**。
- **运行时载体已实现（default-off，可回滚）**：`src/vllm_ascend_split_batch/moe_gate_merge.py`
  + `runtime_source_patches._GATE_MERGE_TRANSFORMS`，env 门 `VLLM_HUST_MOE_GATE_MERGE`（仅 `=1`）。
  安装点在 `Qwen3NextSparseMoeBlock.forward` 开头（幂等，必须早于该函数体内
  `self.shared_expert(...)`，否则 FULL 图捕获到的是 stock 两路）：把 router 消费者
  （`experts.gate` / `experts._gate`）与 shared 消费者（`shared_expert.expert_gate`）换成共享
  同一 `GateMergeState` 的代理 → 每层一次 257×hidden GEMM，两路 `narrow().contiguous()`；
  两消费者拿到同一张量对象时**只跑一次**，对象身份不成立则**重算**（`state.recomputes` 计数）。
  fail-closed：两路消费者缺一即不安装、保持原对象。宿主暴露 `weight_fp32` 时按 fp32 建合并
  权重并返回 fp32 logits，shared 列转回输入 dtype（避免 `sigmoid(gate)*out` 被提升）。
  **回滚**：`VLLM_HUST_MOE_GATE_MERGE=0`（默认）→ 变换不注册、安装器不执行；不改 checkpoint
  键、不改宿主文件，重启即复原。宿主升级后跑 `tests/test_runtime_gate_merge_patch.py`，
  锚点失配会 fail-closed 回 stock（不是静默生效）。
- **L0 配对 A/B 已就绪**（卡回来一条命令）：`tools/bench_moe_gate_merge.py` → 图捕获回放
  （eager 提交地板 34.6–36.6µs 大于本候选量级，故不用 eager 判读）；臂 = 两次 GEMM vs 一次
  合并 GEMM；预声明门 `gain ≥ 5%` 且两路 bitwise 全等才进 plain L3 + compare。
- 设备侧（bf16 重结合、实测收益）仍未验证。
- 收益上限估算：两桶窗内合计 15.78ms/32 iters，按省一次 ~4.77µs/call 级发车估
  **~0.5–0.7% stage**，属估算，**未测**；实现风险见下。
- 实现风险（L3 前必须交代）：合并后 `hidden` 维不变、行数 257，切片
  `[..., :256]` / `[..., 256:]` 都会产生非连续视图；router logits 进 top-k 前需
  `.contiguous()`（KB 卡 `ge_graph_moe_gate_router_weight_layout`），shared gate 的
  sigmoid 分支吃 `[..., 256:]` 一列同样需要处理；权重为加载期一次性构造。

## 5. 待外部（Ascend / 上游）确认的唯一问题

> 在 Ascend910B2 上，BF16、M=1、K=2048、N=6176/4608/2048 的 `MatMulV2` 实际走的是
> 普通 Cube MatMul tiling、还是 Ascend C 的 `CubeFormat::VECTOR`（GEMV）路径？
> 能否给出实际 `blockDim`、tiling 关键参数与 AIC/AIV 分工，以及该形状的带宽上限？

补充背景（本地 CANN 9.1.0 / V100R001C11SPC001B243，离线可查）：

- `include/pto/npu/a2a3/TMatmul.hpp` L46–L48（无 bias）与 L66–L68（带 bias）非 GEMV 分支
  显式 `m = 16`，注释「avoid gemv mode, if m is 1, the gemv mode will be used in **a3**」
  —— 即该模板层在 a2 上主动绕开 GEMV，GEMV 被标为 a3 行为；
- 本 CANN 的 `opp/.../impl/` 下没有 `mat_mul_v2` 的 Ascend C 实现，`MatMulV2` 对应
  legacy TBE（`ops_legacy/mat_mul.py`、`ops_legacy/util/util_gemm.py`），二者无 gemv 字样；
  GEMV kernel 只出现在第三方模板库（catlass/gemv、mega_moe/arch22/template_linear_algebra_v2/gemv）。
- 因此「Ascend C 提供 GEMV 模式」为真（算子开发 API 层），但「`MatMulV2` 在 a2 上用了它」
  **无证据**；实际 tiling 在二进制里（`libopmaster_ct.so`/`liboptiling.so`），离线读不到。
  引用外部文档时须取与 CANN 9.1.0 对齐的版本页，并确认其芯片（a2/a3）与 dtype（示例为 FP16，本栈 BF16）。

## 6. 边界（不宣称）

- 本文所有 L0 是 eager / 图捕获回放的**方向性**读数，不是 FULL 图端到端收益；
  `~1.5×` 装载杠杆与 `~0.5–0.7% stage`、`~2.4% step` 都是上限/量级估算，**无 L3 数值门支撑**。
- §2 的 per-call 由既有 `window_ms` 折算，**未新跑设备**；旧证据由 ruler v1 产出，
  证据文件内保留 `tool_sha256` 可追溯。
- §4 的合并对**尚未实现**，也没有任何性能结论。

## 7. 仓内相关路径

| 用途 | 路径 |
|---|---|
| 标尺（v2） | `tools/bench_matmul_m1_ruler.py` |
| 标尺口径测试 | `tests/test_bench_matmul_m1_ruler.py` |
| 布局 A/B 工具 | `tools/bench_matmul_m1_layout.py` |
| 合并 L0 工具（旧前提） | `tools/bench_router_merge.py` |
| 桶级归因工具 | `tools/analysis/matmulv2_shape_attribution.py` |
| 双门合并 CPU 参考实现 | `tools/moe_gate_merge.py` |
| 双门合并运行时载体（default-off） | `src/vllm_ascend_split_batch/moe_gate_merge.py` |
| 双门合并锚点变换 | `src/vllm_ascend_split_batch/runtime_source_patches.py`（`_GATE_MERGE_TRANSFORMS`） |
| 双门合并 L0 配对 A/B | `tools/bench_moe_gate_merge.py` |
| 载体/变换测试 | `tests/test_moe_gate_merge_carrier.py`、`tests/test_runtime_gate_merge_patch.py` |
| 双门合并契约测试 | `tests/test_moe_gate_merge.py` |

工作台侧（本仓外）：候选 `matmulv2-decode-m1-l0-ruler` / `matmulv2-decode-shape-attribution` /
`router-merge-into-inproj-qkv`（rejected）/ `router-shared-gate-merge`（proposed），
L1 卡 `d7_ruler_metric_correction_1022`、`d7_router_merge_premise_invalid_1022`，
回执与派生证据见 L2 records（`probes/`、`ladder/`）。
