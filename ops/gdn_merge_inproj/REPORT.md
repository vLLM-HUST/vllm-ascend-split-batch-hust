# GDN in_proj 输入投影权重加载期合并 — 验收报告

## 1. 形状事实与单变量上下文

| 项 | 值 |
|---|---|
| 算子 | GDN in_proj 权重加载期合并（消除每层每步一次 MatMulV2 与一次结果拼接） |
| callsite | `vllm_ascend/ops/gdn.py:AscendGatedDeltaNetAttention.process_weights_after_loading`（宿主） |
| 相位 | 权重加载后、dynamo 编译/图捕获前（一次性）；收益作用于 decode/prefill 全部前向 |
| 目标模型 | Qwen3.5-35B-A3B（30 层 GDN） |
| 权重形状事实 | `in_proj_qkvz` 与 `in_proj_ba` 沿输出维拼接为单一权重矩阵 |
| 硬件/拓扑 | 2×910B2 TP2 |
| 开关 | `VLLM_HUST_GDN_MERGE_INPROJ=1`（default-off） |
| 交付层 | host_fork（随宿主镜像发布，插件仓仅登记证据） |

## 2. 四级验证阶梯状态

按 [docs/operator-benchmark-standard.md](../../docs/operator-benchmark-standard.md) 四级阶梯逐级如实记录。**本算子当前无任何已归档的配对实测数据，各级均为未测**；不得据此主张任何性能收益。

### L0 单算子微基准 — 未测

- 仓内无归档的 L0 配对微基准记录（合并前两核 vs 合并后单核的同形状配对测量）。
- 机理预期（仅方向，不构成证据）：每 GDN 层每步 MatMulV2 调用 2 → 1，并省去两次输出的显式拼接。

### L3 数值正确性护栏 — 未测

- 仓内无归档的 L3 记录。注意：`[A|B] @ [W1;W2]` 与两矩阵分别相乘再拼接在浮点累加上**不保证位级一致**，L3 门禁必须在 NPU 上以真模型权重实测后方可放行；未通过则按规范一票否决，保持 default-off。

### L4 调用链路归因 — 未测

- 仓内无归档的 L4 窗口（profiler 中 MatMulV2 调用数 60 → 30 的归因核验）。

### 服务端到端实测 — 未测

- 仓内无归档的端到端 A/B（TTFT/TPOT/吞吐）。

## 3. 历史事实（不构成本算子收益证据）

- 该优化的 v1 形态（forward 内 lazy 合并缓存）在 2026-10-15 `1015-175721` 窗实测 ConcatD **净回归 +2636ms/步**（dynamo 下缓存失效导致每步重合并），因此 v2 改为加载期一次性合并。该窗口是"为什么必须加载期做"的机理证据，不是本算子的收益数字。
- 原始运行树补丁层与历史窗口原件在仓外归档；本仓只保留本报告与 evidence.json。

## 4. 结论与交付状态

| 项 | 状态 |
|---|---|
| 宿主实现 | `work/w3f2-inproj-merge` 提交 `fcbc4458bb8afcc37441e602921ed45dcbe5e775`（宿主 PR #63，Base `based/split-batch-baseline`） |
| 代码搬运等价性 | 与旧分支 `work/issue20-w3f2-host` 的 `ce807cd7c` 树逐字节一致（`git diff` 为空） |
| default-off | 是；未设置开关时与 stock 严格一致 |
| L0/L3/L4/端到端 | 全部未测；放行服务端使用前必须补齐 L3 数值护栏 |
