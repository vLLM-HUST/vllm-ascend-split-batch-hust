# MoE Gating 与 Token Routing 融合探索 (`moe_gating_routing`)

## 1. 技术方案与意图
- **目标阶段**：Qwen3.5-35B-A3B MoE 模块的 Token 路由与分发阶段（`MoeGatingTopK` + `MoeInitRoutingV2`）。
- **优化原理**：在解码阶段，模型通过 Router Logits 计算 256 位专家的 Softmax 分布并选取 Top-8 专家，随后将 Token 按目标专家进行排布与计数。本算子尝试将两步调度在软件侧做单 Kernel 融合。
- **状态**：**不予启用（Rejected / Negative Speedup）**，本目录作为反向结论留档。

## 2. 硬件核验现状与负性能结论
- **硬件拓扑**：Atlas 800T A2 (Ascend910B2, CANN 9.1.0)。
- **微基准实测（NPUGraph 回放）**：
  - 宿主底层直接调用了昇腾原生为 MoE 打造的专用硬件指令 `torch_npu.npu_moe_gating_top_k` 与 `torch_npu.npu_moe_init_routing_v2`；
  - 专用硬件指令在图模式下运行效率极高，单步耗时极低；
  - 纯软件级 Triton 重写的分箱排序与内存散布搬运远慢于专用硬件指令，全流程测得负收益，详细数据见同目录结构化证据文件 `evidence.json`。

## 3. 归档决策
- 严格遵循《AGENTS.md》四级验证阶梯与实事求是原则；
- 明确判定为**负收益**，不引入宿主修改，不发布插件运行时，留存反向结论与数据供后续参考。
