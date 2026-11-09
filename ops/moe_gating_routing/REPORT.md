# 算子验收报告: MoE Gating 与 Token Routing 融合探索 (`moe_gating_routing`)

## 1. 形状事实与单变量上下文
- **模型**: Qwen3.5-35B-A3B (TP=2)
- **精度**: torch.bfloat16
- **相位**: Decode
- **调用点**: `vllm_ascend/ops/fused_moe/fused_moe.py:261` (MoE Router Gating TopK 与 Init Routing 分发)
- **测试形状**:
  - 输入张量: `router_logits` `[32, 256]` (bf16), `hidden_states` `[32, 2048]` (bf16)
  - 专家总数 $E=256$, Top-K 选择 $K=8$
  - 路由展开维度: active_tokens = $32 \times 8 = 256$
  - 解码形状桶: `[1, 4, 8, 16, 32]`，动态探测步长: `[7, 24]`

## 2. 验证阶梯结论
- **L3 数值正确性**: 通过 (PASS)。候选实现（Softmax + TopK 排序 + 按专家分组 Token 搬运与计数）与宿主参考基线在数值与排列元数据上严格等价。
- **L0 纯微基准**:
  - 宿主原生实现直接调用昇腾专用硬件指令 `torch_npu.npu_moe_gating_top_k` 与 `torch_npu.npu_moe_init_routing_v2`；
  - 在 NPUGraph 图重放模式下，原生专用指令组合执行耗时稳定处于极低水平（详见结构化数据文件 `evidence.json`）；
  - 尝试使用纯软件 Triton 或分立算子融合替代时，由于通用计算核处理 256 专家分箱计数与非规则散布内存搬运时缺少专用硬件支持，耗时严重劣化；
  - 详细基准数值见结构化文件 `evidence.json`。
- **瓶颈归因**:
  - CANN 底层已经为 MoE 调度深度定制了专用的硬件级图元（`npu_moe_gating_top_k` 在 AICore 上硬件级融合，`npu_moe_init_routing_v2` 在硬件协同器上完成排布与计数）；
  - 在小批量解码（$T \le 32$）场景下，专用硬件指令是当前硬件架构上的最优解，软件级重写难以超越。
- **最终交付定性**: **负性能（Negative Speedup），一票否决，不予集成**。
