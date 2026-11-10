# moe_down_residual_norm

- **调用位置**: `vllm_ascend/ops/fused_moe/shared_experts.py:down_residual_norm`
- **门控开关**: `VLLM_HUST_MOE_DOWN_RESIDUAL_NORM=1`（默认关闭）
- **外环主测负载**: `random`；动态形状压力负载: `sharegpt`
- **详细度量数据**: 见 `evidence.json`
