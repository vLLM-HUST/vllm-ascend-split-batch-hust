# ffn_activation

- **调用位置**: `vllm_ascend/ops/fused_moe/fused_moe.py:ffn_activation`
- **门控开关**: `VLLM_HUST_FFN_ACTIVATION=1`（默认关闭）
- **外环主测负载**: `random`；动态形状压力负载: `sharegpt`
- **详细度量数据**: 见 `evidence.json`
