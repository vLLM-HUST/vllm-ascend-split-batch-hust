# 算子验收报告: ffn_activation

## 1. 形状事实与门控上下文

- 调用点: `vllm_ascend/ops/fused_moe/fused_moe.py:ffn_activation`
- 精度与并行策略: bfloat16，双卡张量并行
- 门控环境变量: `VLLM_HUST_FFN_ACTIVATION=1`（默认关闭）
- 外环负载绑定: 主测 `random`，压力 `sharegpt`

## 2. 内环与外环验证结论

- L0 单算子微基准: 已通过内环重放验证，分桶数值见 `evidence.json`。
- L3 数值正确性护栏: 已通过容差与动态形状零重编译检查。
- L4 调用归因与服务对比: 由外环采集并回填至 `evidence.json`。
