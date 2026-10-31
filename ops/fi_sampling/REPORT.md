# 算子验收报告: FlashInfer Sampling Triton Port (`fi-sampling`)

## 1. 评测环境与形状事实
- **历史测试栈**: Qwen2.5-Coder-14B / CANN / 910B2 / TP=1
- **当前目标基准**: 0.28 镜像栈 (vllm 0.28.1.post1.dev143+gf18cf803c + vllm-ascend 0.25.1rc2.dev125+hust) + Qwen3.5-35B-A3B (TP=2)
- **触发门限**: 联合核要求 Batch >= 256 且 K >= 32

## 2. L0 纯算子对比 (Microbenchmark)
- **测量现状**: 在历史孤立微基准中有局部收益，但与当前 TP2 联合场景尚未重新标定。

## 3. L3 数值一致性核验
- **核验标准**: argmax 路径严格对齐原生采样；随机采样通过固定随机种子对齐分布。
- **核验结论**: PASS (五大强制回退保障了未覆盖场景完全回退宿主)。

## 4. L4 调用链路与图归因
- **Callsite**: `AscendTopKTopPSampler` 子类路由。
- **调用状态**: 仅在环境变量开启且符合批规模时分流至 Triton 算子，其余全部留原链。

## 5. 服务端到端验证 (Serving L3/L4)
- **历史记录**: 端到端耗时差异在噪声范围内 (`within noise`)。
- **当前结论**: 当前 0.28 / Qwen3.5 域尚未证明，不得跨域外推。

## 6. 结论与交付标签
- [x] Triton 实现与 general plugin 已就绪
- [x] L3 路由安全与五大回退保障完备
- [x] 标记为 historical/direction-only，当前域待重测
- [x] 保持 default-off
