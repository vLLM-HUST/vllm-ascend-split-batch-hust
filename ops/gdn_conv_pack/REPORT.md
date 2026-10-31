# 算子验收报告: GDN Conv1d Weight Pre-pack (D4)

## 1. 评测环境与形状事实
- **硬件拓扑**: Atlas 800T A2 (2×910B2, TP=2)
- **软件栈版本**: CANN 9.1.0 / vllm 0.28.1.post1.dev143+gf18cf803c / vllm-ascend 0.25.1rc2.dev125+hust (Commit `74f0c0a27`)
- **模型与权重**: Qwen3.5-35B-A3B (bf16, ND 格式)
- **张量事实**: conv1d.weight `[4096, 1, 4]` -> 打包连续 `[4, 4096]`
- **执行相位**: Decode / Prefill
- **调度与图模式**: cudagraph_mode=NONE

## 2. L0 纯算子对比 (Microbenchmark)
- **测试状态**: 待 NPU 上线后重新执行全量基准扫描。
- **历史记录**: 去除每步的转置调用开销后，前向卷积准备环节耗时显著下降。

## 3. L3 数值一致性核验
- **核验标准**: 预打包权重与原生每次动态转置张量 `torch.equal` 逐字节一致。
- **核验结论**: PASS (位级别恒等，无数值精度损失)。

## 4. L4 调用链路与图归因
- **目标 Callsite**: `vllm_ascend/ops/gdn.py:AscendGatedDeltaNetAttention.forward`
- **命中验证**: 30 层 GDN 全部命中打包权重分支；未置开关时 100% 回退原生分支。

## 5. 服务端到端验证 (Serving L3/L4)
- **测试状态**: 待在带卡规格下进行并发 A/B 压测后补齐 TTFT / TPOT 最终收益指标。

## 6. 结论与交付标签
- [x] 宿主层源码落地完成 (`vllm-ascend-hust` @ `work/issue20-w3f2-host`)
- [x] L3 数值恒等核验通过
- [x] default-off 门控就绪 (`VLLM_HUST_D4_CONV_PACK=1`)
- [ ] 服务端到端实测待补齐
