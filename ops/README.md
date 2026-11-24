# 算子优化模块目录 (ops/)

## 已登记算子状态分区速览

下表严格对应 [交付索引](../DELIVERY_INDEX.json)，全部算子均默认关闭且已过数值护栏；服务结论以 `serving_status` 为准，不超证据解读。

| 分区 | 算子 ID | 归属层 (`layer`) | 推进状态 (`status`) | 门控开关 (`gate_env`) | 服务结论 (`serving_status`) |
|---|---|---|---|---|---|
| **已合入** | [`gdn_conv_pack`](gdn_conv_pack/REPORT.md) | `host_fork` | `integrated_host_fork_measured` | `VLLM_HUST_D4_CONV_PACK=1` | `within_noise` |
| **已合入** | [`gdn_split_fusion`](gdn_split_fusion/REPORT.md) | `host_fork` | `integrated_host_fork_measured` | `VLLM_HUST_GDN_UNPACK_FUSED=1` | `within_noise` |
| **已合入** | [`moe_epilogue`](moe_epilogue/REPORT.md) | `host_fork` | `integrated_host_fork_measured` | `VLLM_HUST_MOE_EPILOGUE=1` | `within_noise` |
| **已合入** | [`gdn_merge_inproj`](gdn_merge_inproj/REPORT.md) | `host_fork` | `integrated_host_fork_measured` | `VLLM_HUST_GDN_MERGE_INPROJ=1` | `within_noise` |
| **已合入** | [`gdn_split_ba`](gdn_split_ba/REPORT.md) | `host_fork` | `integrated_host_fork_measured` | `VLLM_HUST_GDN_SPLIT_BA=1` | `within_noise` |
| **已合入** | [`gdn_output_norm_fusion`](gdn_output_norm_fusion/REPORT.md) | `host_fork` | `integrated_host_fork_measured` | `VLLM_HUST_GDN_OUTPUT_NORM_FUSED=1` | `within_noise` |
| **已合入** | [`gate_fp32`](gate_fp32/REPORT.md) | `plugin_carrier` | `integrated_plugin_measured` | `VLLM_HUST_GATE_FP32=1` | `within_noise` |
| **待宿主提案** | [`gdn_qkv_norm`](gdn_qkv_norm/REPORT.md) | `host_fork` | `host_proposal_pending` | `unexposed` | `pending_outer_loop` |
| **已否定或历史归档** | [`gdn_norm_gate`](gdn_norm_gate/REPORT.md) | `plugin_carrier` | `superseded_by_host_fork` | `unexposed` | `superseded_by_host_fork` |
| **已否定或历史归档** | [`moe_gate_merge`](moe_gate_merge/REPORT.md) | `rejected_exploration` | `default_off_negative_speedup` | `unexposed` | `negative_or_pending` |
| **已否定或历史归档** | [`moe_gating_routing`](moe_gating_routing/REPORT.md) | `rejected_exploration` | `default_off_negative_speedup` | `unexposed` | `negative_or_pending` |
| **已否定或历史归档** | [`moe_down_residual_norm`](moe_down_residual_norm/REPORT.md) | `rejected_exploration` | `historical_only` | `unexposed` | `not_applicable` |
| **已否定或历史归档** | [`ffn_activation`](ffn_activation/REPORT.md) | `rejected_exploration` | `historical_only` | `unexposed` | `not_applicable` |

## 目录契约

本目录遵循**“每个算子优化独立子目录 + 完备自洽 + 根目录总索引映射”**的标准化规范：

每个优化项拥有自己的独立子目录，目录内必须包含：
- **`README.md`**：算子算法原理、优化背景、以及宿主/插件调用 seam 详细说明。
- **`REPORT.md`**：严格按四级阶梯（L0 纯核微基准、L3 数值正确性护栏、L4 链路归因、服务端到端实测）排版的专属验收报告。
- **`bench_l0.py`**（或对应测试工具）：单算子无卡/有卡复现微基准脚本。
- **`evidence.json`**：符合 `operator-benchmark-evidence/v1` 标准模式的机器可读评测数据。L0 计时入账由 [`../tools/operator_tools/l0_trace_record.py`](../tools/operator_tools/l0_trace_record.py) 机械追加到 `l0_traces`，禁止手抄度量数字。

[交付索引](../DELIVERY_INDEX.json)记录各算子的归属与状态，字段判据见[交付规范](../docs/delivery-inventory.md)。[实验索引](../INDEX.md)可按主题查找证据和[宿主提案](../host-proposals/README.md)；宿主补丁和回滚条件只在提案目录维护。
