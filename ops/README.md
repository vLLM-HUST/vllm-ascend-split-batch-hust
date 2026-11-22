# 算子优化模块目录 (ops/)

本目录遵循**“每个算子优化独立子目录 + 完备自洽 + 根目录总索引映射”**的标准化规范：

每个优化项拥有自己的独立子目录，目录内必须包含：
- **`README.md`**：算子算法原理、优化背景、以及宿主/插件调用 seam 详细说明。
- **`REPORT.md`**：严格按四级阶梯（L0 纯核微基准、L3 数值正确性护栏、L4 链路归因、服务端到端实测）排版的专属验收报告。
- **`bench_l0.py`**（或对应测试工具）：单算子无卡/有卡复现微基准脚本。
- **`evidence.json`**：符合 `operator-benchmark-evidence/v1` 标准模式的机器可读评测数据。L0 计时入账由 [`../tools/operator_tools/l0_trace_record.py`](../tools/operator_tools/l0_trace_record.py) 机械追加到 `l0_traces`，禁止手抄度量数字。

[交付索引](../DELIVERY_INDEX.json)记录各算子的归属与状态，字段判据见[交付规范](../docs/delivery-inventory.md)。[实验索引](../INDEX.md)可按主题查找证据和[宿主提案](../host-proposals/README.md)；宿主补丁和回滚条件只在提案目录维护。
