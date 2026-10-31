# 算子优化模块目录 (ops/)

本目录遵循**“每个算子优化独立子目录 + 完备自洽 + 根目录总索引映射”**的标准化规范：

每个优化项拥有自己的独立子目录，目录内必须包含：
- **`README.md`**：算子算法原理、优化背景、以及宿主/插件调用 seam 详细说明。
- **`REPORT.md`**：严格按四级阶梯（L0 纯核微基准、L3 数值正确性护栏、L4 链路归因、服务端到端实测）排版的专属验收报告。
- **`bench_l0.py`**（或对应测试工具）：单算子无卡/有卡复现微基准脚本。
- **`evidence.json`**：符合 `operator-benchmark-evidence/v1` 标准模式的机器可读评测数据。

根目录通过 [DELIVERY_INDEX.json](../DELIVERY_INDEX.json) 与 [docs/delivery-inventory.md](../docs/delivery-inventory.md) 对本目录下所有算子统一索引。
