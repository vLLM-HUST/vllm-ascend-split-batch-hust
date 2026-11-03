# 算子调优 Harness 使用规范与架构契约

本文档是 `tools/op_agent/` 自动化算子调优 Harness 的权威技术规范与操作指南。

---

## 1. 架构与设计原则

1. **环境解耦与可插拔（Pluggable Cases）**：
   - 算子优化目标统一实现 `tools/op_agent/cases/base.py:OpCase` 接口；
   - 新算子独立存在于 `tools/op_agent/cases/<case_name>.py`，禁止在核心调优循环中硬编码算子逻辑；
   - 调优循环通过 `tools/op_agent/cases/get_case(case_name)` 动态加载。

2. **短路评估与资源节约（Short-circuit Evaluation）**：
   - 候选代码在生成后首先进行严格语法与 L3 数值校验（`evaluator.py`）；
   - 语法错误或数值不一致时立即短路退出，绝不执行耗时的重复基准测量；
   - 知识库采用惰性与单例缓存（`knowledge.py`），杜绝重复扫描开销。

3. **四级验证与防篡改闭环**：
   - 调优过程严格遵循 `L0 纯微基准 -> L3 数值正确性护栏`；
   - 调优产物沉淀至 `ops/<operator>/` 目录，必须包含 `bench_l0.py`、`evidence.json`、`README.md` 与 `REPORT.md`；
   - 交付前必须通过 `tools/op_agent/custody.py` 生成并核对 `checksum.sha256`。

---

## 2. 核心入口与执行方式

### 运行自动化调优循环
```bash
# 执行单算子自动调优循环
python3 -m tools.op_agent.optimizer_loop --case <case_name> --max-iters 5
```

### 单独核验目录防篡改指纹
```bash
# 生成并校验指定算子目录哈希
python3 -c "
from tools.op_agent.custody import write_dir_checksum, verify_dir_checksum
write_dir_checksum('ops/<operator>')
ok, msg = verify_dir_checksum('ops/<operator>')
assert ok, msg
"
```
