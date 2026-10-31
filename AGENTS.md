# AGENTS.md — 开发规范与工作约束

本文件是面向开发者与 Agent 的最高工作契约。只写硬约束与执行规则，不记录历史沿革与叙事。

---

## 1. 核心架构与边界约束

1. **宿主与插件两层分工**（详见 [docs/container-and-plugin-boundary-spec.md](docs/container-and-plugin-boundary-spec.md)）：
   - **宿主 Fork 层（`vLLM-HUST/vllm-ascend-hust`）**：负责模型私有前向修改、权重加载后处理 Hook（如 W3F2、D4）和底层 C++ 修复，随镜像原生发布。
   - **插件层（本仓）**：仅通过官方公开接口（`vllm.general_plugins`）注入纯 Python / Triton 算子，严禁在仓内保留任何就地修改容器运行树的动态打补丁脚本。
2. **默认关闭（default-off）**：
   - 所有算子与优化能力必须由环境变量明确门控，默认状态与原生行为严格一致、零开销、零副作用。
3. **元数据发现延迟导入（Lazy Import）**：
   - 插件顶级模块暴露必须采用延迟导入，保证在包元数据发现和生命周期扫描期间，绝不触发重型实现模块的加载。
4. **外部知识库与原始数据物理隔离**：
   - 外部知识库（`/root/cannbot-knowledge`、`/root/research`）及大体积性能分析 Trace / 数据库（`/root/prof/`）严禁提交进 Git 仓库。
   - 仓内仅沉淀标准算子三件套（`ops/<算子>/README.md`、`REPORT.md`、`evidence.json`）及根目录单一事实源索引 `DELIVERY_INDEX.json`。
5. **容器工作目录纪律**：
   - 严禁以 `/vllm-workspace` 为 CWD 运行任何 Python 或测试命令，避免历史目录遮蔽已安装的真正 Python 包。

---

## 2. 算子开发与评测约束

1. **四级验证阶梯**（详见 [docs/operator-benchmark-standard.md](docs/operator-benchmark-standard.md) 与 [docs/npu-benchmark-execution-guide.md](docs/npu-benchmark-execution-guide.md)）：
   - `L0 纯算子微基准 -> L3 数值正确性护栏 -> L4 调用链路归因 -> 服务端到端实测`；
   - 数值失败（L3 未过）一票否决，严格保持 default-off；
   - 负收益（如当前 `fi_gelu`）或在噪声范围内（如 `fia_demask`）必须如实记录，绝不夸大或伪造加速比。
2. **可比性事实口径**：
   - 算子对比只认**形状事实**（M/K/N、专家频次、相位、调用点）与**权重标签**（模型/精度/格式），条件不同只比方向，不混算数字。
3. **注释与文档原则**：
   - 代码只写必需注释，不写“为什么这样改”的解释性叙事或历史沿革；
   - 文档只写技术规范、判据与约束，不写心路历程。

---

## 3. Agent 交互与输出规范（言行准则）

1. **一律使用通俗大白话**：
   - 无论是进度汇报、技术分析、方案梳理、PR 描述还是执行回执，必须一律使用简明平实、逻辑严密的**工程大白话**；
   - 严禁把用于内部记录的比喻、缩写或黑话代入对外交流，不说让人费解的行话。
2. **坚决杜绝隐晦黑话与内部暗语**：
   - 严禁使用诸如“载波”、“领地”、“点钉”、“husk”、“档2 载体”、“短跑”、“活守卫”等隐晦暗语；
   - 一律直接指出具体的**物理实体**：如具体环境变量名（`VLLM_HUST_*`）、涉及的代码文件相对路径、确切的 Git 分支名称或具体的 shell 测试命令。
3. **实事求是与如实汇报**：
   - 测试通过就是通过、未测试就是未测试、负收益或在噪声范围内就据实记录；
   - 绝不夸大局部微基准收益，严禁将算子耗时直接外推为服务端到端收益，不掩盖任何发现的问题。
4. **直奔主题、拒绝废话**：
   - 汇报直击核心结论与关键依据，给出清晰明确的下一步行动，不写冗长虚泛的铺垫、感叹或毫无意义的沿革叙事。

---

## 4. 常用验证命令

```bash
# 代码风格检查
ruff check .
ruff format --check .

# 仓内关键契约与规范测试
pytest -q tests/test_benchmark_standard.py tests/test_json_hygiene.py tests/test_document_custody.py

# 快速运行单算子测试
pytest -q tests/test_gate_fp32_plugin.py
pytest -q tests/test_fia_demask.py

# 全量 CPU 门禁套件
pytest -q -m "not host_tree"
```

---

## 5. 权威文档导航

详细技术规范与权威文档均位于 [docs/README.md](docs/README.md)，请按单一事实源原则查阅与维护。
