+# AGENTS.md — 工作契约
+
+本文件只定义必须遵守的约束。操作流程见 [workflow.md](workflow.md)。统一语言以 [GLOSSARY.md](GLOSSARY.md) 为准。权威规范见 [docs/README.md](docs/README.md)。
+
+## 1. 边界
+
+1. 宿主层与插件层分工必须符合 [容器与插件边界规范](docs/container-and-plugin-boundary-spec.md)。插件层只使用官方公开接口。
+2. 禁止运行时修改宿主环境，禁止维护就地补丁脚本。宿主改动必须走宿主 Fork 的 `based/split-batch-baseline`。
+3. 服务采集配置必须包含两个宿主的完整 `host_commits`，并由 `tools/operator_tools/host_runtime.py` 在服务解释器内核验。缺失或不匹配即停止。
+4. 算子与优化能力必须 default-off。未启用时与原生行为一致，零导入、零副作用。
+5. 插件对外暴露必须惰性加载。元数据发现期间不得触发重型实现模块导入。
+6. 原始数据、Trace、调试数据库和临时产物不得提交。仓内只保留结构化证据、报告和索引。
+7. 原始采集放仓外；临时修改放仓库内隔离目录。执行环境不得被本地源码遮蔽。
+
+## 2. 证据与结论
+
+1. 算子评测按 [算子基准规范](docs/operator-benchmark-standard.md) 执行。数值失败一票否决。负收益和噪声内结果如实记录。
+2. 对比只使用形状事实和权重标签。不同条件只比方向，不混算数字。
+3. `DELIVERY_INDEX.json` 的结论标签必须与 `ops/<operator>/evidence.json` 同向。标签不得超出证据。
+4. L3 失败时不得写服务收益标签。L4 未完成时不得给出服务结论。历史结果必须使用 `historical_only`。
+5. MOD 验收按 [MOD 最终验收规范](docs/mod-acceptance-standard.md) 执行。算子服务结论不能替代 MOD 验收。
+6. 修改 `ops/<operator>/` 被 `checksum.sha256` 覆盖的文件后，必须重新生成指纹。
+
+## 3. 文档与行文
+
+1. 文档只写技术规范、判据和约束。不写沿革和心路。代码只写必需注释。
+2. Markdown 中的插件数、bundle 数必须与 `pyproject.toml` 一致。
+3. `ops/<operator>/` 必须齐备 `README.md`、`REPORT.md`、`bench_l0.py`、`evidence.json`。
+4. `GLOSSARY.md` 词条四列完整、术语不重复、锚点真实，且必须被本文件和 `docs/README.md` 引用。
+5. 对外文档不得出现“关键词 + 数字”同段。度量数字写入结构化 `evidence.json`。
+6. 新增对外规范文档时，必须登记进 `tests/test_doc_consistency.py` 的 `PROSE_SURFACE`。
+7. 使用工程大白话。禁止使用“载波”“领地”“点钉”“husk”“档2 载体”“短跑”“活守卫”等暗语。
+8. 一律使用 [GLOSSARY.md](GLOSSARY.md) 的唯一写法。新概念先登记词条和锚点，再改代码与文档。
+
+## 4. 提交前检查
+
+```bash
+ruff check .
+ruff format --check .
+pytest @tests/offline-checks.args
+pytest -q -m "not host_tree"
+```
+
+CPU 门禁必须使用隔离虚拟环境、清空 `PYTHONPATH`，并执行 CI 使用的 pytest 命令。
+
+## 5. 凭据
+
+GitHub API 凭据只能来自进程变量或用户指定的环境文件。不得提交、打印、写入日志或写入仓库。
