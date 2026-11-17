# 算子优化执行工作流

本文件补充 [SKILL.md](../SKILL.md) 的执行顺序。术语和状态标签按 [GLOSSARY.md](../../../GLOSSARY.md)、`DELIVERY_INDEX.json` 和 `ops/<operator>/evidence.json`。本文件不改变验收判据。

## 1. 状态分层

多个算子可以同时处于单侧状态。端侧只用一个长测窗口批量验证，不为每个算子重复起服。

| 状态 | 判据 | 允许动作 | 禁止动作 |
|---|---|---|---|
| 方向记录 | 只有优化方向，无可验收代码 | 保留记录，说明缺项 | 建立交付目录或写性能结论 |
| 单侧候选 | 有实现、L0 和 L3；形状合同可复现；缺可验证 callsite 或端侧结果 | 继续单点优化、补宿主接入、等待端侧长测 | 写服务收益标签 |
| 单侧已验证 | 有 L0、L3、逐卡或等价调用证据，且门控可达 | 进入端侧长测计划 | 把单侧收益写成服务收益 |
| 端侧已测 | 有服务 A/B、主测结论、噪声记录和原始归档 | 按证据写 `within_noise`、`verified_faster` 或负结果 | 把组合收益分摊给单项 |
| 历史或失败 | 宿主改写、数值失败、归因失败或旧版本结论 | 保留历史，标明适用范围 | 复用为当前收益 |

状态只按证据推进。时间顺序、代码存在、注册入口存在都不能替代判据。

## 2. 算子挑选

1. 从 `DELIVERY_INDEX.json`、目标源码和现有 `ops/<operator>/` 读取当前状态。
2. 只选择有明确 callsite、真实形状合同、default-off 门控和可实现证据的候选。
3. 排除方向记录、已拒绝、已替代、缺失公开入口且无法补宿主接入的候选。
4. 按全核份额、关键路径风险、数值风险和归因难度排序。SNR 只作优先级依据。
5. 输出候选清单和排除清单。排除原因必须能复核，不写“暂不处理”。

### 实现形态判据

1. 插件库是 Python 包。它只发布 `src/vllm_ascend_split_batch`。
2. 插件库不构建、不携带、不加载自定义 `.so`。
3. Catlass、CANN C 和 Ascend C 编译型 kernel 只能通过外部独立 wheel，或宿主已注册接口接入。
4. 插件 Python 代码可以调用已安装的 `torch.ops.*` 或宿主公开 wrapper。
5. 没有外部 wheel 或宿主接口时，不得把 Catlass/CANN C 当作本插件库可直接实现的方案。

## 3. 单侧验证

每个单侧候选独立维护 L0、L3 和 L4 或等价调用证据。

1. 先核验宿主解释器和提交，再读取真实输入合同。
2. 用 `scripts/sol.py` 建立资源下界，只把上界当优化空间参考。
3. 用 `scripts/profile.py` 的 preset 生成采集命令；正式执行前核对当前帮助。
4. 直接调用 `collect_guard(config, output, arm)` 时，profile 产物在 `output/profile`，不是 `output/<arm>/profile`。热点汇总必须读取实际写入 CSV 的目录。
5. 热点表由 `op_statistic.csv` 重算：按 `OP Type` 合并 `Count` 与 `Total Time(us)`，占比用合并后的总时间作分母。空热点表必须先检查 CSV 路径，不得当成“无热点”。
6. 数值检查通过后才接受计时结果。失败时停止该候选的性能升级。
7. 缺逐卡归因时保持单侧候选，不进入端侧已测。
8. 多个候选可以并行处于单侧状态；每个候选保留独立哈希、命令、退出码和产物路径。

### 工具缺失时 fail-closed

`operator-sol-optimizer` 在正文中可能引用 `triton-op-verifier`、`triton-mfu-analyzer`、`triton-precision-debug`、`triton-simulator-optimizer` 等技能。使用前必须确认其 `SKILL.md` 存在。声明的脚本也必须存在。

如果 `triton-op-verifier` 缺少 `SKILL.md`、`verify.py` 或 `benchmark.py`，不得声称完成功能验证或基准验证。此时只能：

- 保留 CPU 逻辑检查和已有测试；
- 把设备数值、设备计时和基准标为未执行；
- 记录缺失文件和影响范围；
- 不把公共工具脚本当作完整 skill。

任何外部 skill 不可用时都按同样规则处理。

## 4. 端侧长测

端侧统一使用长测工具，不按算子手写重复命令。

1. 在干净分支跑 `pytest @tests/offline-checks.args` 和 `repo_audit.py`。
2. 确认计划只含未完成端侧验证、且具备可验证门控的算子。已测、失败未修、已替代和方向记录不得混入。
3. 使用 `tools/operator_tools/pending_longrun.py` 生成计划。无调用规则的算子只能 guard-only 或 skip。
4. 使用 `workloads/run_pending_longrun.sh` 后台启动。原始归档放仓外，日志和 checkpoint 放 `.openbitfun/`。
5. 长测计划必须使用 `sync_mode: none`。结果不自动写回正式证据。
6. 人工核对 host preflight、贪心对拍、逐卡归因、请求完整性和噪声记录后，才能调用同步工具。

## 5. 同步与状态更新

1. 只同步通过 `serving_sync.validate` 的归档。
2. 数值或归因失败时保留失败归档和原因，不写服务收益标签。
3. 历史结果写 `applicability: historical`，索引用对应历史标签。
4. 更新 `evidence.json` 后同步 `DELIVERY_INDEX.json`，最后重新生成 `checksum.sha256`。
5. 再跑离线检查、风格检查和 `repo_audit.py`。任一失败即停止交付。

## 6. 停止条件

出现以下任一情况时停止当前候选：

- 宿主身份或安装来源不匹配；
- 数值检查失败；
- callsite 不可达或逐卡归因无法成立；
- 必需 skill 或脚本缺失却要求声称验证通过；
- 归档缺请求数、退出码、命令或命中计数；
- 差异落在有效噪声内仍要求写收益；
- 组合结果要求分摊给单个算子。

停止时保留原始记录，写明真实原因和下一轮前置条件。
