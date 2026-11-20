# GLOSSARY — 术语表

统一代码、文档与交付报告中的术语。流程见 [AGENTS.md](AGENTS.md)，测量与状态见 [DELIVERY_INDEX.json](DELIVERY_INDEX.json) 及各算子的 `evidence.json`。

维护时先查重，再补定义、判据和真实路径；同步受影响的引用。只收录影响实现或验收判断的项目术语，不重复记录阈值、测量结果和当前状态。

## 1. 术语

### 1.1 交付边界与运行时分层

| 术语 | 定义与判据 | 锚点 | 易混项 |
|---|---|---|---|
| 宿主（host） | 服务解释器实际加载的 vLLM core 与 vllm-ascend；须核验安装来源和提交。本仓对它只读，只经官方公开面接入。 | `docs/HOST_CONTRACT.md`；`tools/operator_tools/host_runtime.py` | 社区安装包、只读参考源码不等于 HUST 服务运行时 |
| 上游（upstream） | vLLM / vllm-ascend 社区及其 `main` 线。本仓只跟随 release 做版本同步，不向其提 PR。 | `docs/agent-guide.md` 的「不做上游 PR」条；`docs/architecture.md` §1 | 宿主 Fork |
| 宿主 Fork（host fork） | 被允许承载宿主侧改动的两个仓：`vLLM-HUST/vllm-ascend-hust` 与 `vLLM-HUST/vllm-hust`。 | `DELIVERY_INDEX.json` 的 `baseline.host_forks` | 上游社区仓库 |
| 基线分支（baseline branch） | 两个宿主 Fork 上定死的 `based/split-batch-baseline`；特性分支与 PR 的 Base 都必须是它。 | `DELIVERY_INDEX.json` 的 `baseline_branch`、`pr_target_branch` | `main`、`based/image-baseline` 不是当前基线 |
| 镜像锚点（image anchor） | 基线分支内容所对齐的容器镜像源码 commit，用来判断宿主代码是否漂移。 | `DELIVERY_INDEX.json` 的 `baseline_commit` | 镜像源码 commit，不是基线分支头 |
| 插件仓 / 插件层 | 本仓。只承载入口注册、纯 Python / Triton 实现、门控与评测产物。 | `src/vllm_ascend_split_batch/`、`pyproject.toml` | 插件包指发行物 |
| 宿主接缝（seam） | 宿主上可被外部稳定接入的具体位置：函数、方法、注册表或调用点。接缝清单以 `HOST_CONTRACT.md` 为准。 | `docs/HOST_CONTRACT.md` 锚点表 | callsite 指实际调用位置 |
| 宿主提案（host proposal） | 必须改宿主私有代码才能落地的能力，以独立提案目录交付，含补丁、元数据与回滚路径；不在运行时注入。 | `host-proposals/<id>/proposal_meta.json` | 运行时打补丁 |
| 运行时打补丁（禁止） | 在已安装宿主源码上就地改写文本/字节的做法。已整体废除，属禁止项。 | 守卫 `tests/test_runtime_patch_manifest_guard.py` | — |
| Overlay 层 / PVC 层 | 容器易失层（Pod 重建即清空）与持久化层（`/root` 下挂载）。 | `docs/container-and-plugin-boundary-spec.md` §2.1 | 与本地、远端无关 |
| 容器自愈（self-heal） | 规格切换后重装 sshd、恢复密钥权限、重新注册插件入口的固定流程。 | `docs/container-and-plugin-boundary-spec.md` §2.2 | — |

### 1.2 扩展、门控与回退

| 术语 | 定义与判据 | 锚点 | 易混项 |
|---|---|---|---|
| general plugin（插件入口） | `vllm.general_plugins` 下的一条注册，由该模块的 `load()` 决定做什么。 | `pyproject.toml` 的 `[project.entry-points."vllm.general_plugins"]` | extension bundle |
| extension bundle（扩展包） | `vllm_hust.extension_bundles` 下的一条注册，对应一个带 manifest 的描述符目录。 | `pyproject.toml` 的 `extension_bundles`；`src/**/vllm-hust-extension-v0.2.json` | general plugin |
| 插件载体（plugin carrier） | 承载某个能力的可发现单元（入口或 bundle）。"有载体"只等于可发现，不等于被调用。 | `DELIVERY_INDEX.json` 的 `layer: plugin_carrier` | 载体状态快照，见 §2 |
| `active` | 描述符状态：允许显式激活，管理器可启用该载体。不等于目标模型/形状一定调用，也不改变 default-off；真实命中另需 callsite 证据。 | manifest 的 `implementation[].status`；`docs/release.md` 的「状态纪律」段 | `import_only`；`active` 不等于已调用 |
| `import_only` | 描述符状态：可被发现与检查，但管理器不得激活，也没有执行面。 | manifest 的 `implementation[].status`；`tests/test_manifest.py` | 未启用是门控状态 |
| activation environment | enable 时真正注入的环境变量键值，不是说明文本；根 bundle 为空。 | manifest 的 `activation.environment` | 说明文本不是注入值 |
| 门控（env gate） | 用具体环境变量决定能力是否加载；未设置时不导入实现。 | 各插件的 `VLLM_HUST_*`；`docs/support-matrix.md` | MoE gate 权重、SiLU 门控分支，见 §2 |
| default-off | 未显式取启用值时，行为与原生严格一致：零导入、零副作用、零日志。 | `AGENTS.md` 的 default-off 约束条 | — |
| fail-open | 接缝缺失/改名/内部异常时告警一次并保留宿主原实现（插件侧包装用）。 | `docs/HOST_CONTRACT.md` 各组件的默认语义段 | fail-closed |
| fail-closed | 条件不满足时拒绝执行并保持关闭（规划预检与状态判定用）。不允许"形式上成功、实际退化为原生"。 | `src/vllm_ascend_split_batch/planner.py` 的 `precheck_reason` | fail-open |
| `host.version_range` | manifest 中对宿主版本的有界声明；范围之外管理器报 `incompatible`。 | manifest 的 `host.version_range`；`docs/release.md` §0.1 | — |
| 四项提案协议 | `runtime-key` / `split-context` / `graph-pool` / `split-executor` 四个尚未由宿主实现的接入面，当前不是可用 API。 | `docs/HOST_CONTRACT.md` | 提案不等于可用 API |

### 1.3 split-batch 核心域

| 术语 | 定义与判据 | 锚点 | 易混项 |
|---|---|---|---|
| split-batch（批切分） | 把一个前向批次按请求边界切成两片并分别补齐以少做 padding 的规划策略。本仓只做规划，不接管宿主调度与图回放。 | `src/vllm_ascend_split_batch/planner.py` | — |
| `dual_pad` / `dual_inplace` | `SplitBatchConfig.mode` 的两个合法取值；只有 `dual_pad` 会被预检放行。 | `planner.SplitBatchConfig` | — |
| 切分 plan（`DualPadPlan`） | 一次切分的完整描述：总 token 数、不切分时的补齐量、节省量与两个切片。 | `planner.DualPadPlan` | 评测计划 |
| 切片（`SplitSlice`） | 切片单元：请求区间、token 区间与补齐量。 | `planner.SplitSlice` | — |
| precheck / 拒绝码 | 规划之前的 fail-closed 判定，返回拒绝码（如 `speculative_decode_conflict`）；返回 `None` 才表示可规划。 | `planner.precheck_reason`；`tests/test_planner.py` | — |
| `padding_saved` | 切分相对不切分省下的补齐 token 数。 | `planner.DualPadPlan.padding_saved` | — |
| capture size | 图捕获档位；切分结果必须落在已声明的档位内，不得懒捕获无界集合。 | `planner._ceil_capture`；`docs/HOST_CONTRACT.md` 的 graph 段 | 请求批量 |
| uniform decode | 同批请求的解码步长一致。不一致直接拒绝（`non_uniform_decode`）。 | `planner.precheck_reason` | — |
| opaque core | 宿主内部不可见的融合子图（如 GDN core），外部无法在其内部挂载算子。 | `docs/HOST_CONTRACT.md` | — |
| GDN（Gated Delta Net） | Qwen3.5 hybrid 中的线性注意力层类型。 | `ops/gdn_*/`；宿主仓内 `$VLLM_ASCEND_REPO/vllm_ascend/ops/gdn.py` | — |
| MoE（Mixture of Experts） | 稀疏专家前向层；路由网络与 shared expert 是两类不同对象。 | `ops/moe_*/` | — |
| speculative decode / MTP | 投机解码及其 k+1 验证行形态。本仓一律 fail-closed，只能写成边界，不得写成支持。 | `planner.precheck_reason`；`README.md` 的 Speculative decoding boundary | 当前不支持 |
| `op_selector` | 算子实现选择骨架：按形状与场景挑候选，解析不到就回宿主实现。当前是接口与选择骨架，未接线真实算子路径，实现不在发布包内。 | `experiments/op_selector.py`；`docs/design/operator-dispatch.md` | 选择骨架不等于已集成算子 |

### 1.4 算子交付与评测

| 术语 | 定义与判据 | 锚点 | 易混项 |
|---|---|---|---|
| MOD | 在固定基础引擎上以明确开关集合启停的待测能力或组合；身份包含实现版本与依赖，发现入口不等于启用。 | `docs/mod-acceptance-standard.md`；`MOD_METADATA.json` | 单个插件入口、全部算子目录 |
| 组合服务 A/B | 冻结实际可用的优化集合，比较同一宿主下全部目标关闭与全部目标启用；须重验组合数值和各候选调用，不向单项分摊收益。 | `docs/mod-acceptance-standard.md`；`docs/mod-acceptance/evidence.json` | 方向记录不等于已实现候选；组合诊断不等于 MOD 合格 |
| MOD 最终验收 | 对冻结 MOD 组合分别检查测试执行完整性和各项目标，算子服务收益不能替代此验收。 | `docs/mod-acceptance/evidence.json`；`docs/mod-acceptance-standard.md` | 算子验收 |
| B0 / B1 | MOD 配对实验的关闭组与启用组；基础引擎及其他条件相同，网站参考组合不自动成为本仓身份。 | `docs/mod-acceptance-standard.md` | 不固定指某个历史 Engine |
| 同工况参考效率（η_ref） | 完成同工况必要工作的独立参考时间与被测时间之比；设备忙碌度和相对基线速率不是此指标。 | `docs/mod-acceptance-standard.md` | 单算子 MFU、设备忙碌度、基线加速比 |
| 算子（operator） | 在明确 callsite 上进行的等价替换或融合优化，以 `ops/<operator>/` 为一个交付单元。 | `ops/`；`DELIVERY_INDEX.json` 的 `operators[]` | — |
| callsite | 算子被真实调用处的宿主位置（文件、方法、调用点）。可达性只认 callsite 证据。 | `ops/<operator>/evidence.json` 的 `callsite` | 宿主接缝，见 §2 |
| 算子交付目录 | 一个算子的交付文件集合：必备四个文件 `README.md`、`REPORT.md`、`bench_l0.py`、`evidence.json`，加一枚由 `tools/operator_tools/custody.py` 生成的 `checksum.sha256` 指纹。 | `ops/README.md`；`tests/test_ops_delivery_policy.py` | — |
| layer | 算子的归属层：`host_fork`（宿主侧）、`plugin_carrier`（插件侧）、`rejected_exploration`（已否定的探索）。 | `DELIVERY_INDEX.json` 的 `layer` | `unclassified` 仅作占位，不能合入 |
| status / serving_status | 索引中的交付状态与服务结论标签。`historical_only` 表示已有历史测量但不能作为当前实现结论，`unexposed` 表示当前没有公开接入。 | `DELIVERY_INDEX.json` | 标签按字段原值记录 |
| 形状事实（shape facts） | 张量维度、批量、输入输出规模、执行相位、专家分布等可复核的执行条件。 | `ops/<operator>/evidence.json` 的 `shape_facts`；`docs/operator-benchmark-standard.md` §1 | — |
| 权重标签 | 模型名 + 精度格式的组合，是可比性事实的另一半。 | `ops/<operator>/evidence.json` 的 `environment.model` / `dtype` | 模型族名不能替代模型名与精度格式 |
| 单变量上下文 | 两次对比除被测项外全部固定：种子、采样、图模式、并发、负载。 | `docs/operator-benchmark-standard.md` §1.2 | — |
| 服务指标写法 | 对外统一写「首字耗时 (FTL)」「单字耗时 (POTL)」「生成速度 (GenRate)」。行业措辞 `TTFT`、`TPOT`、输出速率是同义写法，MOD 冻结协议与组织政策沿用后者；同一份文档内混排时须在同段声明对应关系，只留一套主写法。度量数字只落结构化 `evidence.json`，行文与标题不得「关键词加数字」同段。 | `docs/operator-benchmark-standard.md` 的服务收益段；`docs/mod-acceptance-standard.md` | 三项指标分别达标，不得互相抵消 |
| 数值护栏 | 按容差与原生实现比对；失败后关闭能力，不接受性能结论。 | `ops/<operator>/evidence.json` 的 `l3_numerical_guard` | — |
| l0_traces | `evidence.json` 里机械追加的 L0 计时入账数组：数字只来自测量 receipt 或 `bench_l0.py` 执行输出，禁止手抄改写；失败用例照原样保留。 | `tools/operator_tools/l0_trace_record.py`；`ops/<operator>/evidence.json` 的 `l0_traces` | 手写的 `l0_microbenchmark` 摘要不是 trace 记录 |
| 一票否决 | 数值不一致时立即关闭能力并保持 default-off，不再谈性能。 | `AGENTS.md` 的「数值失败一票否决」条；`docs/operator-benchmark-standard.md` §2 | — |
| all-kernel 分母 | 模型整体算子耗时，用来判断单算子净省时能否被服务观测到。 | `tools/operator_tools/serving_loop.py` 的 `parse-l4` | 不等于服务墙钟时间 |
| `within_noise` | 服务结论标签：同条件配对测量的组间差异落在组内波动之内（散文里也写作 "within noise"）。不构成收益。 | `DELIVERY_INDEX.json` 的 `serving_status` | 不构成收益 |
| 负结果 | 优化后更慢或无效的结论。必须保留并入索引，不得删除。 | `DELIVERY_INDEX.json` 的 `status` | 不能用待复测掩盖已有负结果 |
| `pending_outer_loop` | 无结论标签：外环还没跑完，服务侧没有可主张的结论。写它时证据里不得出现 `tested: true`，也不得同时给出服务收益。 | `DELIVERY_INDEX.json` 的 `serving_status`、`l4_attribution`；`tests/test_delivery_status_consistency.py` | L4 未完成时只能挂无结论标签 |
| `integrated_plugin_measured` | 归属加测量标签（索引 `status`）：能力由插件层承载、已公开接入且已有当前测量；收益结论仍只看 `serving_status`。 | `DELIVERY_INDEX.json` 的 `status` | 与 `layer: plugin_carrier` 配套，不与 `host_fork` 混用 |
| `integrated_host_fork_measured` | 归属加测量标签（索引 `status`）：能力已下沉到宿主 Fork 源码承载，插件侧只留交付目录与证据。 | `DELIVERY_INDEX.json` 的 `status` | 与 `layer: host_fork` 配套；不得把宿主实现归因给插件 |
| `superseded_by_host_fork` | 取代标签：该路线已被宿主 Fork 的等能力实现取代，不再是候选。必须写明被谁取代。 | `DELIVERY_INDEX.json` 的 `serving_status`、`superseded_by_host_fork`；`tests/test_delivery_status_consistency.py` | 取代不等于该项已测过 |
| cohort | 一组锁定的实验条件（宿主线、模型、请求形态、图配置、精度）。数字只在同一 cohort 内可比。 | `docs/cohort-and-ownership.md` §2 | 不同 cohort 的数字不能混比 |
| 负载口径（load caliber） | 某次采集绑定的负载标识，与 cohort 一起构成产物的生产者标记。 | `VLLM_HUST_LOAD_CALIBER_ID`、`VLLM_HUST_COHORT_ID`；`tools/analysis/matmulv2_shape_attribution.py` | — |
| 跨域外推（禁止） | 把某 cohort 的结论搬到另一 cohort。历史 Qwen2.5 / 0.25 与当前基准互不推导。 | `docs/cohort-and-ownership.md` §3 | — |
| 工作项代号（`D0`/`D4`/`D5`/`W1R`/`F2`/`W3F2`） | 历史工作单编号，只作检索标签，单独出现不构成术语。 | 例：`VLLM_HUST_D4_CONV_PACK` 中的 `D4` | 引用时附环境变量、路径或分支 |

### 1.5 算子优化与服务验证

| 术语 | 定义与判据 | 锚点 | 易混项 |
|---|---|---|---|
| 单算子优化 | Agent 按真实输入合同进行数值核对、采集、单点修改与重复计时。 | `.openbitfun/skills/operator-sol-optimizer/SKILL.md` | — |
| 外环（outer loop） | 整网对拍、逐卡调用归因与服务验证；单项 A/B 按需选测，交付组合须配对测量。 | `tools/operator_tools/serving_loop.py`；`workloads/run_workload.sh` | MOD 最终验收 |
| SOL 达成率（SOL%） | 匹配资源能力下的必要工作时间下界与同范围设备时间之比；能力与缓存假设必须可核查。 | `.openbitfun/skills/operator-sol-optimizer/scripts/sol.py` | MFU、MOD 参考效率 |
| 资源时间下界 | 必要工作量除以匹配的资源能力；跨资源取最大值只给出充分重叠假设下的乐观界。 | `.openbitfun/skills/operator-sol-optimizer/SKILL.md` | 可消除的中间流量不属于必要工作 |
| 优化空间上界 | 由有效下界推得的最多可省比例与最大倍率，不代表一定可实现。 | `.openbitfun/skills/operator-sol-optimizer/scripts/sol.py` | 不等于预期收益 |
| 形状桶 | 必须覆盖的固定形状集合，与实际执行相位和图档位对应。 | `.openbitfun/skills/operator-sol-optimizer/SKILL.md` | — |
| 动态形状探针 | 未参与预热编译的尺寸；编译缓存或日志用于核查是否重编译。 | `.openbitfun/skills/operator-sol-optimizer/SKILL.md` | 单次时间不能证明重编译 |
| 短路评估 | 数值检查失败后不接受该候选的性能结论。 | `.openbitfun/skills/operator-sol-optimizer/SKILL.md` | — |
| 泳道图 | 由 timeline 折叠出的高层级子图片段可视化，供人工审阅。 | `tools/operator_tools/timeline_insight.py` | 不能自动推定计算语义 |
| SNR 预筛 | 用算子份额估算是否值得投入服务实验，不构成服务收益判定。 | `tools/operator_tools/serving_loop.py` | 全核份额不等于墙钟关键路径 |
| 负载（workload） | 主测、动态压力和跨相位诊断输入，按算子生效条件配对。 | `tools/operator_tools/workloads.py`；`workloads/README.md` | MOD 冻结任务 |
| `primary_workload` / `stress_workload` | 主测对比负载与动态形状压力负载，缺一即证据不完整。 | `workloads/README.md` | — |
| 载体状态快照（`carriers.env`） | 一次起服窗口里被核验的开关与入口点状态集合，唯一来源是与 serve 日志同名的环境记录。 | `tools/bench-ladder.sh` | 插件载体 |

### 1.6 证据保管与溯源

| 术语 | 定义与判据 | 锚点 | 易混项 |
|---|---|---|---|
| `evidence.json` | 机器可读的验收凭据；没有凭据的结论不得进索引。 | `ops/<operator>/evidence.json` | — |
| `operator-benchmark-evidence/v1` | `evidence.json` 的 schema 标识；改字段等于改 schema。 | `docs/operator-benchmark-standard.md` §4 | — |
| `DELIVERY_INDEX.json` | 全仓算子交付的权威事实清单；新增或改状态必须登记。 | `DELIVERY_INDEX.json`；`tests/test_json_hygiene.py` | — |
| 证据指纹（`checksum.sha256`） | 目录内文件的 SHA-256 清单，用于校验文件内容和覆盖范围。 | `ops/*/checksum.sha256` | — |
| custody | 生成与校验指纹的工具契约。 | `tools/operator_tools/custody.py` | — |
| provenance（溯源） | 从历史实现或提交提取代码时的原貌保存：版权头、commit、只读。 | `provenance/`；`docs/PROVENANCE.md` | — |
| CPU 门禁 | 不接 NPU 也能跑的测试集；纯逻辑不依赖设备包，可选张量测试缺包时跳过；`host_tree` 排除的宿主源码与入口集成在显式选择后缺依赖即失败。 | `docs/operator-benchmark-standard.md` §6；`pyproject.toml` 的 `markers` | 与统一离线交付检查的范围不同 |
| `MOD_METADATA.json` | 仓库身份、责任人、default-off、回滚与证据限定。 | `MOD_METADATA.json` | — |

## 2. 易混术语

| 词 | 上下文甲 | 上下文乙 | 必须怎么写 |
|---|---|---|---|
| `gate` | 门控开关（env gate） | MoE 路由网络/权重（`gate_fp32` 里的 `gate`）；GDN 的 SiLU 门控分支（`gdn_norm_gate`） | 指开关写"门控 / env gate"；指网络写"MoE gate 权重"；指分支写"SiLU 门控分支" |
| `L3` / `L4` | 算子阶梯：数值护栏、调用链路归因（算子粒度） | `tools/bench-ladder.sh` 服务流程：贪心对拍、profile 窗口（服务粒度） | 写"算子阶梯 L3/L4"或"ladder L3/L4" |
| `L1` / `L2` | ladder 专有档位（单请求、并发突发） | 算子阶梯**没有**这两档 | 不得说"算子 L1/L2" |
| `carrier` | 插件载体（可发现单元） | ①早期 Cascade kernel 发行物（已废止）；②ladder 起服窗口里的载体状态集合（记录轴，不是门禁） | 说清是哪一种：插件载体 / ladder 载体状态 |
| `cohort` vs 基准 | cohort 是条件集合 | "基准"是本仓当前选定的那套 cohort | 说"基准"必须能展开成 cohort 表 |
| `seam` vs `callsite` | seam 是"能接的位置" | callsite 是"实际被调的位置" | 可达性证据只认 callsite |
| `plan` | 切分 plan（`DualPadPlan`） | 评测执行计划（`serving_loop plan`、外环计划） | 写"切分 plan"或"评测计划" |
| `evidence` | 仓内 `evidence.json` | 仓外 benchmark 归档 | 写"仓内凭据"或"外部归档" |
| `插件` | general plugin 入口 | extension bundle | 写"入口"或"bundle" |
| `kernel` | 设备侧核函数 | 独立交付的编译型算子包（kernel wheel） | 写"核函数"或"kernel wheel" |

## 3. 废止术语

| 废止词 | 曾经的指代 | 现在怎么说 | 依据 |
|---|---|---|---|
| Cascade / `cascade-attention` | 已撤除的注意力实现及其入口 | "已撤除实现"，不称能力、不称收益 | `docs/HOST_CONTRACT.md` 的 Cascade 接入边界段 |
| `apply-runtime-patches.sh`、`runtime_source_patches` | 运行时打补丁链 | 宿主提案（host proposal） | `docs/container-and-plugin-boundary-spec.md` §3.5 |
| `based/image-baseline` | 旧基线名 | `based/split-batch-baseline` | `AGENTS.md` 的「宿主改动走宿主 Fork 基线分支」条 |
| `tier-2` | 需要改宿主私有代码才能落地的宿主侧补丁（早期分层编号叫法） | 宿主提案（host proposal） | `docs/container-and-plugin-boundary-spec.md` §3.5。注意 `host-proposals/W1R-gmm-tuning-config/patch.diff` 与 `gen_patch.py` 里的 `tier-2 host proposal` 是补丁标记字符串，由 `tests/test_w1r_gmm_tuning_patch.py` 逐字锁定，不随行文改名 |
| `gdn-qkv-norm` plugin / bundle | 已撤回注册的入口 | 注册已撤回、L0 测量保留待外环；路线转宿主提案 `host-proposals/GDN-qkv-norm/` | `README.md` 的 GDN QKV norm 段；`DELIVERY_INDEX.json` 的 `gdn_qkv_norm` 条目 |
| `Raing5Days/vllm-hust-cascade-kernel` | 早期独立发行的 Cascade kernel 包与其个人账号归属 | 历史专项，仅引用不依赖；org 归属未定 | `docs/cohort-and-ownership.md` §1 归属表 |
| 历史 Qwen2.5 / 0.25 结论 | 可外推的收益结论 | 历史独立专项，禁止跨域外推 | `docs/cohort-and-ownership.md` §2 |
