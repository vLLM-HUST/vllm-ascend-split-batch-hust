# GLOSSARY — 统一语言与术语契约

本文件是本仓库的统一语言（Ubiquitous Language）词表，与 [AGENTS.md](AGENTS.md) 同级：
`AGENTS.md` 管"怎么做事"，本文件管"这个东西叫什么、指什么"。

## 0. 定位与效力

| 本文件负责 | 本文件不负责 |
|---|---|
| 术语的唯一写法、定义判据、代码锚点、禁用别名 | 阈值、流程、阶梯细则（→ [docs/operator-benchmark-standard.md](docs/operator-benchmark-standard.md)） |
| 同一个词在不同限界上下文中的含义分派 | 容器操作步骤与环境路径（→ [docs/container-and-plugin-boundary-spec.md](docs/container-and-plugin-boundary-spec.md)） |
| 废止术语的墓碑，防止旧词回流 | 交付状态与收益数字（→ [DELIVERY_INDEX.json](DELIVERY_INDEX.json)、[docs/delivery-inventory.md](docs/delivery-inventory.md)） |

三条效力规则：

1. **词义以本文件为准**。代码标识符、文档标题、PR 描述、提交信息、Agent 汇报里用到的术语与本文冲突时，以本文为准，冲突处必须改。
2. **本文件不承载事实**。数字、版本、阈值、状态值只允许写在权威事实源里，本文只给指针；本文不得成为第二事实源。
3. **词条必须有锚点**。找不到代码、字段或文件锚点的词不是本仓领域术语，不进本文（判据见 §1.2）。

---

## 1. DDD 规则在本仓的落地版

### 1.1 采用的概念

| DDD 概念 | 原意 | 本仓怎么落地 |
|---|---|---|
| 统一语言（Ubiquitous Language） | 领域专家与开发者共用同一套词，嘴、文档、代码同名 | 本文就是那套词；词条必须同时给出聊天用词、文档用词、代码锚点 |
| 限界上下文（Bounded Context） | 同一个词在不同上下文可以指不同东西，边界内必须一义 | §2 列出 6 个上下文；跨上下文同名一律进 §4 消歧表 |
| 领域术语 vs 技术术语 | 只有承载领域判断的词才值得进词表 | 通用编程词与框架词不收（判据见 §1.2 第 4 条） |
| 术语生命周期 | 词会随模型演进被引入、改写、废止 | §5 保留墓碑，废止词不得复活复用 |
| 模型与代码同源校验 | 术语必须能在模型/代码里指到实物 | 每个词条强制填锚点；锚点失效即词条失效 |

### 1.2 收录判据（四条全中才收）

1. **稳定出现**：在对话、文档、代码中至少两处被反复使用，不是一次性措辞。
2. **有唯一锚点**：能指向具体路径、类/函数、JSON 字段、环境变量或测试文件。
3. **影响判断**：读懂这个词会改变某人的动作（接哪个缝、判是否通过、能不能声称收益）。
4. **不是通用词**：`list`、`forward`、`pytest`、CI 这类词不收；只收本仓自造或本仓专义的词。

### 1.3 词条格式（六项）

| 项 | 要求 |
|---|---|
| 术语 | 唯一写法，含必须保留的英文标识符；出现在代码里时用反引号 |
| 定义（判据） | 一到两句，必须可被反驳；写"是什么、什么时候算、什么时候不算"，不写比喻 |
| 锚点 | 相对路径 / 字段名 / 环境变量 / 测试文件；必须真实存在 |
| 禁用或易混写法 | 明确不接受的同义词；空着就写 `—` |
| 上下文归属 | 由所在小节决定（§3.1–§3.6） |
| 废止标记 | 进 §5，不进 §3 |

### 1.4 新增与修改流程

1. **定上下文**：先判断这个词属于 §2 的哪个上下文；跨两个以上上下文说明它不是术语，而是同名异义（进 §4）。
2. **查重**：在本文搜索同义与同名；已有词条能覆盖就改旧词条，禁止并列造新词。
3. **写判据**：一句话说清"算什么、不算什么"；写不出反例说明这个词还没定型，先不进表。
4. **填锚点并自查**：路径/字段/环境变量必须能直接打开或 `grep` 到；随后同步所有引用处（文档、注释、`DELIVERY_INDEX.json` 标签）。
5. **同步契约**：只有行为规则变化才改 `AGENTS.md`；术语层变化只改本文，`AGENTS.md` 只保留一条指向本文的指针，不复制定义。

### 1.5 红线（违反即返工）

- 不造同义词：同一概念换一个词写，等于制造两套语言。
- 不复活废止词：§5 的词出现在新文档、新代码、PR 标题里即返工。
- 不写比喻与内部暗语：`AGENTS.md` §3.2 的禁词表见 §5.2。
- 不把词条写成设计文档：定义超过三行说明它该拆成词条 + 权威文档。
- 不在本文件写"关键词 + 数字"同段：组织基准数据政策对此 fail-closed（判据见 `AGENTS.md` §2.4）。
- 不在本文件写绝对路径、凭据或内网地址（`tests/test_public_hygiene.py` 会拦）。

---

## 2. 限界上下文地图

| 编号 | 上下文 | 判定边界（什么算这个上下文的事） | 权威文档 |
|---|---|---|---|
| C1 | 交付边界与运行时分层 | 代码放在哪一层、能不能改宿主、证据放哪 | [docs/container-and-plugin-boundary-spec.md](docs/container-and-plugin-boundary-spec.md)、[docs/HOST_CONTRACT.md](docs/HOST_CONTRACT.md) |
| C2 | 扩展、门控与回退 | 能力怎么被发现、怎么被打开、出错往哪退 | [docs/architecture.md](docs/architecture.md)、[docs/support-matrix.md](docs/support-matrix.md) |
| C3 | split-batch 核心域 | 切分规划本身的语义 | [docs/architecture.md](docs/architecture.md)、[docs/HOST_CONTRACT.md](docs/HOST_CONTRACT.md) |
| C4 | 算子交付与评测 | 一个算子怎么算交付完、结论怎么才成立 | [docs/operator-benchmark-standard.md](docs/operator-benchmark-standard.md) |
| C5 | 调优 Harness（内外环） | 自动调优与整网评测怎么跑 | [docs/op-agent-harness-spec.md](docs/op-agent-harness-spec.md) |
| C6 | 证据保管与溯源 | 结论凭什么可信、凭什么可追溯 | [docs/EVIDENCE.md](docs/EVIDENCE.md)、[docs/PROVENANCE.md](docs/PROVENANCE.md) |

跨上下文借义是禁止的：callsite 只能按 C4 的定义用，`import_only` 只能按 C2 的定义用。

---

## 3. 词条

### 3.1 C1 交付边界与运行时分层

| 术语 | 定义（判据） | 锚点 | 禁用或易混写法 |
|---|---|---|---|
| 宿主（host） | 已安装的 vLLM core 与 vllm-ascend 源码及其运行时。本仓对它只读，只经官方公开面接入。 | `docs/HOST_CONTRACT.md`；环境变量 `$VLLM_ASCEND_REPO`、`$VLLM_HUST_REPO` | 不要用"上游"指宿主 |
| 上游（upstream） | vLLM / vllm-ascend 社区及其 `main` 线。本仓只跟随 release 做版本同步，不向其提 PR。 | `AGENTS.md` §1.7；`docs/agent-guide.md` §1.6 | 不要用"上游"指宿主 Fork |
| 宿主 Fork（host fork） | 被允许承载宿主侧改动的两个仓：`vLLM-HUST/vllm-ascend-hust` 与 `vLLM-HUST/vllm-hust`。 | `DELIVERY_INDEX.json` 的 `baseline.host_forks` | 不要写"我们的 vllm" |
| 基线分支（baseline branch） | 两个宿主 Fork 上定死的 `based/split-batch-baseline`；特性分支与 PR 的 Base 都必须是它。 | `DELIVERY_INDEX.json` 的 `baseline_branch`、`pr_target_branch` | `main`、`based/image-baseline` 均不是基线 |
| 镜像锚点（image anchor） | 基线分支内容所对齐的容器镜像源码 commit，用来判断宿主代码是否漂移。 | `DELIVERY_INDEX.json` 的 `baseline_commit` | 不要再叫"基线 commit"，避免与分支头混淆 |
| 插件仓 / 插件层 | 本仓。只承载入口注册、纯 Python / Triton 实现、门控与评测产物。 | `src/vllm_ascend_split_batch/`、`pyproject.toml` | 不要与"插件包"混用指发行物 |
| 宿主接缝（seam） | 宿主上可被外部稳定接入的具体位置：函数、方法、注册表或调用点。接缝清单以 `HOST_CONTRACT.md` 为准。 | `docs/HOST_CONTRACT.md` 锚点表 | 不要用"钩子"泛指接缝 |
| 宿主提案（host proposal） | 必须改宿主私有代码才能落地的能力，以独立提案目录交付，含补丁、元数据与回滚路径；不在运行时注入。 | `host-proposals/<id>/proposal_meta.json` | 不要叫"运行时补丁" |
| 运行时打补丁（禁止） | 在已安装宿主源码上就地改写文本/字节的做法。已整体废除，属禁止项。 | 守卫 `tests/test_runtime_patch_manifest_guard.py` | — |
| Overlay 层 / PVC 层 | 容器易失层（Pod 重建即清空）与持久化层（`/root` 下挂载）。 | `docs/container-and-plugin-boundary-spec.md` §2.1 | 不要用"本地/远端"描述这两层 |
| 容器自愈（self-heal） | 规格切换后重装 sshd、恢复密钥权限、重新注册插件入口的固定流程。 | 同上 §2.2 | — |

### 3.2 C2 扩展、门控与回退

| 术语 | 定义（判据） | 锚点 | 禁用或易混写法 |
|---|---|---|---|
| general plugin（插件入口） | `vllm.general_plugins` 下的一条注册，由该模块的 `load()` 决定做什么。 | `pyproject.toml` 的 `[project.entry-points."vllm.general_plugins"]` | 不要写成"扩展" |
| extension bundle（扩展包） | `vllm_hust.extension_bundles` 下的一条注册，对应一个带 manifest 的描述符目录。 | `pyproject.toml` 的 `extension_bundles`；`src/**/vllm-hust-extension-v0.2.json` | 不要与插件入口互换称呼 |
| 插件载体（plugin carrier） | 承载某个能力的可发现单元（入口或 bundle）。"有载体"只等于可发现，不等于被调用。 | `DELIVERY_INDEX.json` 的 `layer: plugin_carrier` | 不要裸写"载体"，三种含义见 §4 |
| `import_only` | 描述符状态：可被发现与检查，但管理器不得激活，也没有执行面。 | manifest 的 `implementation[].status`；`tests/test_manifest.py` | 不要写成"未启用"，那是门控语义 |
| activation environment | enable 时真正注入的环境变量键值，不是说明文本；根 bundle 为空。 | manifest 的 `activation.environment` | 不要写成"环境配置" |
| 门控（env gate） | 用具体环境变量决定能力是否加载；未设置时不导入实现。 | 各插件的 `VLLM_HUST_*`；`docs/support-matrix.md` | 与 MoE `gate` 混用见 §4 |
| default-off | 未显式取启用值时，行为与原生严格一致：零导入、零副作用、零日志。 | `AGENTS.md` §1.2 | 不要写成"默认关闭的试验" |
| fail-open | 接缝缺失/改名/内部异常时告警一次并保留宿主原实现（插件侧包装用）。 | `docs/HOST_CONTRACT.md` 各组件的默认语义段 | 不要与 fail-closed 互换 |
| fail-closed | 条件不满足时拒绝执行并保持关闭（规划预检与状态判定用）。不允许"形式上成功、实际退化为原生"。 | `src/vllm_ascend_split_batch/planner.py` 的 `precheck_reason` | 不要写成"安全模式" |
| `host.version_range` | manifest 中对宿主版本的有界声明；范围之外管理器报 `incompatible`。 | manifest 的 `host.version_range`；`docs/release.md` §0.1 | 不要写 `>=0` 这类无界声明 |
| 四项提案协议 | `runtime-key` / `split-context` / `graph-pool` / `split-executor` 四个尚未由宿主实现的接入面，当前不是可用 API。 | `docs/HOST_CONTRACT.md` | 不要写成"宿主接口" |

### 3.3 C3 split-batch 核心域

| 术语 | 定义（判据） | 锚点 | 禁用或易混写法 |
|---|---|---|---|
| split-batch（批切分） | 把一个前向批次按请求边界切成两片并分别补齐以少做 padding 的规划策略。本仓只做规划，不接管宿主调度与图回放。 | `src/vllm_ascend_split_batch/planner.py` | 不要写成"分服务"或"双流" |
| `dual_pad` / `dual_inplace` | `SplitBatchConfig.mode` 的两个合法取值；只有 `dual_pad` 会被预检放行。 | `planner.SplitBatchConfig` | — |
| 切分 plan（`DualPadPlan`） | 一次切分的完整描述：总 token 数、不切分时的补齐量、节省量与两个切片。 | `planner.DualPadPlan` | 不要与评测计划混称（见 §4） |
| 切片（`SplitSlice`） | 切片单元：请求区间、token 区间与补齐量。 | `planner.SplitSlice` | — |
| precheck / 拒绝码 | 规划之前的 fail-closed 判定，返回拒绝码（如 `speculative_decode_conflict`）；返回 `None` 才表示可规划。拒绝码本身就是术语，不得改写同义码。 | `planner.precheck_reason`；`tests/test_planner.py` | 不要写成"参数校验" |
| `padding_saved` | 切分相对不切分省下的补齐 token 数。 | `planner.DualPadPlan.padding_saved` | — |
| capture size | 图捕获档位；切分结果必须落在已声明的档位内，不得懒捕获无界集合。 | `planner._ceil_capture`；`docs/HOST_CONTRACT.md` 的 graph 段 | 不要写成"batch 档位" |
| uniform decode | 同批请求的解码步长一致。不一致直接拒绝（`non_uniform_decode`）。 | `planner.precheck_reason` | — |
| opaque core | 宿主内部不可见的融合子图（如 GDN core），外部无法在其内部挂载算子。它是"不可达"的判据。 | `docs/HOST_CONTRACT.md` | 不要写成"黑盒层" |
| GDN（Gated Delta Net） | Qwen3.5 hybrid 里的线性注意力层类型；本仓多数算子与它相关。 | `ops/gdn_*/`；宿主仓内 `$VLLM_ASCEND_REPO/vllm_ascend/ops/gdn.py` | 不要展开成"门控 delta 网络" |
| MoE（Mixture of Experts） | 稀疏专家前向层；路由网络与 shared expert 是两类不同对象。 | `ops/moe_*/` | — |
| speculative decode / MTP | 投机解码及其 k+1 验证行形态。本仓一律 fail-closed，只能写成边界，不得写成支持。 | `planner.precheck_reason`；`README.md` 的 Speculative decoding boundary | 不要写"已支持 MTP" |
| `op_selector` | 算子实现选择骨架：按形状与场景挑候选，解析不到就回宿主实现。当前是接口与选择骨架，未接线真实算子路径。 | `src/vllm_ascend_split_batch/op_selector.py`；`docs/design/operator-dispatch.md` | 不要写成"已集成调度" |

### 3.4 C4 算子交付与评测

| 术语 | 定义（判据） | 锚点 | 禁用或易混写法 |
|---|---|---|---|
| 算子（operator） | 一次有明确 callsite 的等价替换或融合优化，以 `ops/<operator>/` 为一个交付单元。 | `ops/`；`DELIVERY_INDEX.json` 的 `operators[]` | 不要用"优化点"泛指 |
| callsite | 算子被真实调用处的宿主位置（文件、方法、调用点）。可达性只认 callsite 证据。 | `ops/<operator>/evidence.json` 的 `callsite` | 不要与"接缝"互换（见 §4） |
| 算子交付目录 | 一个算子的自洽目录：`README.md`、`REPORT.md`、`evidence.json` 为必备，`bench_l0.py` 为微基准脚本，另有 `checksum.sha256` 指纹。 | `ops/README.md`；`tests/test_ops_delivery_policy.py` | 不要只写"三件套"而漏掉指纹文件 |
| layer | 算子的归属层：`host_fork`（宿主侧）、`plugin_carrier`（插件侧）、`rejected_exploration`（已否定的探索）。工具新登记时可能留下占位值 `unclassified`，合入前必须改成上面三种之一（门禁会拦）。 | `DELIVERY_INDEX.json` 的 `layer` | 不要把 `unclassified` 当成第四种层 |
| status / serving_status | 索引里的交付状态与服务结论标签（如 `default_off_negative_speedup`、`verified_faster`、`within_noise`）。标签本身是术语，不得改写。 | `DELIVERY_INDEX.json` | 不要用"通过/不通过"替代标签 |
| 形状事实（shape facts） | 张量维度、批量、输入输出规模、执行相位、专家分布等可复核的执行条件。 | `ops/<operator>/evidence.json` 的 `shape_facts`；`docs/operator-benchmark-standard.md` §1 | 不要用"典型形状" |
| 权重标签 | 模型名 + 精度格式的组合，是可比性事实的另一半。 | `ops/<operator>/evidence.json` 的 `environment.model` / `dtype` | 不要只写模型族名 |
| 单变量上下文 | 两次对比除被测项外全部固定：种子、采样、图模式、并发、负载。 | `docs/operator-benchmark-standard.md` §1.2 | 不要写"条件差不多" |
| 数值护栏 | 与原生实现的容差比对环节；失败即熔断。 | `ops/<operator>/evidence.json` 的 `l3_numerical_guard` | 不要写成"精度检查"以外的自造名 |
| 一票否决 | 数值不一致时立即关闭能力并保持 default-off，不再谈性能。 | `AGENTS.md` §2.1；`docs/operator-benchmark-standard.md` §2 | — |
| all-kernel 分母 | 模型整体算子耗时，用来判断单算子净省时能否被服务观测到。 | `tools/op_agent/serving_loop.py` 的 `parse-l4` | 不要用"总耗时"含糊表述 |
| within noise | 组间差异落在组内波动之内。只能这样写，不得改写成收益。 | `DELIVERY_INDEX.json` 的 `serving_status` | 不要写"略有提升" |
| 负结果 | 优化后更慢或无效的结论。必须保留并入索引，不得删除。 | `DELIVERY_INDEX.json` 的 `status` | 不要写"待复测"含糊过去 |
| cohort | 一组锁定的实验条件（宿主线、模型、请求形态、图配置、精度）。数字只在同一 cohort 内可比。 | `docs/cohort-and-ownership.md` §2 | 不要用"环境"替代 cohort |
| 负载口径（load caliber） | 某次采集绑定的负载标识，与 cohort 一起构成产物的生产者标记。 | `VLLM_HUST_LOAD_CALIBER_ID`、`VLLM_HUST_COHORT_ID`；`tools/analysis/matmulv2_shape_attribution.py` | 不要写成"负载类型" |
| 跨域外推（禁止） | 把某 cohort 的结论搬到另一 cohort。历史 Qwen2.5 / 0.25 与当前基准互不推导。 | `docs/cohort-and-ownership.md` §3 | — |
| 工作项代号（`D0`/`D4`/`W1R`/`F2`/`W3F2`） | 历史工作单编号，只作检索标签，单独出现不构成术语。 | 例：`VLLM_HUST_D4_CONV_PACK` 中的 `D4` | 每次出现必须同时写出物理实体（环境变量、路径或分支） |

### 3.5 C5 调优 Harness（内外环）

| 术语 | 定义（判据） | 锚点 | 禁用或易混写法 |
|---|---|---|---|
| 内环（inner loop） | 单卡秒级的单算子闭环：数值校验、零重编译门禁、分桶计时。 | `tools/op_agent/optimizer_loop.py`；`tools/op_agent/evaluator.py` | 不要写成"单算子评测" |
| 外环（outer loop） | 双卡服务闭环：整网对拍、调用归因、负载 A/B。 | `tools/op_agent/serving_loop.py`；`workloads/run_workload.sh` | 不要写成"端到端测试" |
| `OpCase` | 算子优化目标的可插拔契约，实现后注册进 `CASE_REGISTRY`。 | `tools/op_agent/cases/base.py` | 不要写成"用例" |
| 形状桶（`shape_buckets`） | 必须覆盖的固定形状集合，对应图捕获档位。 | `OpCase.shape_buckets` | — |
| 动态形状探针（`dynamic_shape_probes`） | 没参与过编译的非常规行数，用来抓"把行数固化进编译期"的实现。 | `OpCase.dynamic_shape_probes` | — |
| `recompile_budget_ms` | 动态探针单次执行的耗时上限，超出即判为发生重编译。 | `tools/op_agent/evaluator.py` | 不要写成"超时时间" |
| 短路评估 | 前一阶段不通过立即返回，不进入下一阶段。 | `tools/op_agent/evaluator.py` | — |
| 知识库缓存（KB） | 单例缓存的历史结论，避免同一形状重复评估。 | `tools/op_agent/kb.py` | 不要写成"数据库" |
| campaign | 多算子批次流程：`discover` 产出候选与批次审核单，勾选 `approved` 后再 `run`。 | `tools/op_agent/campaign.py` | 不要写成"批量测试" |
| 泳道图 | 由 timeline 折叠出的高层级子图片段可视化，供人工审阅。 | `tools/op_agent/timeline_insight.py` | — |
| SNR 门 | 判断算子净省时能否超过服务波动的预判门槛；未过即归档为 `within noise`。 | `tools/op_agent/serving_loop.py` 的 `evaluate_e2e_snr_gate` | 不要写成"收益门槛" |
| 负载（workload） | `random`（定长受控）、`long-prefix`（长上下文预填充）、`sharegpt`（真实长尾）。主测与压力负载按算子类型配对。 | `tools/op_agent/workloads.py`；`workloads/README.md` | 不要写"数据集"，`sharegpt` 仅指负载名 |
| `primary_workload` / `stress_workload` | 配对规则：主测对比负载与动态形状压力负载，缺一即证据不完整。 | `workloads/README.md` | — |
| 载体状态快照（`carriers.env`） | 一次起服窗口里被核验的开关与入口点状态集合，属记录轴而非门禁；唯一来源是与 serve 日志同名的 `<log>.env.json`。 | `tools/bench-ladder.sh` 的 `carrier_write_serve_env`、`carrier_summary` | 不要与 C2 的"插件载体"混称 |

### 3.6 C6 证据保管与溯源

| 术语 | 定义（判据） | 锚点 | 禁用或易混写法 |
|---|---|---|---|
| `evidence.json` | 机器可读的验收凭据；没有凭据的结论不得进索引。 | `ops/<operator>/evidence.json` | 不要写成"结果文件" |
| `operator-benchmark-evidence/v1` | `evidence.json` 的 schema 标识；改字段等于改 schema。 | `docs/operator-benchmark-standard.md` §4 | — |
| `DELIVERY_INDEX.json` | 全仓算子交付的权威事实清单；新增或改状态必须登记。 | 根目录；`tests/test_json_hygiene.py` | 不要写成"清单文件" |
| 证据指纹（`checksum.sha256`） | 目录内文件的 sha256 清单，用于发现证据被静默改写。 | `ops/*/checksum.sha256` | 不要写成"校验和"泛指 |
| custody | 生成与校验指纹的工具契约。 | `tools/op_agent/custody.py` | — |
| provenance（溯源） | 从历史实现或提交提取代码时的原貌保存：版权头、commit、只读。 | `provenance/`；`docs/PROVENANCE.md` | 不要写成"来源说明" |
| CPU 门禁 | 不接 NPU 也能跑的测试集；`host_tree` 标记的用例在缺宿主树时失败而不是跳过。 | `pyproject.toml` 的 `markers`；`pytest -q -m "not host_tree"` | 不要写成"离线测试" |
| `MOD_METADATA.json` | 仓库身份、责任人、default-off、回滚与证据限定。 | 根目录 | — |

---

## 4. 同名异义消歧表

遇到下表左列的写法，按上下文补全，不允许裸用：

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

---

## 5. 废止与禁用术语（墓碑）

### 5.1 已废止词

| 废止词 | 曾经的指代 | 现在怎么说 | 依据 |
|---|---|---|---|
| Cascade / `cascade-attention` | 已撤除的注意力实现及其入口 | "已撤除实现"，不称能力、不称收益 | `docs/HOST_CONTRACT.md` 的 Cascade 接入边界段 |
| `apply-runtime-patches.sh`、`runtime_source_patches` | 运行时打补丁链 | 宿主提案（host proposal） | `docs/container-and-plugin-boundary-spec.md` §3.5 |
| `based/image-baseline` | 旧基线名 | `based/split-batch-baseline` | `AGENTS.md` §1.7 |
| `gdn-qkv-norm` plugin / bundle | 已撤回注册的入口 | 宿主提案 `host-proposals/GDN-qkv-norm/` | `README.md` 的 GDN QKV norm 段 |
| `Raing5Days/vllm-hust-cascade-kernel` | 早期独立发行的 Cascade kernel 包与其个人账号归属 | 历史专项，仅引用不依赖；org 归属未定 | `docs/cohort-and-ownership.md` §1 归属表 |
| 历史 Qwen2.5 / 0.25 结论 | 可外推的收益结论 | 历史独立专项，禁止跨域外推 | `docs/cohort-and-ownership.md` §2 |

### 5.2 禁用表达（出现即返工）

`AGENTS.md` §3.2 已禁的隐晦暗语，全部按"直接写物理实体"处理：载波、领地、点钉、husk、档2 载体、短跑、活守卫。

替换规则：一律改写为具体对象——环境变量名、代码文件相对路径、确切分支名或确切测试命令。

---

## 6. 维护记录

| 日期 | 变更 | 依据 |
|---|---|---|
| 2026-11-14 | 建立本文件：6 个限界上下文、术语表、消歧表、墓碑与新增流程；`AGENTS.md` 增加指向本文的指针 | 全仓文档与代码锚点普查（`docs/`、`ops/`、`src/`、`tools/`、`DELIVERY_INDEX.json`） |

后续每次增删词条都要在这里加一行，写明改了什么与依据出处；只在术语层变化时记，纯错字不入表。
