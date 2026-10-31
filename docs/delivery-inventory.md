# 交付组件、宿主变更与证据清单

## 1. 架构分层原则

本仓库遵循**两层架构分工**，彻底杜绝在运行树动态打补丁：

- **宿主层（宿主 Fork，`vllm-ascend-hust`）**：负责模型架构、Runner 生命周期、执行图及底层基础设施修复。在插件侧无法介入的位置（如内联算子逻辑、权重加载后处理 hook），直接在宿主 Fork 源码中实现，随宿主镜像原生发布。
- **插件层（本仓，`vllm-ascend-split-batch-hust`）**：作为标准的 `vllm.general_plugins`，仅负责纯 Python / Triton 算子实现、default-off 门控策略、算子调度与边界验证。插件仓不承载任何运行时打补丁工具。

## 2. 插件发布面与独立算子索引

入口来自 `pyproject.toml` 的 entry points；源码模块、探针和测试不计作插件。全仓各算子的独立实现、专属文档与详细报告结构索引见 [DELIVERY_INDEX.json](../DELIVERY_INDEX.json) 及 [ops/ 模块目录](../ops/)。

| 名称 | 可发现 | 可激活 | 真实调用 | 算子收益 | 服务收益 | 独立子目录与报告 |
|---|---|---|---|---|---|---|
| `fia-demask` | 是，general plugin + bundle | active，默认关闭 | 仅纯 decode 且 Q_S=1/请求 | A/B inconclusive；不主张收益 | 未证明 | [ops/fia_demask](../ops/fia_demask/REPORT.md) |
| `rope-fix` | 是，general plugin + bundle | import_only | 无正式启用路径 | 未证明；不主张性能收益 | 未证明 | - |
| `zerocost-wiring` | 是，general plugin + bundle | active，双开关默认关闭 | 依赖运行时门控 | A/B inconclusive | 未证明 | - |
| `gate-fp32` | 是，general plugin + bundle | active，默认关闭 | 受支持权重加载/执行路径 | 有历史局部 Cast 外提记录；只适用于原形状与权重 | 未证明当前端到端服务收益 | - |
| `gdn-merge-inproj` | 是，general plugin + bundle | import_only，fail-closed | 插件侧缺少安全消费 seam；不合并权重 | 插件自身无收益；实际权重合并由宿主层承载 | 未证明 | - |
| `split-batch-full-graph` planner | 是，bundle | import_only，不可激活 | 仅纯规划函数；没有宿主执行面 | 不适用 | 未证明 | - |
| `fi-sampling` | 是，general plugin | 无 bundle | 依赖运行时路由 | 历史 Qwen2.5/0.25 证据仅方向；当前域未证明 | 历史端到端差异在噪声内；不得外推 | [ops/fi_sampling](../ops/fi_sampling/REPORT.md) |
| `fi-gelu` | 是，general plugin | 无 bundle，默认关闭 | OOT provider 受形状/类型门控；当前主模型不走 GELU | probed 独立形状慢于原生链 | L3/L4 未完成 | [ops/fi_gelu](../ops/fi_gelu/REPORT.md) |
| `aot-cache-guard` | 是，general plugin | 无 bundle | 宿主修复后 no-op | 不主张稳态收益 | 仅冷启动问题另行评估 | - |
| `mlp-chunk` | 是，general plugin | 无 bundle，默认 chunks=1 | 仅配置启用时 | 目标是内存控制，不主张延迟收益 | 未证明容量/并发收益 | - |
| `runtime_source_patches` 源码变换 | 随包模块，由 `gate-fp32:load` 调用 install；不是独立 entry point | 四个 env 默认关闭 | 锚点精确匹配且模块尚未导入才安装；漂移回原生 | 独立旧快照不自动验收当前转换模块；未证明完整 L0/L3/L4 配对 | 未证明 | - |

宿主侧下沉算子索引：
- **D4 (GDN conv-weight 转置外提与连续打包)**：专属报告与证据结构见 [ops/gdn_conv_pack](../ops/gdn_conv_pack/REPORT.md)。

探索与负性能留存算子索引：
- **GDN QKV 单 pass 解包 (`gdn_qkv_unpack`)**：专属报告与证据结构见 [ops/gdn_qkv_unpack](../ops/gdn_qkv_unpack/REPORT.md)。910B2 实测因 Triton 启动发射开销慢于原生链，按规范留存负性能事实，保持未激活。


Cascade 不属于当前发布组件：没有源码、注册入口或可执行路径。历史报告不代表当前能力。

## 3. 宿主 Fork 变更清单与分支铁律（下沉源码修改，彻底废弃运行时补丁）

宿主侧无法由插件外部介入的修改已全部下沉到宿主 Fork 仓库，随宿主镜像构建发布；本仓废除 `tools/runtime-tree/apply-runtime-patches.sh` 等任何在容器运行树上打补丁的工具。

### 3.1 两个宿主仓库与基线分支铁律
由于社区 `main` 分支走在非常靠前的上游（已领先本地镜像 260+ commits），底层依赖已与当前 910B2 镜像运行时脱节。因此两个宿主仓库**定死以 `based/image-baseline` 为基线与 PR 目标**：
1. **昇腾适配宿主 (`vllm-ascend-hust`)**：
   - 仓库：`git@github.com:vLLM-HUST/vllm-ascend-hust.git`
   - **基线分支**：`based/image-baseline`（锚定当前镜像 `74f0c0a27`）
   - **PR 铁律**：所有特性分支从 `based/image-baseline` 切出，发起 PR 的 Target 分支必须指定为 `based/image-baseline`，严禁向 `main` 提 PR。
2. **vLLM 核心宿主 (`vllm-hust`)**：
   - 仓库：`git@github.com:vLLM-HUST/vllm-hust.git`
   - **基线分支**：`based/image-baseline`（锚定当前镜像 `f18cf803c`）
   - **PR 铁律**：所有特性分支从 `based/image-baseline` 切出，发起 PR 的 Target 分支必须指定为 `based/image-baseline`，严禁向 `main` 提 PR。

### 3.2 现存下沉变更清单
- **对应开发分支**：`work/issue20-w3f2-host`（已推送到 `vLLM-HUST/vllm-ascend-hust`，基于 `based/image-baseline`）
- **变更明细**：
  1. **W3F2 (GDN in_proj 权重合并与 Hook)** — Commit `ce807cd7ce972b9a410663cebe10b7ea25fb2a31`
     - `vllm_ascend/ops/gdn.py`：实现 `process_weights_after_loading` hook，当 `VLLM_HUST_GDN_MERGE_INPROJ=1` 时合并 `in_proj_ba` 与 `in_proj_qkv`。
     - `vllm_ascend/worker/model_runner_v1.py`：在加载权重后调用该 hook。
     - `vllm_ascend/patch/worker/patch_qwen3_5.py`：对齐 Qwen3.5 模型层的权重重绑定。
  2. **D4 (GDN conv-weight 转置外提与预打包)** — Commit `deb5ffefe32b53c61ae6193ff4ee4368cf1889ba`
     - `vllm_ascend/ops/gdn.py`：在加载期物化 conv1d.weight `[4096,1,4]` 为连续 `[4,4096]`（由 `VLLM_HUST_D4_CONV_PACK=1` 开启）；forward 内部统一使用预打包权重，消除每步 `transpose(0, 1)` 开销；单层每步消除动态转置开销，全模型 30 层 GDN 净消除多余转置。
     - 插件仓内履约与证据：详见 [ops/gdn_conv_pack](../ops/gdn_conv_pack/REPORT.md)。

## 4. 工具边界

- **交付运行时**：`src/vllm_ascend_split_batch/`、entry points 与必要的回退/门控。
- **交付验证**：`tests/`、必要的算子基准 `tools/bench_*.py` 与宿主接口静态校验 `tools/runtime-tree/check-tree-seams.py`。
- **外部工作台，不属于插件交付面**：opt 搜索/候选台账、一次性迁移工具、工作区 keeper 等已剥离至仓外归档 `/root/operator-cleanup-workbench-185f48f3.tar`，由 `tools/workbench-archive.json` 保留索引。

## 5. 性能证据最小字段与评测阶梯

算子性能评测与收益主张必须全面遵守 [docs/operator-benchmark-standard.md](operator-benchmark-standard.md) 中定义的四级验证阶梯（L0 纯算子微基准 -> L3 数值正确性护栏 -> L4 调用链路归因 -> 服务端到端实测）。
每一项性能主张必须关联固定原始结果和 SHA256，至少记录：

- 算子、callsite、phase、目标调用数/耗时以及 all-kernel 分母；shape（M/K/N 或输入/输出张量尺寸）。
- 模型/精度/权重格式，host/core/runtime 版本与源码提交或散列，硬件/拓扑。
- 请求输入与输出 token 数或张量元素数、batch/并发、prefill/decode、图模式、调度及关键配置。
- OFF/ON 开关与实际调用证据、重复次数、比较方法、失败/噪声/边界。
- 算子结论与端到端服务结论分开；服务指标明确给出 TTFT/TPOT/吞吐或任务质量、负载长度和统计方式。

字段缺失时标记“未证明”或“仅方向”，不得补造条件；负收益和失败记录保留。不得跨模型、权重、算子形状或 cohort 直接比较数字。

## 6. 整体验收与分批验证矩阵（PR-E）

| PR | 目标与范围 | 依赖前置 | 验收命令与结果 | 边界与声明 |
|---|---|---|---|---|
| PR-A (`cleanup/pr-a-delivery-inventory`) | 交付清单与单一事实源导航；确立两层架构分工 | 起点 `185f48f` (`origin/main`) | `pytest tests/test_document_custody.py` (5 passed)；Ruff PASS | 不升级任何历史收益主张；entry point 逐项对应 |
| PR-B (`cleanup/pr-b-host-fork-migration`) | 废除仓内运行时打补丁工具；下沉 W3F2/D4 至宿主 Fork | PR-A | `pytest tests/test_runtime_patch_manifest_guard.py tests/test_runtime_tree_seams.py` (9 passed, 1 deselected)；`check-tree-seams.py --selftest` PASS | 宿主 Fork 分支 `work/issue20-w3f2-host` 已推送；插件仓零运行时打补丁脚本 |
| PR-C (`cleanup/pr-c-capability-docs`) | 支持表、README、release/architecture/pitfalls/progress 过时操作清扫与 planner 边界 | PR-B | `pytest tests/test_document_custody.py` (6 passed)；Ruff PASS | Cascade 彻底撤除且无执行指南；root planner 明确不可激活 |
| PR-D (`cleanup/pr-d-workbench-strip`) | 剥离 opt 搜索与外部工作台 helpers，保留归档索引 `workbench-archive.json` | PR-C | `pytest -q -m "not host_tree"` (401 passed, 23 skipped, 15 deselected)；tar sha256 逐项核验 PASS | 归档外部维护；测试不依赖 `/root/opt-ops` 与本地数据库 |
| PR-E (`cleanup/pr-e-integration-audit`) | 整体构建分发清单、打包与整体验收固化 | PR-D | `build` + `tools/check_distribution.py` + `twine check` 全部 PASS；全量 CPU 套件 401 passed | 当前环境无昇腾卡；不据此声称 NPU dispatch、正确性或服务收益通过 |
