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

### 2.1 运行时源码变换退役记录（2026-11-01，issue #33 方案 A）

`runtime_source_patches` 源码变换模块已整体退役，连同备用加载入口 `runtime_source_patches_pth.py` 与随之失去消费点的 `moe_init_triton.py`。`gate-fp32` 的 `vllm.general_plugins` 入口不再在加载期安装任何 meta-path 源码变换钩子（退役前该调用是无条件的，与 `VLLM_HUST_GATE_FP32` 无关）。原挂在模块下的 5 个开关归宿如下，每条均有仓内记录可查：

| 开关 | 归宿 | 依据 |
|---|---|---|
| `VLLM_HUST_D4_CONV_PACK` | 归宿主 Fork，插件侧锚点表删除 | 宿主已有正式实现并登记：[ops/gdn_conv_pack](../ops/gdn_conv_pack/REPORT.md)、`DELIVERY_INDEX.json` 条目 `layer: host_fork`（宿主 commit `deb5ffe`，分支 `work/issue20-w3f2-host`，基于旧基线；迁入新基线 `based/split-batch-baseline` 由 issue #32 承接） |
| `VLLM_HUST_GDN_UNPACK_VIEWS` | 删除 | [host-proposals/ConcatD-zero-copy-unpack](../host-proposals/ConcatD-zero-copy-unpack/OWNERSHIP.md) 已定性为早期探索、其收益与连续性已被宿主卷积权重预打包方案覆盖；且其变换表依赖卷积预打包锚点注入的行，无法独立成立 |
| `VLLM_HUST_MOE_INIT_TRITON` | 删除 | 仓内无已归档的 L0 门通过记录（`tools/bench_moe_init_triton_only.py` 的 10% 门从未回填结果），收益未证明；其唯一仓内消费点是本模块注入 token_dispatcher 的分支。若重启，须按宿主提案流程下沉 |
| `VLLM_HUST_NZ_SKIP_GATE` | 删除 | 全仓无评测证据、无交付登记，仅为 `aclnnMatmulWeightNz` EZ1001（n=1/k=1）的应急豁免；宿主如有同类需要，按宿主提案流程处理 |
| `VLLM_HUST_DBO_ENABLE` | 删除 | [docs/comm-overlap-dbo-feasibility.md](comm-overlap-dbo-feasibility.md) 已结题"不可行"（2026-10-22）：起服拦截于 microbatching 后端配置校验断言，且 DBO 钩子消费者仅存在于 DeepEP 系内核，TP allReduce 路径无可行切入点 |

退役守卫：`tests/test_runtime_patch_manifest_guard.py` 断言上述模块不再存在、`gate-fp32` 入口不再挂接源码变换、seam 断言表不再登记源码变换条目。

宿主侧下沉算子索引：
- **GDN 卷积权重加载期预打包 (GDN conv-weight 转置外提与连续打包)**：专属报告与证据结构见 [ops/gdn_conv_pack](../ops/gdn_conv_pack/REPORT.md)。宿主分支 `work/d4-conv-pack`（宿主 PR #64，叠层于 PR #63）。
- **GDN in_proj 输入投影权重加载期合并 (GDN 输入投影两次 MatMul 合一)**：专属报告与证据结构见 [ops/gdn_merge_inproj](../ops/gdn_merge_inproj/REPORT.md)。宿主分支 `work/w3f2-inproj-merge`（宿主 PR #63）。L0/L3/L4/端到端均未测，放行前必须补齐 L3 数值护栏。
- **MoE 共享专家后置门控乘融合 (`moe_epilogue`)**：把 `AscendSharedExperts.part2` 里的广播乘换成单 kernel，保留宿主 `sigmoid`。算子层单次 3.64 → 1.34 µs，输出逐字一致；全链路 Mul 家族 −37.0 ms；端到端不可判定，不主张服务收益。见 [ops/moe_epilogue](../ops/moe_epilogue/REPORT.md)。
- **Qwen Attention 纯文本 Split-QKV 与 RoPE 融合 (`qwen_text_split_qkv_fuse`)**：在纯文本路径下消除 Split、RMSNorm 与 RoPE 的多步离散调用，统一接驳至片上 UB 的 Triton 融合 Kernel。见 [ops/qwen_text_split_qkv_fuse](../ops/qwen_text_split_qkv_fuse/REPORT.md)。宿主分支 `work/qwen-text-split-qkv-fuse`。
- **GDN Part 3 Output Norm 与 Rearrange 融合 (`gdn_output_norm_fusion`)**：在 GDN 输出端将 RMSNormGated 与 einops rearrange 融合为单步 Triton 算子，直接写入连续 2D 张量直通投影层。见 [ops/gdn_output_norm_fusion](../ops/gdn_output_norm_fusion/REPORT.md)。宿主分支 `work/gdn-output-norm-fusion`。

探索与负性能留存算子索引：
- **GDN QKV 单 pass 解包 (`gdn_qkv_unpack`)**：专属报告与证据结构见 [ops/gdn_qkv_unpack](../ops/gdn_qkv_unpack/REPORT.md)。910B2 实测因 Triton 启动发射开销慢于原生链，按规范留存负性能事实，保持未激活。
- **MoE 双门合并 (`moe_gate_merge`)**：专属报告与证据结构见 [ops/moe_gate_merge](../ops/moe_gate_merge/REPORT.md)。虽然纯 MatMul 有微弱加速，但非对齐切片搬运带来额外开销，流水线跑输原生分离方案，按规范留存负性能事实，不予集成。


Cascade 不属于当前发布组件：没有源码、注册入口或可执行路径。历史报告不代表当前能力。

## 3. 宿主 Fork 变更清单与分支铁律（下沉源码修改，彻底废弃运行时补丁）

宿主侧无法由插件外部介入的修改已全部下沉到宿主 Fork 仓库，随宿主镜像构建发布；本仓废除 `tools/runtime-tree/apply-runtime-patches.sh` 等任何在容器运行树上打补丁的工具。

### 3.1 两个宿主仓库与基线分支铁律
由于社区 `main` 分支走在非常靠前的上游（已领先本地镜像 260+ commits），底层依赖已与当前 910B2 镜像运行时脱节。因此两个宿主仓库**定死以 `based/split-batch-baseline` 为基线与 PR 目标**：
1. **昇腾适配宿主 (`vllm-ascend-hust`)**：
   - 仓库：`git@github.com:vLLM-HUST/vllm-ascend-hust.git`
   - **基线分支**：`based/split-batch-baseline`（分支头 `ad82d77cb`，内容为镜像 `74f0c0a27` 的宿主源码 + 3 个构建文件修复）
   - **PR 铁律**：所有特性分支从 `based/split-batch-baseline` 切出，发起 PR 的 Target 分支必须指定为 `based/split-batch-baseline`，严禁向 `main` 或旧基线 `based/image-baseline` 提 PR。
2. **vLLM 核心宿主 (`vllm-hust`)**：
   - 仓库：`git@github.com:vLLM-HUST/vllm-hust.git`
   - **基线分支**：`based/split-batch-baseline`（锚定当前镜像 `f18cf803c`；与旧 `based/image-baseline` 指向同一 commit，内容零差异）
   - **PR 铁律**：所有特性分支从 `based/split-batch-baseline` 切出，发起 PR 的 Target 分支必须指定为 `based/split-batch-baseline`，严禁向 `main` 提 PR。

### 3.2 现存下沉变更清单

两项宿主层算子已按新基线重新提交并合入 `based/split-batch-baseline`（2026-11-01，基线头 `771b847da`；宿主 PR #63/#64 已合并，vLLM-HUST/vllm-ascend-split-batch-hust#32 承接）：

- **对应开发分支**：
  - `work/w3f2-inproj-merge`（宿主 PR #63，从 `based/split-batch-baseline` 头 `ad82d77cb` 切出，cherry-pick 旧分支提交，树逐字节一致）；
  - `work/d4-conv-pack`（宿主 PR #64，叠层于 `work/w3f2-inproj-merge` 之上——其 pack 触发点位于 in_proj hook 内部，存在合并顺序硬依赖）。
  - 旧开发分支 `work/issue20-w3f2-host`（基于旧基线 `based/image-baseline`）仅作历史溯源，其内容已被上述两分支完整取代。
- **变更明细**：
  1. **GDN in_proj 输入投影权重加载期合并与 Hook（历史代号 W3F2）** — Commit `fcbc4458bb8afcc37441e602921ed45dcbe5e775`（cherry-pick 自 `ce807cd7ce972b9a410663cebe10b7ea25fb2a31`）
     - `vllm_ascend/ops/gdn.py`：实现 `process_weights_after_loading` hook，当 `VLLM_HUST_GDN_MERGE_INPROJ=1` 时合并 `in_proj_ba` 与 `in_proj_qkv`。
     - `vllm_ascend/worker/model_runner_v1.py`：在加载权重后调用该 hook。
     - `vllm_ascend/patch/worker/patch_qwen3_5.py`：对齐 Qwen3.5 模型层的权重重绑定。
     - 插件仓内履约与证据：详见 [ops/gdn_merge_inproj](../ops/gdn_merge_inproj/REPORT.md)。
  2. **GDN 卷积权重加载期预打包与转置外提（历史代号 D4）** — Commit `771b847dae5e9493fae9ca234b7498a072ffd39c`（cherry-pick 自 `deb5ffefe32b53c61ae6193ff4ee4368cf1889ba`）
     - `vllm_ascend/ops/gdn.py`：在加载期物化 conv1d.weight `[4096,1,4]` 为连续 `[4,4096]`（由 `VLLM_HUST_D4_CONV_PACK=1` 开启）；forward 内部统一使用预打包权重，消除每步 `transpose(0, 1)` 开销；单层每步消除动态转置开销，全模型 30 层 GDN 净消除多余转置。
     - 插件仓内履约与证据：详见 [ops/gdn_conv_pack](../ops/gdn_conv_pack/REPORT.md)。
  3. **MoE 共享专家后置门控乘融合** — Commit `ace3d38423123c101ca2cdca8aeaaf21a7a40c5c`（分支 `work/moe-shared-expert-epilogue`，基于 `based/split-batch-baseline`）
     - `vllm_ascend/ops/fused_moe/shared_experts.py`：`part2` 的广播乘改为调用 `torch.ops.vllm.moe_gate_mul`，由 `VLLM_HUST_MOE_EPILOGUE=1` 开启；关闭时逐字走原表达式。
     - `vllm_ascend/ops/triton/moe_epilogue.py`：单 kernel 与运行期形状判定（≤32 行走融合，prefill 回退）。
     - `vllm_ascend/ops/register_custom_ops.py`：注册 `moe_gate_mul`（`PrivateUse1`）。
     - 插件仓内履约与证据：详见 [ops/moe_epilogue](../ops/moe_epilogue/REPORT.md)。
  4. **Qwen Attention 纯文本 Split-QKV 与 RoPE 融合** — Commit `493ec93d243816217e15daaab480fbe073051981`（分支 `work/qwen-text-split-qkv-fuse`，PR #67，已合入 `based/split-batch-baseline`，合并提交 `d1014aadb`）
     - `vllm_ascend/patch/worker/patch_qwen3_5.py`：在纯文本 1D 模式下将 Split、RMSNorm 与 RoPE 接驳至已有 Triton 融合算子 `triton_split_qkv_rmsnorm_mrope`，由 `VLLM_HUST_SPLIT_QKV_ROPE_FUSED=1` 开启；关闭时保留原生多步离散链路。
     - 插件仓内履约与证据：详见 [ops/qwen_text_split_qkv_fuse](../ops/qwen_text_split_qkv_fuse/REPORT.md)。
  5. **GDN Part 3 Output Norm 与 Rearrange 融合** — Commit `68c0790818dd15ff28b55d95713c2f0f4b3df66e`（分支 `work/gdn-output-norm-fusion`，基于 `based/split-batch-baseline`）
     - `vllm_ascend/ops/gdn_epilogue.py`：新增 `gdn_output_norm_rearrange` 融合算子与门控 `is_output_norm_enabled()`，由 `VLLM_HUST_GDN_OUTPUT_NORM_FUSED=1` 开启；关闭时保留原生多步分立链路。
     - `vllm_ascend/ops/gdn.py`：在 Part 3 输出端对接融合算子，将输出直通 `self.out_proj`。
     - 插件仓内履约与证据：详见 [ops/gdn_output_norm_fusion](../ops/gdn_output_norm_fusion/REPORT.md)。

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
| PR-B (`cleanup/pr-b-host-fork-migration`) | 废除仓内运行时打补丁工具；下沉权重合并与卷积权重预打包至宿主 Fork | PR-A | `pytest tests/test_runtime_patch_manifest_guard.py tests/test_runtime_tree_seams.py` (9 passed, 1 deselected)；`check-tree-seams.py --selftest` PASS | 宿主 Fork 分支 `work/issue20-w3f2-host` 已推送；插件仓零运行时打补丁脚本 |
| PR-C (`cleanup/pr-c-capability-docs`) | 支持表、README、release/architecture/pitfalls/progress 过时操作清扫与 planner 边界 | PR-B | `pytest tests/test_document_custody.py` (6 passed)；Ruff PASS | Cascade 彻底撤除且无执行指南；root planner 明确不可激活 |
| PR-D (`cleanup/pr-d-workbench-strip`) | 剥离 opt 搜索与外部工作台 helpers，保留归档索引 `workbench-archive.json` | PR-C | `pytest -q -m "not host_tree"` (401 passed, 23 skipped, 15 deselected)；tar sha256 逐项核验 PASS | 归档外部维护；测试不依赖 `/root/opt-ops` 与本地数据库 |
| PR-E (`cleanup/pr-e-integration-audit`) | 整体构建分发清单、打包与整体验收固化 | PR-D | `build` + `tools/check_distribution.py` + `twine check` 全部 PASS；全量 CPU 套件 401 passed | 当前环境无昇腾卡；不据此声称 NPU dispatch、正确性或服务收益通过 |
