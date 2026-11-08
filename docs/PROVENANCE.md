# Provenance

> **核对状态（2026-09-26）**：逐条核对结论见
> [`docs/provenance-verification.md`](docs/provenance-verification.md)（含命令与原始输出）。
> 机械守卫：`tests/test_provenance_hashes.py`（vendored hash / `api.py` 逆变换复核 / 许可声明 /
> 本文件的 commit 引用）。一句话：**14 个归档提交全部可复核**（sha 存在于源仓、作者唯一、
> PR 状态与本文一致）；**vendored 4 个文件全部可复核**（其中 `api.py` 用逆变换复核，见 §4.1）；
> **1 项如实记录的缺口**：14 个提交**均无** DCO 类 trailer（见 §2）。

## 1. 源仓与历史（已核对）

| 项 | 值 | 核对结果 |
|---|---|---|
| core 源仓 | [intellistream/vllm-hust-legacy-20260831](https://github.com/intellistream/vllm-hust-legacy-20260831) | public、**archived**、license **Apache-2.0**、pushed 2026-08-31 |
| ascend 源仓 | [intellistream/vllm-ascend-hust-legacy-20260831](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831) | public、**archived**、license **Apache-2.0**、pushed 2026-08-30 |

| PR | 状态（2026-09-26 复核） | merge commit | head | 提交数（=归档数） |
|---|---|---|---|---|
| core #273 | **closed + merged** | `13f11da44455` | `ab3a536aac3a` | 7 = 7 ✓ |
| core #263 | **closed + NOT merged** | —（仅测试合并值） | `13ee9f7db462` | 2 = 2 ✓ |
| ascend #280 | **open + not merged** | — | `ecb70fb47826` | 2 = 2 ✓ |
| ascend #281 | **open + not merged** | — | `e08f79d014b6` | 1 = 1 ✓ |
| ascend #282 | **open + not merged** | — | `82a8cc7dd770` | 1 = 1 ✓ |
| ascend #283 | **open + not merged** | — | `73c121d4c212` | 1 = 1 ✓ |

**Closed 不等于 merged，open 不等于 accepted。这些是迁移证据，不是发布凭据。**

## 2. 归档提交（14 个，逐条可复核）

`provenance/legacy-patches/` 保存 `git format-patch` 原始导出（**只读**，勿改）。
每个 sha 都已在对应源仓用 `GET /repos/{repo}/commits/{sha}` 复核（HTTP 200）。

| # | patch | commit（前 12） | 作者 | +/- |
|---|---|---|---|---|
| 1 | `core-pr-263/0001-828cf30d84cd` | `828cf30d84cd` | Raing5Days <M202574102@hust.edu.cn> | +268/−11 |
| 2 | `core-pr-263/0002-13ee9f7db462` | `13ee9f7db462` | 同上 | +54/−1 |
| 3 | `core-pr-273/0001-0c5f3714f4be` | `0c5f3714f4be` | 同上 | +268/−11 |
| 4 | `core-pr-273/0002-1ca673fddf05` | `1ca673fddf05` | 同上 | +54/−1 |
| 5 | `core-pr-273/0003-3f019aaec355` | `3f019aaec355` | 同上 | +315/−172 |
| 6 | `core-pr-273/0004-d351c22e18a6` | `d351c22e18a6` | 同上 | +19/−16 |
| 7 | `core-pr-273/0005-1a9eb1a14d35` | `1a9eb1a14d35` | 同上 | +346/−357 |
| 8 | `core-pr-273/0006-860c7a9657f8` | `860c7a9657f8` | 同上 | +825/−77 |
| 9 | `core-pr-273/0007-ab3a536aac3a` | `ab3a536aac3a` | 同上 | +1/−2 |
| 10 | `ascend-pr-280/0001-c46c31b550f1` | `c46c31b550f1` | 同上 | +1059/−30 |
| 11 | `ascend-pr-280/0002-ecb70fb47826` | `ecb70fb47826` | 同上 | +1/−2 |
| 12 | `ascend-pr-281/0001-e08f79d014b6` | `e08f79d014b6` | 同上 | +1317/−1 |
| 13 | `ascend-pr-282/0001-82a8cc7dd770` | `82a8cc7dd770` | 同上 | +3404/−149 |
| 14 | `ascend-pr-283/0001-73c121d4c212` | `73c121d4c212` | 同上 | +4333/−3 |

- 合计 **+12264/−833**，触及 **36 个不同路径**（清单见核对记录 §2）。
- **作者**：14 个提交的 `From:` 均为 `Raing5Days <M202574102@hust.edu.cn>`；源仓侧
  commit author 元数据同名。仓库内主维护者登记见 `MAINTAINERS.md`。
- **trailers**：14 个提交**均无** `Signed-off-by` / `Co-authored-by` / `Change-Id`
  —— 没有 DCO 签署证据。这不阻塞归档（原样留档），但**不能**据此宣称签核链完整。
- **版权头**：core 侧 patch 只改既有文件；ascend 侧 #281/#282/#283 的 patch 自身含
  Apache-2.0 头。本仓抽取代码的许可覆盖由 `tests/test_provenance_hashes.py` 机械守住。

## 3. 可迁移范围的判定

**结论：这批 legacy patch 一律不移植（工作区决策），只作为技术论据与缺陷定位参考。**
本仓当前代码**不是**这些 patch 的搬运，而是按宿主公开面重写；映射关系如下（按模块标注）：

| 本仓模块 | 对应 legacy 面 | 关系 |
|---|---|---|
| `planner.py`（`plan_dual_pad` / `precheck_reason`） | Ascend #281 `worker/dual_pad_utils.py`、`tests/test_dual_pad_planner.py` | **语义参照**，按宿主无关重写；`import_only`，无执行面 |
| `cascade_plugin.py` / `cascade_graph_plugin.py` / `cascade_runner_patch.py` | Ascend #282 的 `attention_v1.py` / `compilation/acl_graph*.py`、core #273 的 `forward_context.py` / `v1/cudagraph_dispatcher.py` | **行为参照**（两段式 cascade、图孪生、cudagraph key）；插件侧重写，宿主源码零改动 |
| `rope_fix_plugin.py`、`zerocost_wiring.py`、`mlp_chunk_plugin.py` | 无对应 patch | **独立开发**，不是 legacy 搬运 |
| legacy 的 `inplace_split_*` 系列、`csrc/third_party/catlass` 子模块 pin、`examples/**` | Ascend #280/#282/#283 | **未迁移**（inplace 车道未进入本仓） |

> 判定依据：legacy patch 中 **36 个路径有 32 个落在宿主树内**（`vllm/` 与 `vllm_ascend/`），
> 而本仓硬约束是"不修改宿主源码"（见 `AGENTS.md`）；`csrc/third_party/catlass` 是子模块 pin，
> 与插件无关；`examples/**` 是演示脚本。故可迁移面只剩"策略/契约/预检"这类宿主无关部分，
> 已在上表按模块落到本仓，其余留档不迁移。

## 4. 复核方式（复现）

```bash
# 归档自身的一致性 + 上游状态 + 许可覆盖（只读，联网部分可跳过）
python3 docs/provenance-verification.md 中的命令清单
pytest -q tests/test_provenance_hashes.py     # 机械守卫（离线）
```
