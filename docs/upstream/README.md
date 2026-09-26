# docs/upstream/ — **已冻结（不再对外提交）**

> 决定日期：2026-09-11（用户裁定）。纪律见工作区 `AGENTS.md` 硬性约束与
> [../agent-guide.md](../agent-guide.md) §1 第 6 条。

## 状态

**本目录不再维护、不再提交上游。** 保留原因只有一个：这里的技术论据（冗余性构造证明、
谓词面、验证清单）在本仓本地 patch 的维护中仍有参考价值——宿主升级重验时可能要重新推导
同一谓词。名字 `upstream/` 是历史遗留，内容**只作技术论据留档**。

## 边界

| 做 | 不做 |
|---|---|
| 本仓（插件侧）本地 patch，default-off、fail-open、三证据齐备后翻 `active` | 向 vllm / vllm-ascend 上游提交补丁或 PR |
| 跟随 `vllm-hust` release 节奏做版本同步（重钉锚点与 `host.version_range`） | "推动宿主提供 typed contract" 这类上游化动作 |
| 宿主升级时按 [../release.md](../release.md) §4 重核 monkeypatch/公开面签名 | 在宿主源码树里就地改代码 |

## 现有素材

| 文件 | 内容 | 状态 |
|---|---|---|
| [fia-decode-demask.md](fia-decode-demask.md) | decode FIA 冗余掩码省略：PR 描述草稿 + 论据链 + 上游化形态建议 | **冻结**，论据链可参考；PR 部分不再维护 |
