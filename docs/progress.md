# 进度日志与计划

历史时间线已迁入 benchmark；原文中的失败、噪声、回退与后续订正均原样保留。
本页仅作索引，不以旧测试结果代表当前资格。当前边界见 [support-matrix.md](support-matrix.md)、[release.md](release.md) 和 [注册关闭记录](../host-proposals/GDN-qkv-norm/registration-closure.json)。

## 1. 时间线

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L5)

### 阶段 0:split-batch legacy → 契约提取

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L7)

### 阶段 1:cascade attention 插件(eager, default-off)

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L16)

### 阶段 2:图模式(graph twin)

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L22)

### 阶段 3:自适应 gate(W2)

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L30)

### 阶段 4:工程化与发布准备(2026-09,本轮)

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L37)

## 2. 现状(2026-09-09)

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L53)

### 阶段 5:fi_sampling 采样接入(W2b,2026-09-09)

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L121)

## 3. 计划与开放问题

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L149)

## 2026-09-11 fia-demask（decode FIA 去掩码）

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L172)

## 2026-09-11 fia-demask 翻 active（独立 bundle）

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L182)

## 2026-09-11 rope-fix（fork RoPE 三缺陷的插件侧 carrier + 反向漂移守卫）

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L192)

### 复审收尾（同日，9 条 F 全部落实）

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L217)

## 2026-09-13 zerocost-wiring §3 启用验证（**阻塞，未翻 active**）

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L243)

## 2026-09-13 zerocost-wiring `_host_func` 多 wrapper 修复 → **§3 通过，已翻 active**

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L262)

## 2026-09-26 cascade 准入与默认关闭隔离（PR #5 评审落地）

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L284)

## 2026-09-26 插件包上 PyPI（0.1.0 → 0.1.1）+ 算子轮改走 Release 附件

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L310)

## 2026-09-27 0.1.3 已发布：`host.version_range` 放开为有界窗口

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L331)

## 2026-09-26 0.1.2 已发布（让 `test` extra 摘钉生效）

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L347)

## 2026-09-26 续做：把"本机可直接做"的后续项清空（issue #2 §7）

[查看该节原文及其适用边界](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/progress.md#L360)
