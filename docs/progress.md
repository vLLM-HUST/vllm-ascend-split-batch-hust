# 进度日志与计划

时间线基于 git log 事实;计划一节是开放问题,动手前先在团队对齐。

## 1. 时间线

### 阶段 0:split-batch legacy → 契约提取

- 历史 split-batch patch(ascend PR 280-283 / core PR 263/273)效果一般,
  按 AGENTS.md 决策**不移植 legacy patch**,原始实现存档于 `provenance/`(只读)。
- 提取 host-independent 组件为 inert 契约描述符(merge PR #3,
  `2f55b42`):纯 planner(`plan_dual_pad`)+ manifest 描述符,
  `import_only`,等待宿主提供 `vllm.graph.runtime-key.v1` 等 seam
  (提案见 HOST_CONTRACT.md)。

### 阶段 1:cascade attention 插件(eager, default-off)

- `7bddf66` default-off `vllm.general_plugins` 插件 + `256eed9` CPU 单测。
- eager 两段式:stage-1 共享前缀摊平读 + stage-2 suffix + LSE merge,
  Tier0(bf16)/Tier1(fp32,依赖 ascend_kernel wheel)双精度层。

### 阶段 2:图模式(graph twin)

- `8f8df73` 图模式两段式 cascade(default-off):twin graph 按
  `BatchDescriptor` 侧表存取、update-before-replay、bucket-static workspace。
- 系列修复:图池中间量生命周期(09264a1/e9fd67a/b99963c)、twin miss
  fail-open(672a255)、redundant update 跳过(afd487d/39f7616)、
  microbatching 互斥(7484a27/134d172)。

### 阶段 3:自适应 gate(W2)

- `2f041d6` 按 (B, shared) 桶启动期 micro-bench,亏区自动回落 FULL。
- 稳定性线:探针块表隔离(9899f00,aicore 踩内存修复)→ 子进程隔离
  (b479508)→ BNSD 探针 + OOM skip(7858229/904fa93/a3cbbf4)→
  dispatch 级消费(f423a58,HEAD)。

### 阶段 4:工程化与发布准备(2026-09,本轮)

- 对照 bidkv 打包发布指南(vllm-hust-docs 仓库)完成打包验证:wheel 含
  manifest 与双 entry point、`extension_bundles` 发现路径打通。
- 修复 manifest `activation.environment` 占位符隐患(注入语义化),新增
  版本一致性与值域守护测试。
- 建立 [release.md](release.md) 发布流程与宿主升级核对清单。
- 建立本 docs/ 知识库与仓库级 AGENTS.md。

## 2. 现状(2026-09)

- 分支 `feat/cascade-attention-plug`;bundle `import_only`,能力 default-off。
- e2e 证据锚点见根 README(16k 段 −21%~−38% 等,kernel README §6 有单算子锚点);
  测量报告在 `cascade-c3-results/`(工作区)。
- CPU 门槛:`pytest -q` 27 passed + `ruff check` 干净。

## 3. 计划与开放问题

> 优先级以工作区 AGENTS.md 为准(flashinfer 移植/e2e 为主线,本仓库是承载壳)。

1. **翻 `active` 的证据门槛**:default-off 零回归冒烟、正确性对齐、性能对比
   三项归档后,按 release.md 第 3 节走启用验证,再改 `implementation.status`。
2. **kernel wheel extras 联动**:插件 `[project.optional-dependencies]` 增加
   kernel 包软依赖(Tier1 需要),缺 wheel 时 fail-open 降级。
3. **宿主 seam 上游化**:split-batch 四协议仍是提案;推动 vllm-hust 宿主提供
   typed contract(参照 bidkv/victim-selector 的宿主适配路径),摆脱
   monkeypatch 弱契约。
4. **发布渠道**:证据齐后走 `uv publish` 到 pypi 或内部索引;发布前补
   release.md 第 2 节的隔离安装冒烟自动化。

每条动手前:更新本文件状态,证据落到 PR 描述或 `docs/evidence/`(新建)。
