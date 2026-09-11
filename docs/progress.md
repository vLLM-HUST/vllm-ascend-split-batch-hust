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
- kernel wheel 软依赖 + fail-open(W4 收尾,2026-09):`b1e7e61` 声明
  `kernels` extra(钉 `ascend-kernel==2026.3.9`);wheel 探针守卫——
  缺失/注册失败时 cascade 整体禁用 + 单条 warning(探针/eager gate/twin
  capture/gate bench 子进程四处守卫),单测
  `tests/test_cascade_fail_open.py`(含删 wheel 子进程等价场景);
  manifest 保持 `import_only`,翻 active 三项证据归档于工作区
  `knowledge/evidence/cascade/`(仓库外,不入库)。

## 2. 现状(2026-09-09)

- 分支 `feat/cascade-attention-plug`;W4 后 cascade 两个 carrier 已翻 **active**
  (`8751839`,证据复审通过),planner 保持 `import_only`;能力仍 default-off。
- **基线切换(2026-09-09)**:工作区默认环境换为 conda `hust`——vllm-hust v1
  (`0.28.1.post1.dev143`)+ vllm-ascend-hust main(`0.25.1rc2.dev125+hust`)++
  CANN 9.1.0。全部锚点在新宿主重核存活(runner 五方法经 `GPUModelRunner`
  基类继承,位于 `vllm/worker` 侧;`acl_graph.py` 四符号同模块;spec_decode/
  eplb 只缺 `scipy`+`decorator` 依赖,已补)。manifest `host.version_range`
  重钉为实测点 `==0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`
  (packaging 有序比较符拒 local,故用点 `==`);`extension check` 由
  INCOMPATIBLE 转 **compatible+degraded**。
- **cascade 降级回 `import_only`**(2026-09-09):W4 的 active 翻转基于
  0.23.0rc1 三项证据,不跨宿主继承;按纪律降级,回 active 路径 = 在本基线
  重跑 release.md 三项证据(default-off 零回归冒烟 / 正确性对齐 / 性能对比)。
  kernel wheel 已按 CANN 9.1 重编并全绿(S1 逐 bit、30/30、54/54、56/56、
  fastpath、S4 10/10),证据见 cascade-merge-op `73bf96a`。
- e2e 证据锚点见根 README(16k 段 −21%~−38% 等,kernel README §6 有单算子锚点);
  测量报告在 `knowledge/evidence/c3-legacy/`(工作区)。**换环境后性能数字降级为参考。**
- **新基线三项证据重验(2026-09-09)**:证据① default-off 冒烟 **PASS**;证据②
  正确性 **部分**——图模式车道首次未 shim 跑通(原为引擎无法启动),确定性 0/64、
  自然答案 63/64 通过,但强制续写形态发散 21–22/64 高于历史 ≤8/64 口径(字面形态
  1/64 通过;发散臂集合修复前后一致,非修复引入);证据③ 性能 **部分**——拓扑复现
  (4k 亏 +12.3%/+6.7%、8k −3.5%/−13.5%、16k −20.2%/−31.6%,区间均分离),但
  `(64,4096)` 亏损未被自适应 gate 覆盖(gate 微基准判 `on`)。**故 cascade 保持
  `import_only`,不翻 active**;详情 `knowledge/evidence/cascade/
  EVIDENCE-V1-BASELINE.md`。
- **D1–D4 宿主签名漂移已修复**(`ddc0120`):`_capture_cudagraphs` 缺 `profiler`、
  `_update_full_graph_params_if_needed` 多传 `positions`、`update_graph_params` 多传
  `num_dcp_pcp_tokens`、`_model_forward` 参数约定。所有包装器改 `*args/**kwargs`
  原样转发 + try/except 单次 warning 委托(fail-open 契约兑现);新增
  `tests/test_host_signature_drift.py`(13 例,含变异测试证明有效)。
- **G6 gate 覆盖缺口已修复**(`c4121e2`,2026-09-10):证据③暴露的 `(64,4096)`
  亏损格 bench 判 `on` 不回落。归因:gate 微基准探测用逐请求不相交 block table
  (W0 防故障约束),测的是无 prefix-cache 复用世界,4k 共享前缀在引擎内可被 FULL
  路径经 L2 复用,cascade 流量优势归零;且 `(64,4096)` 与 `(32,8192)` 留档 bench
  margin(+10.7% vs +10.9%)不可分离,单一阈值无解。修复:分档 margin——
  `P≤4096` 桶要求 ≥25%(`SMALL_PREFIX_MARGIN`,env 可调),其余桶维持 2%;
  默认 `MIN_PREFIX=8192` 不 bench 出小桶,默认行为零变化。验证:留档 8 格回放
  8/8;NPU 4 腿重验(预注册判据)——4/4 腿 `(64,4096)→OFF`,p420_b64 亏格收回
  (+6.7%→+2.1%),8k/16k 赢格保留(−14.5%/−32.5%),`(128,4096)` 大 margin 格
  不受影响,0 TypeError;cascade key hit 379→254(Δ≈125,step 级关闭佐证)。
  证据:`knowledge/evidence/cascade/section3-gatefix-addendum.md`
  + `logs/v1-ev3-gatefix/`(含跑前预注册)。**G6 关闭;翻 active 剩余阻塞 = 缺口 A**
  (stand-in 形态正确性口径,等真模型或团队决策)。
- **W3.1 fi_gelu(gelu_and_mul triton 融合)已结题**(`51f5818`,2026-09-10):
  exact-erf 融合 kernel(29 测)+ default-off 接线(`VLLM_HUST_FI_GELU`,patch
  `GeluAndMul.forward_oot`,13 测)+ eager/图/e2e 三段 bench。结论:**能力落地、
  默认关闭、不建议生产启用**——triton eager launch ~110µs 主导(eager 负结果);
  图模式 kernel 可捕获且 replay bit 精确,B=64 单算子 −31%(23.7 vs 34.2µs)但
  e2e A/B 噪声带内不可判(Δmedian −0.311% < spread 1.95%,预注册投影 0.2–0.6%)。
  报告 `docs/evidence/w31-fi-gelu/REPORT.md`;附带发现宿主基线缺陷
  FULL_AND_PIECEWISE 启动崩(见 pitfalls §1.3)。
- CPU 门槛:`pytest -q` **153 passed**(2026-09-10,+29 gelu kernel +13 接线)
  + `ruff check` 干净。
- **cascade 翻 `active` + release.md §3 启用验证通过**(2026-09-11):manifest 两个 cascade
  carrier(`cascade_plugin:load` / `cascade_graph_plugin:install`)由 `import_only` → `active`,
  planner 保持 `import_only`;三项证据在新基线齐备(真模型 6/64 见
  `section2-real-model-rerun.md`,gatefix 见 `section3-gatefix-addendum.md`)。
  `extension check` 转 **compatible+configured**,`run --dry-run` 注入
  `VLLM_ASCEND_ENABLE_CASCADE_DECODE/GRAPH=1`;真模型 serve 启动日志 2 条
  `cascade plugin loaded (gate=1, graph_gate=1, kernel_wheel=ok)`,cascade twin
  `capture body SUCCESS` ×144、0 TypeError,`/health` 200、chat 冒烟 200。发现:cascade
  图车道在 `--gpu-memory-utilization 0.85` 下需把 `cudagraph_capture_sizes` 收到
  `[32,64,128]`(默认全网格 twin capture 触发 NPU OOM → capture 级 fail-open)。CPU 门槛
  `pytest -q` **190 passed** + `ruff check` 干净。详情
  `knowledge/evidence/cascade/section4-active-enablement.md`。

### 阶段 5:fi_sampling 采样接入(W2b,2026-09-09)

- 落点:`fi_sampling_plugin.py`(`vllm.general_plugins` entry `fi-sampling`,
  类级替换 `AscendTopKTopPSampler`)+ `fi_sampling_route.py`(纯逻辑分流/回退)
  + vendor 目录 `fi_sampling/`;开关 `VLLM_HUST_FI_SAMPLING`(默认 0)。
- 分流:无截断 → FI api1;k=1 → argmax;B≥256 且 k≥32 且 top-p → FI joint;
  其余 → fork 链。五类强制回退(per-request generators / processed_logprobs /
  batch-invariant / reduce-sample / async-exponential)全部落地且**零设备同步**。
- e2e 证据(见 `docs/evidence/w2b-fi-sampling/REPORT.md`):无截断 on/off 两轮
  median TPOT 差 **−0.14%**(噪声带内;采样仅占 TPOT ≈1%,路由 1300/1300 已确认
  生效);joint cell 在 4k 上下文单卡形态 B_max≈122 < 256 门槛而**不可达**
  (需 ~512 上下文),即 joint 分支是安全保留的死分支。**e2e 正收益未显形,
  manifest 不翻 active。**
- 修正:初版"先读每步 top-k 主机值再判回退"导致回退腿 +4.9% median TPOT,
  已改为回退先行(由 e2e 证据倒逼,见 REPORT §4)。
- **评审修复(2026-09-09,needs_changes 的 F1/F2)**:
  - F1(宿主 import 失败崩进程 → 真 fail-open):`install()` 的宿主模块 import 与
    `_fallback_flags()` 的 batch-invariant 探测收进 try,失败 → 单条 warning +
    回 fork 链;新增 5 例 fail-open 测试(含子进程级证明)。
  - F1 追加(复审 5 项残留全部落实):全部 env 整型解析改容错 `_env_int`(非法值 →
    单条 warning + 默认值,覆盖 enable/阈值/k1/seed),`install()` 旋钮解析前移
    消除半安装窗口;新增 4 例 env 容错测试。fi_sampling 包 44 例全绿。
    `torch.softmax` 不纳入 fail-open(评审 nit:fork 链同样依赖 torch,救援无意义)。
  - F2(joint 腿证据产自修复前构建 → 降级):§3.3 对比作废(不支持"纯噪声"旧归因),
    路由可达性事实保留;REPORT §1.2 JSON 份数、§5 复现命令改为实际执行序列;
    api1 数字已用 summarize_e2e.py 对最终构建 JSON 重放复核。详见 REPORT §8。
  - F3–F10(medium/minor)留待后续批次。修复后整体原子提交。

## 3. 计划与开放问题

> 优先级以工作区 AGENTS.md 为准(flashinfer 移植/e2e 为主线,本仓库是承载壳)。

1. **翻 `active` 的证据门槛**:default-off 零回归冒烟、正确性对齐、性能对比
   三项归档后,按 release.md 第 3 节走启用验证,再改 `implementation.status`。
2. ~~**kernel wheel extras 联动**~~(已完成,见阶段 4 W4 条目):`kernels`
   extra 已声明,fail-open 守卫与单测已落地。
3. **宿主 seam 上游化**:split-batch 四协议仍是提案;推动 vllm-hust 宿主提供
   typed contract(参照 bidkv/victim-selector 的宿主适配路径),摆脱
   monkeypatch 弱契约。
4. **发布渠道**:证据齐后走 `uv publish` 到 pypi 或内部索引;发布前补
   release.md 第 2 节的隔离安装冒烟自动化。
5. **fi_sampling e2e 收益显形**(W2b 遗留):无截断路径在 B=64 下采样仅占
   TPOT ≈1%,需在采样占比更大的形态复测(大 B 短上下文 / 更小模型 /
   用单算子比例解析外推);joint cell 需 ~512 上下文才能达 B≥256,若要在常规
   形态显形须下调 `VLLM_HUST_FI_SAMPLING_JOINT_MIN_BATCH`,但下调前须补
   B∈[64,256) 区间证据(W2 bench 显示该区间 ours 输 1.33×–3.83×)。

每条动手前:更新本文件状态,证据落到 PR 描述或 `docs/evidence/`(新建)。

## 2026-09-11 fia-demask（decode FIA 去掩码）

- 新 carrier `fia_demask_plugin:load`（**import_only**，开关 `VLLM_HUST_FIA_DEMASK=1`，
  observe `VLLM_HUST_FIA_DEMASK_TRACE=1`），op 面 wrapper（FIA v1 + .out + workspace 查询），
  纯 decode 谓词（TND + sparse_mode=3 + pre/next=INT_MAX + 方形 int8 mask + Q_S=1/请求），
  fail-open 检查 / fail-closed 安装，default-off 零差异。
- 三级验收全绿：CPU 35 tests；逐 token 等价 8/8；e2e decode 账 B32 **+3.10%** / B64 **+4.61%** TPOT
  （r1 预热离群对称，热轮腿内 0.2%）。证据：
  `profiles/qwen14b-instruct-hotspot-20260910/probe-fia/E2E-mask-removal.md`。

## 2026-09-11 fia-demask 翻 active（独立 bundle）

- `org.vllm-hust.fia-demask` carrier 翻 active + §3 启用验证全绿：check/enabled/
  dry-run 注入 `VLLM_HUST_FIA_DEMASK=1` / 真模型 serve（与 cascade enable 态共存，
  capture_sizes 封顶）→ `de-mask applied` + `ACTIVE` 证据行 → /health → 冒烟 OK，
  0 fail-open。补账：六格 TPOT +3.3~+10.1%（geomean +6.43%）、e2e 吞吐 +3.52%。
- 证据：`profiles/.../probe-fia/E2E-mask-removal.md` §3.1、`serve_demask_s3.log`、
  review 材料 `docs/evidence/fia-demask-review.md`、上游素材 `docs/upstream/fia-decode-demask.md`。
