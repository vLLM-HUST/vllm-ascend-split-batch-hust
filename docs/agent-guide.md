# 编程规范(agent 与新人通用)

本文是本仓库的规范性约束。目的:任何后来者(人或 agent)在这里做出的
改动都符合插件纪律,不破坏 default-off,不污染宿主。

## 1. 硬约束(违反即返工)

1. **宿主只读**:`/vllm-workspace/vllm-hust` 与 `/vllm-workspace/vllm-ascend-hust`
   禁止修改。宿主能力只走官方公开面:`vllm.general_plugins` entry point、
   `--worker-cls` 子类、`--additional-config`、官方 CLI/env。
   (旧基线 `/vllm-workspace/vllm`、`/vllm-workspace/vllm-ascend` 已于 2026-09-10 删除。)
2. **不注册 `vllm.platform_plugins`**(已被 vllm-ascend 的 AscendPlatform 占用)。
3. **default-off**:任何新能力关闭时必须与原生行为零差异。新开关一律
   env-gated、默认缺省,patch 方法未开启时逐字委托原实现。
4. **manifest 纪律**:bundle 保持 `import_only`,直到三项证据齐备
   (default-off 零回归冒烟、正确性对齐、性能对比);`host.version_range`
   明确锁定已验证区间,禁止写 `>=0`。
5. **provenance/ 只读**:提取历史代码必须保留原版权头(Huawei, Apache-2.0)
   与 commit 溯源。
6. **不做上游 PR(2026-09-11 定)**:改动只在本地落地(本仓 + 本地 clone/PYTHONPATH
   覆盖),不向上游(vllm / vllm-ascend 社区)提补丁,也不做"推动宿主提供 typed
   contract"这类上游化动作。与上游的关系只有一条:**跟随 `vllm-hust` release
   节奏做版本同步**(重新锁定基线与 `host.version_range`,见 release.md §4)。
   `docs/upstream/` 素材冻结留档,不再维护、不提交。

## 2. 开发工作流

| 改动类型 | 必要检查 | 不应自动运行 |
|---|---|---|
| 纯文档、结果、证据指针 | 相关格式、引用、哈希、Git 范围 | 全仓 pytest、模型 smoke、起服、L3、screen |
| 流程/工具，不改模型数值路径 | 受影响 CPU 回归、相关静态检查、有效失败码传播 | 模型 L3、模型/包加载 smoke、完整 screen |
| 实际加载、执行分支、输入输出协议或比较语义 | 定向回归及对应正确性检查；加载变化才补对应加载 smoke | 无关模型测试、重复 smoke |
| 算子、权重处理、调度执行顺序 | 定向测试、必要 L0、plain L3 + compare；成熟后必要 light L4 | 无端到端问题的 L1/L2 screen、反复 trace |
| 广泛公共接口/共享基础设施、依赖/打包、发布/合入要求、定向覆盖不足 | 写明触发原因，在稳定源码快照集中全量验证；发布门见 release.md | 每个实验/JSON/提交重跑全套 |


开发只跑受影响的定向测试与静态检查，失败后定位子集，不从头重放全套。
不得删除有效用例、降低数值判据或吞掉失败来省时。L3 失败不能被 pytest、
coverage、吞吐或其他通过项抵消；失败证据保留，能力保持 default-off。

必要全量结果记录源码范围/哈希、环境、命令与结果，三个并行会话共享。
相关源码和环境不变时，不因证据、候选指针、提交哈希或文案更新重跑；
改变相关源码/环境后按影响判断失效。既有发布/合入准入不变，不能以路径过滤
或结果复用将未验证源码声明为三腿全绿。缺有效快照时补齐准入要求。

L1/L2 screen、服务 smoke、包加载 smoke 各须回答独立问题；无打包/注册/加载变化
不重复包加载 smoke。`tools/bench-ladder.sh` 仅在满足其目标 serving 主机前置条件时作为端到端入口，
不是 clean-checkout CPU 测试，也不是算子循环默认步骤。
保留 `/health`、`99255c0` 启动 PID 归属守卫与回归。graph/system 的完整 screen
采集门不变，不得伪装 operator 路由绕过。历史数据不得为新流程重采。

Extension CI 的 push/PR 路径过滤仅排除 Markdown 与 `docs/evidence/` 下的
JSON/TXT/LOG 结果；混合源码改动、测试、workflow、依赖或打包变动仍运行原三腿。
发布工作流不使用此过滤，保留全量 pytest、lint、构建与包检查。

正确性(token 对齐/容差声明)与性能证据留在 `docs/evidence/` 或 L2；
形状事实与权重标签必须随证据保留。
设备验证仍需按原审批及实验协议执行。结果先写到仓库外，审核后发布到
`vllm-hust-benchmark`；本仓只保留固定版本链接、解析工具和测试。
失败记录与适用边界不得丢弃。历史原件见 [benchmark 归档](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git)。


## 3. 代码规范

- Python ≥3.10 目标;ruff `line-length = 88`,rules: `B, E, F, I, SIM, UP`
  (见 `pyproject.toml`,受影响路径的 `ruff check` 必须通过；发布保留全仓检查)。
- 纯逻辑与设备路径分离:shape 推导、gate 判定、planner 等写成 CPU 可测的
  纯函数;设备路径用 mock 测 + 真机冒烟。
- patch/包装代码集中放在对应插件模块目录中，不要散落新文件；
  新 monkeypatch 点必须同步更新 `HOST_CONTRACT.md` 与 `docs/release.md` 第 4 节。
- 不引入新三方依赖;确需引入时先在 PR 里说明理由并更新 `pyproject.toml`。

## 4. 测试规范

- 每个源文件有对应 `tests/test_*.py`;manifest 类不变量(版本一致、
  activation.environment 可注入、bundle 不可激活)有守护测试,改 manifest
  必须让对应测试表达新语义。
- 测试默认跑在 CPU(env 未装 torch_npu 也能全绿);需要设备的测试显式
  skip,不在 CPU 门槛里假装通过。
- 对照测试(与参考/基线实现)写明数值容差,纳入可自动跑的范围。

## 5. Git 与网络

- commit 原子:一个逻辑变更一个 commit,不混格式化与功能修改。
- message 用 conventional 风格:`feat(plugins): ...`、`fix(plugins): ...`、
  `docs: ...`、`test: ...`。
- clone/push 统一走 **SSH over 443**(一次配置永久生效):流程见工作区
  `/vllm-workspace/docs/git-remote-and-network.md`;gh-proxy 前缀仅留给 pip git 依赖。

## 6. 文档更新义务

改了什么就要同步它的权威文档(分工表见 [README.md](README.md)):
行为/开关 → 根 README;patch 面 → HOST_CONTRACT.md + release.md;
新坑 → pitfalls.md;里程碑/计划 → progress.md。agent 在收尾时自查一遍。
