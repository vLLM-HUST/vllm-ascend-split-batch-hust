# 原生路由请求探针执行计划（native-route-probe）

入口：`tools/operator_tools/native_route_probe.py`；定向测试：`tests/test_native_route_probe.py`。
选题登记：[docs/optimization-next.json](../../docs/optimization-next.json)、
[docs/optimization-history/gmm1-swiglu-screen.json](../../docs/optimization-history/gmm1-swiglu-screen.json)。
本计划只覆盖一个诊断动作：以原生 `--enable-return-routed-experts` 起一次同工况服务，
发与历史参考档同工况的一批请求，逐请求核对逻辑 top-k 与贪心文本；不是性能实验、
不是服务收益对比、不写 `ops/`、不改索引、无第二阶段自动开启口子。
旧诊断止步于 `starting_native_service`（登记 `route-status.json`），无请求级回执；
本探针当前仍未上卡。

## 1. 命令

CPU 只读自检（默认路径；不占卡、不起服、不写原件，任何检出可先跑）：

```bash
python -B -m tools.operator_tools.native_route_probe
```

占卡命令（评审通过后执行一次；限共用锁的权威 checkout）：

```bash
/root/venvs/split-batch-runtime/bin/python -B -m tools.operator_tools.native_route_probe --run
```

输出归档：`/root/vllm-workspace/benchmark-archives/native-route-probe-<UTC 时间戳>-p<pid>`
（自动分配、目录已存在即失败；或 `--archive-dir` 指定归档根下不存在的受控子路径）。
设备号与端口如显式给出，必须与当时真实映射/空闲值一致，不接受猜测值。只此一个窗口。

## 2. 准入（全部满足才允许执行）

1. 审批：`docs/optimization-next.json` 的 `reviewed_plan` 登记为
   `{"path": "experiments/native-route-probe/EXPERIMENT_PLAN.md", "sha256": <本文件 SHA256>}`；
   工具复用 `tools/operator_tools/next.py` 的 `check_hash` 逐字核对，路径或指纹不符即拒绝。
   测试、ruff、离线检查通过不构成评审；CLI 无自我放权开关。
2. 源码不漂移：本文件 §5 冻结的 `native_route_probe.py` SHA256 与仓内实际文件一致。
3. 权威 checkout 与干净树：`--run` 只绑定共用 `serving-longrun.lock` 的权威
   checkout `/root/vllm-workspace/vllm-ascend-split-batch-hust`——独立 clone 或
   worktree 的同名锁文件互不相干，一律拒绝；CPU 自检不需要该约束。
   且 `git status --porcelain` 为空。
4. 只读自检通过：`python -B -m tools.operator_tools.native_route_probe` 退出码 0
   （含登记 SHA256、双宿主一致、工况三源一致、guard 含路由 flag、归档宿主预检回执）。
5. 设备与工况收窄：`npu-smi info -m` 显示恰好两张计算卡映射当前容器逻辑 0/1，
   且为两张不同物理卡；`npu-smi info` 中两卡均无进程占用。设备号只用于忙卡
   检查与映射核对，不作运行时可见性掩码。本计划只覆盖容器直通两卡且
   `ASCEND_RT_VISIBLE_DEVICES` 键未设置的工况；该键只要已设置（含空串、纯逗号、
   单卡、物理/逻辑号）一律拒绝，不因字面可转换而放行，也不覆盖、不建议 unset
   绕过平台分配；平台必须设置该键时，停止执行并回 CPU 另核该映射，不猜物理/
   逻辑号。`--devices` 仅是显式物理卡号核验，取值必须严格 `N,M` 纯数字
   （无空格/空字段/尾逗号/数值重复），缺省不指定、按当前映射自动核对。

## 3. 只读输入与运行内容（锁内预检到服务退出）

只读输入先按登记指纹核对：筛查记录；诊断归档的 `config.json`、`diagnostic-plan.json`、
`host_preflight.json`、`route-status.json`；参考归档的 `collection_manifest.json`、
`round_00_off/invocation.json`、`round_00_off/profile_responses.json`。

1. 锁内预检顺序：设备预检（写 `npu-before.txt`、`device-mapping.txt`）→
   gate=0 并入 `config["environment"]`（`verify_host` 与服务看到同一最终关键环境）→
   `serving_collect.verify_host` → 新端口（不复用历史端口）→ 起服。
2. 起服：`serving_collect.server(profile=False, arm='off')`；guard flags 并入真实
   `server_command`（不并入就会丢路由 flag），全程不开 profiler；合并后命令与参考
   invocation 截去 `--profiler-config` 起的 profiler 尾段（含其后配置项）、端口归一后
   逐字一致（工况一致判据）。服务解释器 `PYTHONPATH` 无仓库源码路径、无空组件。
3. 请求：先按登记工况 warmup，再以 `build_profile_prompts` 提示、登记 token 数与
   并发发一批请求。每条请求完成即存 `requests/request-NN.json`（经临时文件 rename
   落位，中断不留半个有效命名件）；同批不因一条失败丢其他原件，不重试、不换卡换
   端口、不做第二窗口。
4. 校验（严格，任一失败即失败）：每条 `routed_experts` 为非空 NPY base64，解码失败
   须给出含请求 id 的清晰错误；形状 `(prompt_tokens + completion_tokens - 1) ×
   model_layers × top_k`；整数 dtype；值域 `[0, expert_count)`；同一 token×层 top-k
   无重复；请求 id 唯一且单 choice；逐请求 `prompt_tokens` 与参考档一致（提示集不同
   即不可比）；按请求顺序逐条核对贪心文本；参考档不得携带 `routed_experts`
   （防空数据假阳性）。numpy 只在解码 payload 时导入。
5. 收尾：`native/exit.json` 五字段齐全且为真布尔；`returncode` 只接受整数 0 或
   SIGTERM 请求关闭的 -15（真 int，不接受 bool/浮点）；
   `returncode_before_shutdown` 必须为 None 且 `shutdown_requested`
   为真（不得提前退出）；不得 forced kill；`collection_completed` 必须为真。
   `logical-topk.json` 只标记请求级（`per_step_rank_verified=false`），无性能数字。

## 4. 打断、失败与互斥边界

- 打断（SIGTERM 与 1350 秒闹钟抛探针取消异常；SIGINT 走 KeyboardInterrupt，
  同一清理路径）：取消异常立即向上传播，不被起服健康重试当超时吞掉；
  未启动的请求直接取消，在途请求按其自身请求超时返回，闹钟时限不是硬上界
  （该快速返回只用于本探针的留档 sink 路径；serving_collect 既有调用方失败时
  仍等池退出再交还）；向本探针所属服务进程组发 SIGTERM 尽力清理，等待上限
  30 秒，超时硬杀；硬杀一律不判成功。
- 任一核验失败：非零退出，保留已落盘原件（含逐请求留档）与阶段状态；不删除、不重跑、
  不降级、不放宽判据。
- 互斥范围（如实声明）：独占 flock `.openbitfun/serving-longrun.lock` 只保证与同根使用
  该锁的 launcher（serving_longrun、serving_e2e_auto 等）互斥，不保证与其他根或其他
  工作区的工具互斥；因此 `--run` 绑定共用该锁的权威 checkout（见 §2.3），CPU 自检
  不受此限。不向全局有卡队列派活。
- 无扩采：本计划不含第二窗口、profiler、合成 dispatch、性能采样或服务 A/B；任一项
  须另行评审立项。探针不写 `ops/`、不回填索引、不改宿主环境。

## 5. 源码摘要（防漂移；改动后须重新冻结并重新评审）

```text
native_route_probe.py sha256 b1727fe6e102ed8367c6db6ac6db7502cc770e723123da50110a0442cffda7b5
```
