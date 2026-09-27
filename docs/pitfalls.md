# 踩坑史

按"症状 → 原因 → 正确姿势"组织。新坑请追加到对应小节,注明发现时间。

## 1. 环境类

### 1.1 CWD 遮蔽 vllm 包(⚠️ 唯一高频陷阱)

- **症状**:CWD=`/vllm-workspace` 时 `import vllm` 异常、`vllm serve` 报怪错。
- **原因**:`/vllm-workspace/vllm/`(源码 checkout,无 `__init__.py`)遮蔽真实包。
  该目录已于 2026-09-10 删除,但**规则保留**——工作区顶层仍不应作为 python CWD
  (任何未来的 checkout 都会重新引入同一遮蔽)。
- **姿势**:任何 python/vllm 命令**严禁以 `/vllm-workspace` 为 CWD**;到
  `/tmp`、各仓库根或子目录执行都安全(`vllm-ascend-hust` 带连字符不会遮蔽 `vllm_ascend`)。

### 1.2 triton-ascend 安装覆盖坑

- **症状**:`pip install triton-ascend` 连带拉原生 triton 3.5.0,覆盖 patched 包;
  后续任何依赖 `triton` 的 pip 安装也会重新覆盖。出现 triton import 报错先怀疑这个。
- **姿势**(重装流程,禁 `--no-deps` 之外的方式):

```bash
pip uninstall -y triton triton_ascend
pip install triton-ascend==3.2.1 --no-deps --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
pip install pytest-xdist
```

> (2026-09-10 注:hust 新基线为 **triton-ascend 3.2.2**,恢复命令的版本号以工作区
> AGENTS.md 陷阱 #2 为准,勿按上文 3.2.1 覆盖安装。)

### 1.3 默认 cudagraph 模式 FULL_AND_PIECEWISE 启动即崩(2026-09-10,新基线)

- **症状**:本基线(torch 2.13 + vllm-ascend-hust main)默认
  `cudagraph_mode=FULL_AND_PIECEWISE` 时引擎初始化失败:
  `AssertionError: expected OutputCode, got GraphModuleImpl`
  (`fusion_pass_compile`,torch 2.13 AOT-autograd 缓存打包与 torch_npu 不兼容)。
  与任何插件无关(env 不设、插件 inert 时同样崩,OFF 腿已复证)。
- **姿势**:显式 `--compilation-config '{"cudagraph_mode":"FULL"}'`
  (npugraph_ex + ACL graph,自带 AOT 缓存规避);所有 A/B 腿必须**对称施加**。
- **另注**:`VLLM_DISABLE_COMPILE_CACHE=1` 仍是必须项(共享 AOT 缓存命中硬崩,见证据③口径)。

- **副作用须知**:`triton` 包元数据不存在(`pip show triton` 查不到、`pip check`
  常驻告警),vllm 的 HAS_TRITON 检测基于 import 不受影响。裸调
  `vllm_ascend.ops.triton` kernel 前需先调 `init_device_properties_triton()`。

> **订正(2026-09-16,追加不改上文)**:本条的**根因已定位到行并给出正解**。
> - 根因:`AscendCompiler.compile` 的 `enable_npugraph_ex=False` 分支
>   (`fusion_pass_compile` → `compile_fx` → `aot_autograd`,compiler_interface.py:80/:97)
>   返回普通 `GraphModule`,而 torch 2.13 的 AOTAutograd 缓存写入要求 `OutputCode`。
>   宿主已有 guard(`compiler_interface.py:40`)但**只**在 npugraph_ex 分支被进入。
> - 上文"姿势"(`cudagraph_mode=FULL` 走 npugraph_ex)是**绕过**,且**不适用于验收冻结配置**
>   ——V3.8 表附-2 冻结 `cudagraph mode=piecewise`,正好落在崩的那个分支;`VLLM_DISABLE_COMPILE_CACHE=1`
>   也被表附-4 末行("其他执行变量｜禁止")排除。
> - **正解**:插件侧 carrier `aot_cache_guard_plugin.py`(本仓 `src/vllm_ascend_split_batch/`,
>   默认开、`VLLM_HUST_AOT_CACHE_GUARD=0` 关),只对未加 guard 的分支进入宿主自身 guard;
>   宿主将来自修则该 carrier 变 no-op(漂移守卫 `tests/test_aot_cache_guard_drift.py`)。
> - 证据:`bench/runs/20260916-abcert-b1/FINDINGS.md` §阻塞 1(真机:断言计数 0、
>   `Compiling a graph (1,32768) 13.38s` 成功);跨仓条目已在工作区 `AGENTS.md` 陷阱 9 登记。

### 1.4 模型卷只读

`/data/shared_models`(与 `/data/shared_datasets` 同卷)是只读挂载;拉新模型放
容器内可写目录。当前主力模型:Qwen2.5-Coder-14B-Instruct(单卡,48 层/hidden
5120/bf16)。

## 2. 宿主与集成类

### 2.1 monkeypatch 面是弱契约

- **症状**:宿主(vllm/vllm-ascend)升级后插件行为怪异或静默失效。
- **原因**:patch 的是宿主**私有方法**(签名无兼容承诺),契约只有
  `host.version_range` 单方面钉死。
- **姿势**:升级宿主前按 [release.md](release.md) 第 4 节清单逐项核对签名,
  差异收敛在 `cascade_runner_patch.py`;核对前不要翻 `host.version_range`。

### 2.2 microbatching 互斥

cascade 两段式与任何 microbatching(`use_ubatching`: DBO 或 `ubatch_size>1`)
互斥,插件侧镜像官方 core gate 强制回落。当前 vllm-ascend 平台层会重置
`enable_dbo` 与 `ubatch_size`,该 gate 在此宿主上是休眠的——但改宿主版本时
要重新确认(关联 2.1)。

### 2.3 kernel 侧危害外溢到插件测试

- 同进程内 `fa_fp32_stage1` 之后跑 kv≥8k 的 FIA v2 TND 块表形态会触发
  fftsplus aicore 0x800000——micro-bench 必须分组进程(gate 的
  `cascade_gate_self.py` 因此用子进程,别改回同进程)。
- FIA TND 计时类探针**必须逐请求传** `actual_seq_kvlen`（传累计值会越界读 →
  垃圾块号 → MTE DDR 越界 0x800000 或静默 NaN；2026-09-08 定案，非布局限制、
  非算子缺陷）。**任何 FIA 计时前先同数据对拍正确性,存活≠正确**。
- gate 探针的 block table 必须逐请求独立分配,shared/reused 块表会踩内存
  (历史 fix 9899f00)。
- 详见 kernel 仓库 README §4-§5。

### 2.4 宿主换图重放机制 ⇒ cascade 孪生卡死(2026-09-27 定位并修)

**症状**:目标栈(`vllm-ascend` `fbe4911bb`,自编 `0.25.1rc2.dev605+…`)上,图孪生开启时
**第一个 decode step 之后请求再也不出 token**(4 并发实测 1/4 完成、其余 >90 s 零吞吐;
引擎 CPU ~60% **自旋**,不是 I/O 等待)。`capture body SUCCESS` 照常 96 次、`cascade plugin
loaded (gate=1, graph_gate=1, kernel_wheel=ok)` 照常、`twin missing` 0 次、0 TypeError
⇒ **所有旧观测点都显示"正常",只有吞吐是 0**。关图孪生(`VLLM_ASCEND_ENABLE_CASCADE_GRAPH=0`)
则 4/4 正常(0.5–0.7 s) ⇒ 问题专属图重放路径。

**机理**:新宿主给 FULL 图换了重放机制 —— 新增 `vllm_ascend/compilation/updatable_graph.py`
的 `UpdatableGraph` 与 `use_updatable_graph`(`vllm_ascend/utils.py:1773`),`ACLGraphWrapper.__call__`
对 `FULL + use_updatable_graph(attn_backend)` 改走 `_updatable_graph_replay(...)`
(`acl_graph.py:297`),**不再**调 `update_full_graph_params`(该函数对可更新图直接
`return`);而插件的重参数化包装挂在 `impl_cls.update_graph_params` 上 ⇒ **一次都不会被调到**
(实测 `replay update` = 0;旧宿主上是 2)。孪生图里每个 stage 任务组前后都有
`ExternalEvent.wait`(捕获期写入),**只有**插件的重参数化会 `record` 这些事件 ⇒
没人 record ⇒ 图内等待永不满足 ⇒ 重放挂死。

**修法**(`cascade_graph_plugin._wrap_updatable_graph_replay`,特征探测 + fail-open):
包装 `ACLGraphWrapper._updatable_graph_replay`,在 **`orig(...)` 之前**调
`_update_cascade_on_updatable_replay`。**顺序是硬约束**:放在 `orig` 之后会挂
(实测:同一个 4 并发负载,更新放后面 ⇒ >90 s 超时;放前面 ⇒ 8/8 成功、0.2–0.4 s)。
宿主自己的 `enable_enpu` 分支也是"先 update 再 replay",插件与之对齐。

**判据(两代宿主都要跑)**:

| 宿主 | 期望 |
|---|---|
| 新机制(≥`fbe4911bb`) | trace 出现 `updatable replay: cascade update ran=True`,且 `cascade key hit` **= 0**(那是旧接缝的观测点,新机制下不再触发) |
| 旧机制(基线 `74f0c0a27`) | trace 出现 `updatable-replay seam absent on this host`,`cascade key hit` > 0,且 `cascade update ran` **= 0**(新接缝必须 no-op) |

**同源坑(诊断期踩到,值得记住)**:
- **`VLLM_ASCEND_CASCADE_MIN_PREFIX` 小于一个 block ⇒ 孪生捕获静默关闭**:
  `cascade_runner_patch.py:266-271` 里 `dummy_prefix = min(MIN_PREFIX, …)`,若 `< block_size`
  就 `return`,**无任何日志**;运行期 gate 仍判 on ⇒ 表现为"图开了但没接管"
  (`cascade twin missing … step replays the standard full-KV graph`)。冒烟想放宽运行期门槛时
  别把它设成 0,用 ≥ 一个 block(本机用 1024)。
- **挂死的引擎无视 SIGTERM**:上述死锁状态下 `kill <pid>` 无效(实测仍在、HBM 位不释放),
  必须 `kill -9`;杀完等 10 s+ 再复查 `npu-smi info -t usages -i <card>`。
- **单请求冒烟会给出假绿**:`VLLM_ASCEND_CASCADE_MIN_REQS` 默认 32(本机冒烟设 2),
  1 个请求时常 < 门槛 ⇒ cascade 未准入 ⇒ 走标准图 ⇒ **0.3 s 返回正常**,而 ≥2 并发才触发孪生。
  验证 cascade 图路径**必须**发 ≥ 门槛的并发请求。

## 3. 打包与发布类

### 3.1 manifest `activation.environment` 填文档字符串(2026-09 已修)

- **症状**:无(处于 `import_only` 时值不生效——这也是它潜伏的原因)。
- **风险**:该字段语义是 **enable 时注入的值**;填 `"0 (default) | 1"` 这类
  说明文字,翻 `active` 后会被原样注入环境变量,直接破坏 default-off。
- **姿势**:只填真实值(enable=注入 `"1"`);`tests/test_manifest.py` 有守护
  测试,值域必须是 `"0"`/`"1"`。

### 3.2 版本双写漂移

版本在 `pyproject.toml` 与 manifest `extension_version` 两处,手改漏一处会导致
`extension inspect` 版本错乱。已有测试守护(`test_extension_version_matches_
distribution_version`),发版两处同步改。

### 3.3 平台 wheel 的假 tag

携带 `.so` 的 wheel 若打成 `py3-none-any`,用户可在任何平台安装、import 才炸。
kernel wheel 由 `NpuExtension` 自动产出 `cp312-cp312-linux_aarch64` 正确 tag;
若将来手工打包,发布前必须核对 tag(见 kernel README §4)。

### 3.4 vllm-hust-ext fail-closed 常见原因

`run` 拒绝启动时依次排查:宿主版本/`api_range` 不匹配;手工设置了 manager
拥有的 `VLLM_EXTENSION_MANIFESTS`/`VLLM_EXTENSION_BUNDLES`(这两个变量归
manager,手工设置即拒绝);已 enable 的扩展被卸载(卸载前先 `disable`/`forget`)。
完整清单见 bidkv 指南 §13(vllm-hust-docs 仓库 `operations/bidkv-packaging-
and-release-guide.md`)。

### 3.5 cascade 图捕获在默认 capture_sizes 下 NPU OOM(2026-09-11)

**症状**:`vllm-hust-ext run`(cascade 已 enable)用默认
`cudagraph_capture_sizes` 起服,cascade twin capture 反复
`Tried to allocate 418.00 MiB ... 224.63 MiB free` NPU-OOM →
`cascade graph capture failed; capturing the standard graph instead`
fail-open 刷屏(143+ 次),引擎永远到不了 ready。

**修复**:启用 capture-size 封顶(§3 验证过的 harness 口径):

```bash
--compilation-config '{"cudagraph_capture_sizes":[32,64,128]}'
```

**定性**:config 面问题,非正确性缺陷——封顶后 144 次 capture(3 桶 × 48 层)
全成功、0 fail-open。§3 启用验证日志:
`knowledge/evidence/cascade/logs/v1-active-enablement/serve-active-enablement.log`。
