# 踩坑史

按"症状 → 原因 → 正确姿势"组织。新坑请追加到对应小节,注明发现时间。

## 1. 环境类

### 1.1 CWD 遮蔽 vllm 包(⚠️ 唯一高频陷阱)

- **症状**:CWD=`/vllm-workspace` 时 `import vllm` 异常、`vllm serve` 报怪错。
- **原因**:`/vllm-workspace/vllm/`(源码 checkout,无 `__init__.py`)遮蔽真实包。
- **姿势**:任何 python/vllm 命令**严禁以 `/vllm-workspace` 为 CWD**;到
  `/tmp`、各仓库根或子目录执行都安全(`vllm-ascend` 带连字符不会遮蔽 `vllm_ascend`)。

### 1.2 triton-ascend 安装覆盖坑

- **症状**:`pip install triton-ascend` 连带拉原生 triton 3.5.0,覆盖 patched 包;
  后续任何依赖 `triton` 的 pip 安装也会重新覆盖。出现 triton import 报错先怀疑这个。
- **姿势**(重装流程,禁 `--no-deps` 之外的方式):

```bash
pip uninstall -y triton triton_ascend
pip install triton-ascend==3.2.1 --no-deps --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
pip install pytest-xdist
```

- **副作用须知**:`triton` 包元数据不存在(`pip show triton` 查不到、`pip check`
  常驻告警),vllm 的 HAS_TRITON 检测基于 import 不受影响。裸调
  `vllm_ascend.ops.triton` kernel 前需先调 `init_device_properties_triton()`。

### 1.3 模型卷只读

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
- CANN FIA v2 的 TND 均匀变长+无 mask 角落会静默 NaN:**任何 FIA 计时前先
  同数据对拍正确性,存活≠正确**。
- gate 探针的 block table 必须逐请求独立分配,shared/reused 块表会踩内存
  (历史 fix 9899f00)。
- 详见 kernel 仓库 README §4-§5。

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
