# 架构与契约

## 1. 插件边界

本仓是 vllm-ascend 的插件壳与算子载体，不是宿主 fork。
宿主源码只读；接入仅使用 `vllm.general_plugins`、`--additional-config`、
`--worker-cls`、官方 CLI/env。不注册 `vllm.platform_plugins`，不向上游提 PR；
版本同步跟随 vllm-hust release。

```text
vLLM / vllm-ascend（只读）
    ↓ general_plugins / 已审查的宿主接缝
插件 Adapter：门控、元数据转换、能力检查、回退
    ↓ 明确的 Tensor / shape / dtype / layout / state 语义
算子 API 与实现选择
    ↓
本仓 Python / Triton 实现，或独立交付的编译型算子
```

> 验证域随具体组件和证据而异，不能由架构图推断。当前兼容声明及 cohort 边界见 [release.md](release.md) 与 [cohort-and-ownership.md](cohort-and-ownership.md)。

加载入口不等于稳定执行接口。对私有方法的运行时包装仍是宿主耦合，
不得称为宿主主动调用的公开 Backend 契约。新增替换必须证明真实 callsite 可达，
不能以注册成功或复制整个 forward 代替。


## 2. 已撤除实现

Cascade 不属于当前包：无源码、注册入口、执行路径或 kernel-wheel 依赖。保留的历史报告只用于溯源，不代表当前能力或收益。安装/升级及进程重启边界见 [release.md](release.md)。

## 3. Extension Bundle 语义

| 对象 | 契约 |
|---|---|
| `vllm_hust.extension_bundles` | 保留 6 个 bundle 的发现入口；发现只读元数据，不导入执行实现 |
| `org.vllm-hust.split-batch-full-graph` | 仅保留 `dual-pad-planner`，`import_only`，activation environment 为空，管理器不得激活 |
| 其他 bundle | 按自身 manifest 状态、环境变量与 host 范围判定 |
| `implementation[].status` | 未具备三项验收证据的载体保持 `import_only` |
| `activation.environment` | enable 时注入的真实值，不是说明文本 |
| `host.version_range` | 保持有界的宿主声明，不得写 `>=0`；范围与验证域见 release.md |
| `protocols[].version_range` | 四项 split-batch 提案协议为 `null`，不是宿主已实现的独立版本化接口 |

planner 是纯规划与预检，不执行 split-batch，不改变宿主行为。
运行时接入仍依赖 `HOST_CONTRACT.md` 中的 runtime-key、split-context、graph-pool、
split-executor 提案，不得通过翻转 manifest 状态绕过。

## 4. 门控与执行适配

- 各现有插件保留自己的环境门控与回退语义，不新增全局开关。
- `aot-cache-guard` 保持既有默认开、可显式关闭的前置修复语义；其他开关
  以支持矩阵为准，不把整个包笼统描述为所有能力默认关闭。
- 算子选择只能在已验证的 shape、dtype、layout、phase 和 state 语义范围内执行。
- 不支持的输入优先在执行前回原生路径；有状态算子不得在部分修改状态后无恢复地重试。
- 现有 monkeypatch 面以 `HOST_CONTRACT.md` 为准；宿主升级须核对参数与调用约定，
  不以符号存在替代兼容性验证。
- GDN opaque core 内部子图不可见时，不注册不可达的外部融合；tier-2 接缝独立交付。

## 5. 算子实现与证据

- Python / Triton 载体在本仓；Triton lazy import，关闭时不引入设备实现。
- CCE / Ascend C / CATLASS 等编译型实现独立交付，不在本仓增加 C/C++ 构建。
  独立实现不得反向依赖插件壳或 vLLM 内部对象。
- `op_selector.py` 是 CPU 可测的选择骨架，尚未接线真实算子路径；不能将其存在
  当作 Backend 已集成。接口见 `design/operator-dispatch.md`。
- 正确性与性能声明必须绑定形状事实、权重标签、宿主版本及 cohort；不同载体和历史域的验收不能互相继承。
- `tests/test_plugin_registration.py` 覆盖当前安装包的入口面及已撤除模块隔离；该回归不替代 NPU dispatch、数值、graph replay 或服务性能验证。
