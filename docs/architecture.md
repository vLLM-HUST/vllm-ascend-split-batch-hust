# 架构与契约

## 1. 一句话定位

本仓库是 **vllm-ascend 宿主的插件壳**:以 default-off 方式为 vLLM 注入
cascade 两段式 decode(调度 + 图模式)能力,并承载 split-batch 规划契约的
提案描述符。算子实现在仓库外的 `ascend_kernel` wheel 里,本壳只做适配、
门控与调度决策。

## 1.1 仓库命名约定(团队沟通用语)

| 简称 | 仓库 | 远端 |
|---|---|---|
| **插件库**(本仓库) | `vllm-ascend-split-batch-hust`,唯一对外交付面 | `vLLM-HUST/vllm-ascend-split-batch-hust` |
| **算子库** | `cascade-merge-op`(ascend-kernel 工程 + S1 锚点数据,wheel 工厂) | `Raing5Days/vllm-hust-cascade-kernel` |

**插件库 ⊃ 算子库**(逻辑包含):本库经 `kernels` extra 钉版消费算子库
wheel,单向依赖;算子库对 vllm-hust-ext 与插件机制零感知。日常工作流见
算子库 `docs/workflow.md` Loop B。

## 2. 三层关系

```text
┌─ 宿主(只读,禁止修改)────────────────────────────┐
│  vllm 0.23.0 (VLLM_TARGET_DEVICE=empty)          │
│  vllm-ascend 0.23.0rc1                           │
└──────────────┬───────────────────────────────────┘
               │ 唯一官方入口: vllm.general_plugins
               │ entry `cascade-attention` → cascade_plugin:load()
┌──────────────▼───────────────────────────────────┐
│  本壳 vllm-ascend-split-batch(纯 Python)         │
│  · bundle manifest(0.2-experimental, import_only)│
│  · env 门控 + monkeypatch(见 HOST_CONTRACT.md)   │
│  · 纯逻辑 planner / gate(CPU 可测)              │
└──────────────┬───────────────────────────────────┘
               │ 单向依赖: torch.ops.npu.*
┌──────────────▼───────────────────────────────────┐
│  ascend_kernel wheel(CCE 算子, 仓库外)          │
│  fa_fp32_stage1 / lse_merge;对本壳零感知          │
└───────────────────────────────────────────────────┘
```

依赖方向铁律:**本壳 → kernel 包**单向;kernel 包禁止 import 本壳或任何
vllm 模块;宿主源码不可修改,宿主能力只走公开面(entry point、
`--additional-config`、`--worker-cls`、官方 CLI/env)。

## 3. Extension Bundle 语义(vllm-hust-ext 0.2-experimental)

| 概念 | 本仓库的取值与含义 |
|---|---|
| Bundle ID | `org.vllm-hust.split-batch-full-graph`,注册于 `vllm_hust.extension_bundles` entry point,value 指向包目录(含 manifest JSON) |
| `implementation[].status` | `active` 才可 `enable`;`import_only` 仅可 `inspect`(当前状态)。状态语义 = manifest 纪律:证据齐前不翻 `active` |
| `activation.environment` | **enable 时注入的值**,不是文档。当前声明 `ENABLE_CASCADE_DECODE=1` + `ENABLE_CASCADE_GRAPH=1`;不 enable 则什么都不注入 |
| `components[].contracts` / `execution_planes` | `dual-pad-planner` 与 `cascade-two-stage-decode` 声明在 worker/device plane,permissions 含 `device_access` |
| 发现 vs 导入 | `extension list/inspect/check` 只读元数据不 import 实现;import 实现的只有 vllm 的 `general_plugins` 加载(`load()` 内部再按 env 门控) |

manifest 与 entry point 打包完整性由构建检查保证,流程见 [release.md](release.md)。

## 4. default-off 的双层保证

1. **门控层**:所有 env 开关默认缺省,`load()` 与被 patch 方法首行检查
   gate,未开启时逐字委托原实现——宿主行为 bit-identical(README 有验证证据)。
2. **bundle 层**:bundle 处于 `import_only`,manager 不注入任何
   activation;即使用户手工装了 wheel,不改 env 也零差异。

附加互斥:任何 microbatching(`use_ubatching`: DBO 或 `ubatch_size>1`)下
强制回落标准路径,与官方 core gate 对齐。

## 5. monkeypatch 面(弱契约,重点维护区)

本壳对宿主的实际控制面 = 以下方法/对象的运行时包装(权威清单与约束见
[HOST_CONTRACT.md](../HOST_CONTRACT.md)):

- `vllm.v1.cudagraph_dispatcher.add_cudagraph_key` / `dispatch`
- Runner 私有方法:`_capture_cudagraphs`、`_warmup_and_capture`、
  `_determine_batch_execution_and_padding`、`_model_forward`、
  `_update_full_graph_params_if_needed`
- vllm-ascend:`update_full_graph_params` / `get_graph_params` / `GraphParams`
- `ACLGraphWrapper` variant-entry 侧表(按标准 `BatchDescriptor` 键)

这是与 bidkv 类"宿主主动调插件"的 typed contract 最大的不同:**宿主没有
给我们的 seam,契约靠我们单方面遵守 + `host.version_range` 钉死**。宿主
升级时必须按 [release.md](release.md) 第 4 节逐项核对签名。

## 6. 与 kernel 库的边界

- 调用面只有 `torch.ops.npu.fa_fp32_stage1` / `torch.ops.npu.lse_merge`
  (`import ascend_kernel` 即注册)。
- Tier1(fp32 stage-1)依赖该 wheel;缺失时插件必须 fail-open 降级而非报错。
- kernel 侧的兼容四元组(平台 tag / PyABI / torch_npu / CANN)、链接纪律、
  已知危害,见 kernel 仓库 README §4-§5(工作区
  `cascade-merge-op/ascend-kernel/`),本仓库不重复维护。
