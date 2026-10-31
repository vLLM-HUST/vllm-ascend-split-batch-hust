# 容器与插件边界规范：运行时分层、生命周期自愈与证据隔离规范
*(Container Storage Lifecycle, Plugin Carrier Boundary and Evidence Isolation Specification)*

- **版本**: v1.0
- **状态**: 权威规范 (Normative Specification)
- **适用范围**: `vllm-ascend-split-batch-hust` 插件仓、宿主 Fork 仓 (`vllm-ascend-hust`) 及相关容器运行时环境

---

## 1. 规范背景与核心定位

在昇腾（Ascend NPU）环境下的推理加速与算子交付过程中，常见的三大痛点是：
1. **容器环境易失与自愈混乱**：在云平台切换 GPU/NPU 规格或重启 Pod 时，容器 Overlay 层重置导致依赖缺失、SSH 断连、插件注册丢失或运行时报环境假错误；
2. **架构边界混乱与运行时打补丁**：试图在插件仓中维护动态脚本直接入侵修改容器已安装的宿主源码，导致补丁版本漂移、依赖成网、无法回滚与难以维护；
3. **实验数据与代码库混杂**：GB 级别的 Profiler 数据库、离线分析 Trace、非标准测试草稿混入 Git 仓库，破坏了软件发布的纯粹性与可重现性。

本规范确立**容器内外关系**、**插件内外关系**与**证据沉淀规范**的绝对边界，作为人机协同开发与发布验收的单一事实源依据。

---

## 2. 容器内外边界规范 (Container Storage & Lifecycle Boundary)

### 2.1 存储分层：易失 Overlay 层 vs 持久化 PVC 层
在 Kubernetes / 云原生容器环境中，文件系统分为两个根本不同的物理层次：

```text
┌────────────────────────────────────────────────────────────────────────┐
│ 容器易失层 (Overlay FS, Pod 重启/规格切换即清空)                          │
│  - /usr/sbin/sshd 二进制与 /etc/ssh 配置文件                            │
│  - /run/sshd 运行锁与运行时系统用户                                      │
│  - /tmp 临时测试与缓存目录                                              │
│  - Python site-packages 中的构建产物与安装入口元数据                     │
└────────────────────────────────────────────────────────────────────────┘
                                    ▲
                         mount      │  自愈还原 / 隔离挂载
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 持久化存储层 (Persistent Volume Claim, 挂载在 /root, 规格切换不丢失)       │
│  - /root/vllm-workspace/vllm-ascend-split-batch-hust (插件 Git 仓库)    │
│  - /root/host-trees/ (宿主 Fork 源码仓库：vllm-ascend, vllm-hust)       │
│  - /root/ssh-debs/ (sshd 离线安装 deb 包)                               │
│  - /root/.ssh/ (SSH 密钥对持久化备份)                                    │
│  - /root/cannbot-knowledge/、/root/prof/ (外部知识库与原始性能 Trace)   │
└────────────────────────────────────────────────────────────────────────┘
```

- **契约规则 1（Overlay 零信任）**：代码和关键测试基准**绝不**依赖 Overlay 层的临时文件；任何假设存在于 Overlay 层的修改在 Pod 重建后必须能够被幂等自愈。
- **契约规则 2（PVC 持久化原则）**：所有重要的代码开发、基线分支、测试数据与文档**必须**存放在持久化存储（`/root`）路径下。

### 2.2 规格切换与容器重建自愈规范 (`/root/start-sshd.sh`)
当开发者或评测流程在网页端切换容器规格（例如从无卡规格切换为 `npu-2x` 规格）时，底层的容器镜像会被重新拉起。系统统一通过持久化的 `/root/start-sshd.sh` 脚本执行自愈，其职责边界定义如下：

| 自愈环节 | 执行动作 | 边界与约束 |
|---|---|---|
| **Step 1: 离线安装 SSHD** | 优先利用 `/root/ssh-debs/` 内的离线 deb 包重装 openssh-server | 杜绝外部网络拉取失败导致容器不可登录 |
| **Step 2: 恢复系统依赖** | 还原 `libwrap.so.0` 并创建 `sshd` privsep 系统用户 | 解决基础镜像中缺失底层库的 127 错误 |
| **Step 3: 恢复主机密钥与权限** | 还原稳定的 host key，并将用户私钥权限严格纠正为 `600` | 解决平台每次重建容器将私钥误重置为 660 导致 git 权限被拒 |
| **Step 4: 启动 SSHD** | 启动守护进程并在 22 端口监听 | 完成基础网络连接恢复 |
| **Step 5: 插件载体挂载** | 使用 `pip install -e /root/vllm-workspace/vllm-ascend-split-batch-hust --no-deps` 重新注册入口点 | 恢复 `vllm.general_plugins` 入口，防止新容器将环境变量判定为未知变量 |
| **Step 6: 宿主层状态确认** | 声明宿主修改由宿主 Fork（`vllm-ascend-hust`）源码承载，进行只读检查 | **绝不**运行旧式的 `apply-runtime-patches.sh`；不自动拉起模型服务（避免占用 NPU 资源） |

### 2.3 执行工作目录规范 (CWD 隔离与遮蔽防护)
- **严禁以 `/vllm-workspace` 为 CWD**：
  镜像预置的 `/vllm-workspace` 包含旧版的空目录或历史源码，若在该目录下执行 `python3` 或 `pytest`，Python 的搜索路径会将当前工作目录置于首位，从而严重遮蔽已安装的真正 Python 包，引发无法解释的符号缺失或属性报错。
- **规范执行目录**：
  - 代码修改与测试开发：`cd /root/vllm-workspace/vllm-ascend-split-batch-hust`
  - 端到端压测与基准运行：`cd /tmp`（配合显式绝对路径引用脚本，如 `bench/bench_fisampling_refresh_e2e.sh`）

### 2.4 外部知识库与原始数据物理隔离准则
- **外部知识库定位**：`/root/cannbot-knowledge/`、`/root/research/` 等外部知识库是探索分析的参考资料库，不是代码库的一部分；
- **原始 Trace 定位**：`/root/prof/` 中的全量 Profiler 数据库（`.db`）、文本日志（`.log`）体积极大（通常几百 MB 至数 GB），属于重型原始凭据；
- **物理隔离契约**：
  - **严禁**将上述目录或其下未结构化的文件直接 `git add` 提交到本插件仓中；
  - 评测中产生的离线数据存留在 `/root/prof/` 或发布至统一的外部 benchmark 归档；
  - 仓内仅接收从原始数据中提取的结构化度量指标（以 JSON/Markdown 形式沉淀到 `ops/` 模块）。

---

## 3. 插件内外边界规范 (Plugin Carrier vs Host Fork Boundary)

### 3.1 架构分工：为什么必须两层分立
vLLM / vLLM-Ascend 官方采用模块化架构，但并非所有优化都能通过外置插件直接实现。
试图用一个插件仓包揽所有事情，导致了以往在运行时动态打补丁的严重混乱。为此确立**绝对两层分工**：

```text
┌────────────────────────────────────────────────────────────────────────┐
│ 宿主层 (Host Fork: vLLM-HUST/vllm-ascend-hust @ based/image-baseline)   │
│                                                                        │
│  负责：                                                                 │
│   1. 模型前向内部私有方法的代码修改 (如 forward 内部连续张量分配)        │
│   2. 权重加载后的专用钩子 (process_weights_after_loading)               │
│   3. 内存连续化预打包 (如 D4 卷积权重预打包为 [4, 4096])                 │
│   4. 权重合并 (如 W3F2 GDN in_proj 权重在编译前的合并)                  │
│   5. 底层 C++ 运行时及计算图生命周期修复                                 │
│                                                                        │
│  形态：随宿主镜像构建原生直出，绝对不在运行时打补丁。                    │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                     标准公开接口     │  vllm.general_plugins
                     环境变量门控     │  --additional-config
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 插件层 (本仓: vllm-ascend-split-batch-hust)                              │
│                                                                        │
│  负责：                                                                 │
│   1. 官方 vllm.general_plugins 注册与加载                               │
│   2. 纯 Python / Triton-Ascend 融合算子实现 (如 fi-sampling, fi-gelu)  │
│   3. default-off 环境变量门控控制 (未开启时零差异、零导入、零开销)      │
│   4. 算子调度策略与形状匹配选择 (op_selector)                           │
│   5. 标准化四级评测验证与交付报告 (ops/<operator>/)                     │
│                                                                        │
│  形态：标准 Python Wheel 包，遵循独立插件分发规范。                    │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.2 宿主 Fork 层边界与沉淀清单
- **代码位置**：`/root/host-trees/vllm-ascend`
- **上游远端**：`git@github.com:vLLM-HUST/vllm-ascend-hust.git`
- **基线锚点**：`based/image-baseline`（精确锁定容器运行时镜像 Commit `74f0c0a27`）
- **开发分支**：`work/issue20-w3f2-host`
- **已沉淀算子**：
  - **W3F2**（Commit `ce807cd`）：实现 `process_weights_after_loading` Hook，在加载模型权重后合并 `in_proj_ba` 与 `in_proj_qkv`；
  - **D4**（Commit `deb5ff`）：在模型加载期物化 `conv1d.weight` 为连续 `[4, 4096]` 格式，消除前向传播中的 `transpose(0, 1)`。

### 3.3 插件载体层边界与标准公开接口
- **代码位置**：`/root/vllm-workspace/vllm-ascend-split-batch-hust`
- **注册入口**：严格且仅使用 `pyproject.toml` 中的 `[project.entry-points."vllm.general_plugins"]`；
- **禁止项**：
  - **严禁**注册 `vllm.platform_plugins`（已被昇腾平台占用）；
  - **严禁**在插件初始化中对未声明的宿主私有模块执行不受控的全局 Monkeypatch。

### 3.4 模块加载纪律：纯元数据发现与延迟导入 (Lazy Import)
- **元数据扫描不导入实现**：
  扩展管理器（Extension Manager）或服务起服阶段会枚举 Entry Points 收集插件列表，该过程只应读取包元数据，**绝不应触发任何重型实现代码或 PyTorch/CANN C 扩展模块的导入**。
- **顶级模块的延迟暴露规范**：
  顶层 `src/vllm_ascend_split_batch/__init__.py` 采用模块级 `__getattr__` 实现延迟加载：
  ```python
  def __getattr__(name: str):
      if name in _EXPORTS:
          from . import planner
          return getattr(planner, name)
      raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  ```
  该设计保证 `test_extension_config_boundary` 等管理面测试纯离线、零设备依赖、零实现污染。

### 3.5 彻底废除运行时补丁链条的硬性约束
- 仓库内**严禁**再引入任何类似 `apply-runtime-patches.sh`、`patch_*.py` 等试图在容器运行树上就地修改 Python/C++ 文件的自动化脚本。
- 如发现新的必须改动宿主代码的场景，统一按《宿主提案（Host Proposal）》标准在宿主 Fork 分支实现并提交 PR，严禁在插件仓私开补丁打入逻辑。

---

## 4. 证据文档规范与数据流转契约 (Evidence & Delivery Spec)

### 4.1 仓内算子模块化三件套规范
每个受控算子在 [ops/](computer:///root/vllm-workspace/vllm-ascend-split-batch-hust/ops) 目录下建立独立子目录，包含且仅包含三个标准化文件：

1. **`README.md`**：技术原理与边界说明
   - 登记 entry point、开关环境变量、优化机制、理论预期与适用边界。
2. **`REPORT.md`**：正式验收报告（Markdown 呈现）
   - 包含硬件拓扑、环境版本、张量形状事实；
   - 记录 L0 纯算子耗时对比、L3 数值等价判定、L4 链路命中次数及 Serving 端到端加速指标。
3. **`evidence.json`**：结构化机器可读凭据
   - 符合 `operator-benchmark-evidence/v1` Schema 规范；
   - 记录可用于自动化审计与看板消费的指标字段及哈希指纹。

### 4.2 根目录索引 `DELIVERY_INDEX.json` 单一事实源
- 根目录 [DELIVERY_INDEX.json](computer:///root/vllm-workspace/vllm-ascend-split-batch-hust/DELIVERY_INDEX.json) 是全仓算子交付的权威事实清单；
- 记录基线环境（2×910B2 TP2, Qwen3.5-35B-A3B）、所有算子元数据及其当前实测状态；
- 新增或调整算子状态时，必须同步更新该索引并通过测试校验。

### 4.3 四级评测阶梯数据流转
数据采集必须按阶梯递进，前置阶梯未通过绝不推进到后置阶梯：

```text
 ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
 │ L0 纯算子微基准  │ ───►  │ L3 数值正确性    │ ───►  │ L4 调用链路归因 │ ───►  │ 服务端到端实测  │
 │ 测量核函数单步耗时 │       │ 严格等价/误差门禁│       │ 验证全链路真实命中│       │ 最终 TTFT/TPOT │
 └─────────────────┘       └─────────────────┘       └─────────────────┘       └─────────────────┘
```

- **L0 失败（如耗时更长）**：标记为负收益（如 `fi_gelu`），保持 default-off，如实记录；
- **L3 失败（数值不一致）**：一票否决，直接熔断，绝对禁止进入生产评测；
- **L4 失败（未命中或未全部消除）**：检查调度与图模式，判定是否因计算图捕获限制导致。

---

## 5. 机械守卫与自动化保证

为确保本规范中的各项边界不被意外打破，仓库建立了以下自动化测试守卫：

| 测试模块 | 守卫目标与断言 |
|---|---|
| **`tests/test_benchmark_standard.py`** | 验证各算子目录三件套完整性，校验 `evidence.json` 结构合规 |
| **`tests/test_json_hygiene.py`** | 验证根目录不存在随意散落的调试 dump，验证 `DELIVERY_INDEX.json` 覆盖全部目录 |
| **`tests/test_extension_config_boundary.py`** | 验证生命周期与元数据发现阶段绝对不导入重型实现模块 |
| **`tests/test_provenance_hashes.py`** | 验证历史溯源归档提交的哈希未被篡改，保证开源合规 |
| **`tests/test_document_custody.py`** | 验证历史证据链接有效，不补造数据，确保单一事实源 |

所有提交合入 `main` 前必须通过上述全部机械守卫测试（`pytest -q -m "not host_tree"` 保持 100% 绿灯）。
