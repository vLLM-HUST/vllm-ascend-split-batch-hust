# 华为昇腾 Ascend 910B 系列高级算子优化工程指南
### —— 基于 NVIDIA SM110 (Thor/Blackwell) 极致优化经验的体系结构映射与推测性演进

- **文档定位**：技术架构映射与推测性高级算子优化指南
- **目标平台**：Ascend 910B 系列（DaVinci V300 / Cube Core + Vector Core 分离微架构，Ascend C 编程模型）
- **经验输入源**：NVIDIA SM110 (Thor) 边缘平台上的大模型端到端推理优化实践（涵盖 PDL 硬件依赖流水、CUTLASS 3.x Epilogue Visitor Tree 融合、Cluster 分布式共享内存跨 SM 规约、TMA 异步搬运与 Warp-Specialized 专职分工等实战成果）

---

## 一、核心体系结构深度映射：NVIDIA SM110 vs. 华为昇腾 910B

在 GPU（尤其是 SM90/SM110）上，算子优化的核心手段围绕 **TMA（异步拷贝）+ Warp-Specialization（分工流水）+ Shared Memory/DSMEM（片上交换）+ Tensor Core/CUDA Core** 展开。

而在华为昇腾 910B 上，其底层是基于**达芬奇（DaVinci）架构**的专用 AI Core，核心特征是 **Cube Core（矩阵计算单元）与 Vector Core（向量计算单元）硬隔离分立、显式片上多级 SRAM 缓存、以及依赖硬件多端口流水线同步队列（Queue）驱动的显式异步流水线**。

| 概念维度 | NVIDIA SM110 (Thor / Blackwell) | 华为昇腾 Ascend 910B (DaVinci Core) | 架构差异与优化启示 |
|---|---|---|---|
| **编程模型与抽象** | CUDA / CUTLASS 3.x / CuTe (Layout & Tensor) | CANN Ascend C (C++ 原生扩展，基于 SPMD 与流水线 API) | Ascend C 是显式多级乒乓流水（CopyIn $\to$ Compute $\to$ CopyOut），无隐式线程调度 |
| **计算核心形态** | Tensor Core (MMA) + CUDA Core (ALU) 混排在同一 SM 流水线 | **Cube Core**（3D 矩阵乘单元）与 **Vector Core**（1D/2D 向量计算单元）物理分立 | 矩阵与向量在 910B 上是不同硬件执行单元，为 **Cube-Vector 异构并发** 提供了硬件级天然隔离 |
| **片上显存层次 (SRAM)** | 统一 Shared Memory / L1 Data Cache (单 SM ~256KB) + DSMEM | **显式分层片上存储**：<br>- L1 Buffer (片上全局/Cube输入)<br>- L0A / L0B (Cube 乘数专用寄存器/缓存)<br>- L0C (Cube 累加器 Accumulator)<br>- Unified Buffer (UB，Vector 计算与搬运核心中枢) | 910B 无法直接在 Cube 内部做任意复杂的逐元素计算，必须通过内部总线流转至 UB 或经过 L1 缓冲 |
| **数据搬运机制** | TMA (Tensor Memory Accelerator)，硬件异步 1D~5D 拷贝 | **MTE (Memory Transfer Engine)**：<br>- MTE1 (L1 $\to$ L0A/B)<br>- MTE2 (Global/HBM $\to$ L1/UB)<br>- MTE3 (UB/L0C $\to$ Global/HBM) | 910B 具有极强的硬件 DMA 引擎，支持多维 Strided Copy，搬运耗时必须与计算完全重叠 |
| **异步流控与同步** | `cuda::barrier` / `mbarrier` / TMA Transaction Count / PDL (`griddepcontrol`) | **TPIPE 同步队列 (TQue / TBuf)**：<br>- `EnQue` / `DeQue` 硬件 FIFO 队列<br>- `SetFlag` / `WaitFlag` 硬件事件屏障 (Sync ID) | 910B 依赖硬件硬件级的 Flag 机制实现流水节拍控制，开销极低（几周期），无 GPU 线程发散困扰 |
| **跨计算单元互联** | Cluster Distributed Shared Memory (DSMEM, 跨 SM SMEM 直读) | AIV (Vector Core) 与 AIC (Cube Core) 内部总线共享 + 片上 L2 Cache / 多核间硬件 Barrier | 跨核协作更依赖于高效利用片上 L2 Cache 切块或通过硬件 Sync 实现跨 Core 同步 |

---

## 二、六大高级算子优化模式在 910B 上的映射与落地推测

### 模式 1：Cube 与 Vector 垂直流水线融合（对标 GPU Down GEMM + Norm 垂直流式）

#### 1. GPU 上的经验本质
在 SM110 上，我们试图利用 CUTLASS 3.x Epilogue Visitor Tree (EVT) 或 DSMEM，在 Down GEMM 的 Tile 刚离开 Tensor Core 累加寄存器时，直接就地做残差加法并跨 SM 规约算 RMSNorm，消除往返 DRAM (HBM) 的开销。

#### 2. 910B 上的体系结构机会与瓶颈
- **机会**：910B 的 Cube 算完矩阵乘（Down GEMM）后，输出处于 **L0C**。传统写法是：
  $$\text{Cube: L0C} \xrightarrow{\text{MTE3}} \text{Global (HBM)} \xrightarrow{\text{MTE2}} \text{UB} \xrightarrow{\text{Vector}} \text{Residual Add + RMSNorm} \xrightarrow{\text{MTE3}} \text{HBM}$$
  这在 910B 上造成两次昂贵的 HBM 读写，严重拖累 MTE 带宽！
- **优化路径（片上闭环）**：
  利用 910B 内部互联架构，让 Cube 产出的 Tile 直接流转至 **UB (Unified Buffer)**，由 Vector Core 在片上就地接力：
  $$\text{Cube: L0C} \xrightarrow[\text{内部数据通路}]{\text{Fixpipe / MTE3}} \text{UB} \xrightarrow{\text{Vector Core}} \text{原地加 Residual + 算局部 } \sum x^2 \to \text{Norm 缩放}$$

#### 3. 910B Ascend C 推测性语法实现方案
在 Ascend C 编程模型中，定义多阶段异构流水（Heterogeneous TPIPE）：
```cpp
// [推测性设计] Cube 计算与 Vector 规约片上垂直缝合流水
class FusedDownNormKernel {
public:
    __aicore__ inline void Process() {
        // 分块循环：每次处理一个 [M_tile, N_tile] 的 Down GEMM 分块
        for (int i = 0; i < numTiles; ++i) {
            // Stage 1: Cube Core 负责执行 Matmul (Down GEMM)
            CopyIn(i);      // MTE2: HBM -> L1 -> L0A/L0B
            ComputeGEMM(i); // Cube: L0A * L0B -> L0C
            
            // Stage 2: 垂直穿透，不写回 Global Memory，直接定向至 UB 队列
            // 利用 Fixpipe 单元直接将 L0C 数据搬入当前 Core 的 Unified Buffer
            tqueCubeToUB.AllocTensor(ubTile);
            DataCopy(ubTile, l0cTile); // 内部低延迟总线
            tqueCubeToUB.EnQue(ubTile);
            
            // Stage 3: Vector Core 原地消费该 Tile 进行残差累加与 RMSNorm 统计
            tqueCubeToUB.DeQue();
            // Vector 指令：原地 Residual Add
            Add(ubTile, ubTile, ubResidualTile, tileSize);
            // Vector 指令：计算该 Tile 对应 Token 行的局部平方和 (Row-wise Square & Sum)
            Mul(ubSquare, ubTile, ubTile, tileSize);
            RowSum(ubPartialSum, ubSquare); // 存入片上局部累加寄存区
            
            tqueCubeToUB.FreeTensor(ubTile);
        }
    }
private:
    TQue<QuePosition::CUSTOM_CUBE_TO_UB, 2> tqueCubeToUB; // 双缓冲乒乓队列
};
```
> **910B 优化关键细节**：
> 1. **Cube-to-UB 带宽约束**：Cube 计算速度极快，若 Token 维度 $M$ 很小（如 Decode 阶段 $M=1$ 或 $10$），Cube 极易饥饿；必须确保 L0C $\to$ UB 的 Fixpipe 通道与 Vector 指令周期匹配，避免 Vector 成为瓶颈。
> 2. **Cross-Core 规约策略**：Hidden 维 $K=2048$ 通常跨越多个 Core 分片，单个 Core 计算完局部平方和后，通过 910B 的片上 L2 缓存原子加（Atomic Add）或硬件屏障同步更新归一化分母，最后做全局标量缩放。

---

### 模式 2：双缓冲深度流水化（Double Buffering / Multi-Stage Ping-Pong）

#### 1. GPU 上的经验本质
在 CUDA/CUTLASS 3.x 中，通过 `cute::TMA` 配合异步共享内存多阶段管线（`Stage=3` 或 `Stage=4`），利用 `cuda::barrier` 实现内存加载与 Tensor Core 计算完全隐藏。

#### 2. 910B 上的体系结构机会与瓶颈
910B 的硬件底层完全依赖 **队列（Queue）驱动的显式流水线**。Ascend C 官方推荐的典型流水为 3 阶段（`CopyIn $\to$ Compute $\to$ CopyOut`），但在高负载大算子（如 Prefix/Suffix Gate/Up）中，3 级单缓冲会导致单元间等待。

#### 3. 910B 进阶优化策略
- **多级深度乒乓（Multi-Buffer Pipeline）**：
  在 UB 和 L1 Buffer 空间允许的前提下，将关键深度设为 **3 缓冲或 4 缓冲**：
  ```cpp
  // 传统推荐 2 缓冲，进阶优化应显式根据 SRAM 预算申请 3~4 缓冲
  TQue<QuePosition::A1, 3> qA1; // L1 缓冲队列深度设为 3
  TQue<QuePosition::B1, 3> qB1;
  TQue<QuePosition::UB, 4> qUB; // Vector 处理申请 4 深度
  ```
- **拆分 Compute 阶段解除串行锁定**：
  在 910B 上，严禁将一个大分块的计算与多次连续小搬运混杂在一个 `Compute` 函数里。必须拆分为纯粹的事件级细粒度推进：
  $$\text{Event } i: \text{MTE2 (搬第 } i+2 \text{ 块)} \parallel \text{Cube (算第 } i+1 \text{ 块)} \parallel \text{Vector (消第 } i \text{ 块)}$$
  在 Ascend C 中，通过分离的 API 调用与非阻塞查询确保 **MTE 引擎、Cube 单元、Vector 单元三者利用率同时达到 90%+**。

---

### 模式 3：小 M 维度（Batch/Decode 场景）下的分块与填充规避策略

#### 1. GPU 上的经验本质
在 S-B 阶段的 Suffix 优化中，我们发现当 $M=10, K=1024, N=4096$ 时，若直接套用大模型通用的 $Tile_N=64$ 或 $128$ 核函数，会导致 **84% 的计算资源因 Padding 或尾部 Wave 空转而浪费**。我们通过定制 $Tile_N=16$ 的小 M 专用核函数，结合 Cluster 2x1 调度，将耗时逼近 DRAM 物理 SOL 下限的 99.5%。

#### 2. 910B 上的体系结构瓶颈
- 910B Cube 单元的硬件原生最小矩阵乘乘法原子形状为 **$16\times 16\times 16$（FP16）** 或 **$16\times 16\times 32$**。
- 当大模型在 Action Decode 或逐 Token 生成阶段，输入 Token 数 $M < 16$（例如 Pi05 Suffix 的 $M=10$ 或单步 Decode $M=1$）：
  - 传统算子如果机械地将 $M$ 用 0 填充（Pad）到 16，Cube 内部至少有 $37.5\% \sim 93.8\%$ 的 MAC 单元在做无效零乘法，且浪费搬运带宽。

#### 3. 910B 推测性优化方案：Batch 跨时间步聚合 / 维度转置策略
1. **多 Step / 多请求微批拼接（Virtual M-Packing）**：
   在 910B 算子入口处，避免针对单个 Token 调用一次 Cube Kernel；应在片上内存中将相邻去噪步或多个并发请求的特征向量在 $M$ 维紧密拼接，使实际送入 Cube 的有效 $M_{\text{packed}} \ge 16$ 且为 16 的整倍数。
2. **转置利用 GEMV 向量化替换（Vector-Fallback for Extremely Small M）**：
   当 $M=1$ 或 $M \le 4$ 时，Cube 的调度启动开销和 Padding 损失极大。在 910B 上应设计动态分支（Dynamic Dispatch）：
   - 若 $M \ge 16$：走 Cube Core GEMM 分块流水；
   - 若 $M < 8$：**完全放弃 Cube 单元，改用 Vector Core 的 GEMV 向量乘加指令（`MulSum` / `Dot`）**。910B Vector Core 拥有极强的 256-bit SIMD 并行能力，纯 Vector GEMV 在极小 Batch 下访存带宽利用率远高于强行 Pad 进 Cube。

---

### 模式 4：GeGLU / SwiGLU 激活函数的无缝融合与 ULP 精度控制

#### 1. GPU 上的经验本质
在 GPU 上，我们发现独立算子串联时，由于 TensorRT 插入了冗余的 `__myl_Move` 拷贝以及在近零强负值区（GELU 处于 subnormal 悬崖区）的 FP16 舍入次序不同，会导致与原生参考出现数个 ULP 的偏差。

#### 2. 910B 上的体系结构与精度特性
- 910B 的 Vector Core 内部计算默认支持 **FP32 累加与高精度数学函数**。
- `GeGLU(x) = Gelu(Gate(x)) * Up(x)` 涉及两个大矩阵乘输出的元素级交叉乘法。

#### 3. 910B 推测性融合实现与精度指南
- **两路 GEMM 交替出栈与片上交乘**：
  若分别计算 Gate 和 Up 矩阵乘再写回 HBM，中间激活值显存占用高达数十兆。
  推测的最佳流水架构为：
  1. Cube 算出一块 $Tile_{\text{Gate}}$，通过 Fixpipe 送入 UB；
  2. Cube 紧接着算出一块 $Tile_{\text{Up}}$，通过 Fixpipe 送入另一块 UB；
  3. **Vector Core 原地执行 GeGLU**：
     ```cpp
     // 在 UB 内完成高精度激活与融合乘
     // 1. 调用 Vector 高性能 GELU 估算指令 (避免慢速 exp)
     Gelu(ubGate, ubGate, tileElements); 
     // 2. 原地与 Up 分块逐元素相乘，直接产出 FFN 中间结果
     Mul(ubOut, ubGate, ubUp, tileElements); 
     ```
  4. 随后直接作为下一阶段 Down GEMM 的输入，或者通过 MTE3 写回。
- **次正规数与负值截断策略**：
  昇腾架构对 FP16 次正规数（Subnormal Numbers，绝对值 $< 2^{-14} \approx 6.1\times 10^{-5}$）默认可能开启 Flush-to-Zero (FTZ)。若模型中存在强负值（经过 GELU 后输出接近零），必须在 CANN 编译选项中显式检查：
  - 配置 `enable_high_precision = true`；
  - 避免激活函数因 FTZ 硬件直接归零产生阶跃跳变，确保精度满足 Cosine $> 0.99999$ 与 ULP $\le 2$ 的工业级门禁。

---

### 模式 5：基于硬件 Flag 的细粒度跨算子依赖提前放行（对标 GPU PDL / Early-Q）

#### 1. GPU 上的经验本质
在 RoPE $\to$ XQA 实验中，我们攻克了 **Early-Q 放行**：RoPE 不需要等 Key 也写完，只要 Query 矩阵写完，立刻通过 `griddepcontrol.launch_dependents` 放行后继 XQA，使 XQA 能提前在片上做页表寻址和元数据初始化。

#### 2. 910B 上的等价机制：跨核事件硬件同步（Cross-Core Event Synchronization）
910B 内部没有直接叫“PDL”的指令，但具备更强大的**硬件级跨 Core 事件同步机制**（Task Event / Flag 寄存器系统）。

#### 3. 910B 落地建议：前驱算子“分部放行”
在大模型推理中，以 `QKV Projection $\to$ Attention` 为例：
1. **传统做法**：QKV Projection 算完整个张量，向 Runtime 报告 Kernel 结束，Host 驱动下发下一个 Attention Kernel。这会引入数微秒的 Host 调度延迟与 GPU/NPU 空转。
2. **Ascend C 细粒度放行设计**：
   - 在单 Core 内部或多 Core 协同中，将 QKV 的计算次序重排为：**优先计算 Q，再计算 K 和 V**。
   - 当 Q 计算完毕并通过 MTE3 开始写入 L2/HBM 的瞬间，调用硬件指令发送事件标识：
     ```cpp
     // [推测性设计] 利用 Ascend 硬件跨核心 Flag 机制
     // 前驱核函数 (QKV Projection)
     if (isQBlockFinished) {
         // 设置硬件事件标志，通知下游等待的 Core 数据已落盘/准备好
         set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID_Q_READY);
     }
     ```
   - 后续执行 Attention 的 Core 可以提前完成加载 KV 页表元数据、计算 RoPE 旋转角等无需等待 Q 数据的工作，并在真正需要 Q 时执行 `wait_flag(..., EVENT_ID_Q_READY)`，实现近乎零延迟的微观流水搭接。

---

### 模式 6：片上内存（UB/L1）Bank 冲突与数据对齐避坑指南

#### 1. GPU 经验对照
在 CUDA 中，Shared Memory 的 32 个 Bank 若发生跨步访问（Strided Access）会引发多路 Bank Conflict，导致访存吞吐断崖式下跌。

#### 2. 910B 上的对齐硬约束
华为 910B 的片上内存（尤其是 UB 和 L1 Buffer）对数据访问的**物理地址对齐**和**数据块对齐**有极度严苛的硬件约束：
- **32 字节对齐规则（32-Byte Alignment Rule）**：
  MTE 搬运引擎和 Vector 指令的核心物理处理粒度为 **32 字节（即 16 个 FP16 或 8 个 FP32）**。
- **高危陷阱与应对策略**：
  1. **跨步搬运（Strided Copy）性能骤降**：若在 HBM $\to$ UB 搬运非连续数据（如截取注意力多头的某一个 Head），若单次连续段长度不足 32 字节或起始地址未对齐 32 字节，MTE 引擎会退化为低效的逐字节拼接模式，搬运效率成倍衰减。
     - *对策*：必须使用 `DataCopyPad` 或重构内存排布为 `[Batch, Head, Seq, Dim]`（连续排布），严禁在内层循环做小跨度非对齐搬运。
  2. **UB 空间局部重叠（Memory Aliasing）**：
     在同一个 Vector Compute 阶段，若源操作数 Tensor 与目的操作数 Tensor 在 UB 中存在未对齐的部分重叠，会直接触发硬件未定义行为或静默计算错误。必须使用 Ascend C 的 `TBuf` 严格隔离临时变量空间。

---

## 三、工程落地与调优行动清单（Actionable Checklist）

当在 Ascend 910B 上开发或优化大模型关键链路（如 Prefix/Suffix GEMM、Attention、Down-Norm 融合）时，建议按以下五步法排查推进：

1. **【第一步：算子边界与融合形式审计】**
   - 检查 CANN Profiling 产生的 Timeline（利用 `msprof` 工具导出）。
   - 确认是否存在 `Matmul $\to$ Transpose/Cast $\to$ Mul/Add $\to$ Norm` 这种碎片化小算子序列；若有，优先合并为单一 Ascend C 自定义核函数。
2. **【第二步：Cube/Vector 算力平衡分析】**
   - 检查当前算子是处于 **Cube Bound**、**Vector Bound** 还是 **MTE (Memory) Bound**。
   - 若 MTE 占用率极高而 Cube 处于停顿（Stall），立刻引入 3~4 级多缓冲（TQue 深度增加）或扩大分块大小以提高数据复用比。
3. **【第三步：小 M 维度专属优化分支】**
   - 检查 Decode 阶段 $M$ 的实际取值。若 $M < 16$，坚决避免简单零填充进 Cube，应实测评估 **Vector GEMV 分支** 与 **跨 Step Packing 分支** 的延迟，选择最优派发路径。
4. **【第四步：显式内存闭环（消除 HBM 往返）】**
   - 对于 Down $\to$ Residual $\to$ Norm 链路，验证中间结果是否能全程保留在 L0C/UB 内；坚决消灭“算完写回 HBM、下一算子再读入”的无谓带宽消耗。
5. **【第五步：严守 32 字节对齐与数值精度门禁】**
   - 确保所有片上缓冲分配与 MTE 传输首地址满足 32 字节物理对齐。
   - 针对次正规数敏感区域（如 GELU 强负值截断），验证是否触发了 FTZ，采用复合门禁（`Cosine > 0.99999`，`max_abs <= 1e-5`）收敛精度。

---

*文档生成归档路径：`wty-11-temp/ASCEND_910B_ADVANCED_OPERATOR_OPTIMIZATION_GUIDE.md`*
