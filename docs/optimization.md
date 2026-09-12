# 优化思路提炼

这份文档提炼最终实现中的主要优化机制、适用条件与失败经验，不公开完整的逐次实验历程。

其中仅第 8 节的任务连续分配策略参考了 **2026 年 7 月该赛题冠军公开实现**：具体包括活动 MIX group / `blockDim` 选择，以及让同一 KV group 的相邻 chunk 尽量由同一执行组连续处理。其余章节所述架构和实现均由本项目独立开发。

## 1. 先固定数学语义和布局

问题：`scaled` 名称、位置下三角与 `g_i-g_j<0` 容易混淆；tail 的逻辑列数与 64 列工作布局也容易混淆。

机制：以题面公式为唯一语义，明确 `beta` 乘在行 `i`，只按 `g_i-g_j<0` mask；full/tail 都输出 64 列，padding 显式清零。

证据：本地 golden 覆盖非单调 row0、全相等、near-zero、tail 和多序列。

边界：小而单调的数据会遮蔽错误；编译通过不代表语义正确。

## 2. 按 `(chunk, kvHead)` 复用 Gram

问题：同一 KV head 被 `R=H/Hg` 个 query heads 复用，逐 head 计算 Gram 会重复 Cube 工作。

机制：AIC 计算一次 `64×K · K×64`，R 份 head-specific 的 beta/exp/mask 在 AIV 上完成。

适用条件：输入 head 映射确实共享 K，且中间 Gram 的传递成本小于重复计算成本。

收益口径：最多消除 Gram 部分的 R 倍重复，不是让整个算子获得 R 倍加速；R 份输出仍存在。

## 3. AIC:AIV = 1:2 的生产—消费流水

问题：两阶段串行会让 Cube 与 Vector 互相等待。

机制：每个 MIX group 使用两个 workspace slot。AIC 等待 FREE、写入 KKT、发 READY；两个 AIV 各处理 32 行并在完成 KKT 读取后归还 FREE。

关键不变量：

- READY 表示该 slot 的 KKT 可读。
- FREE 表示两个消费者都已完成从该 slot 读取 KKT，不表示整个 epilogue/写回完成。
- 没有实际任务的 slice 也必须遵守协议，避免生产者永久等待。
- Host schedule mode、blockDim 和 kernel 中 group/slice 映射是一套协议。

反例：把 `PipeBarrier<PIPE_V>` 当作跨 AIC/AIV 同步，或在多 task 间复用事件但不闭合生命周期，都会留下死锁/数据竞争风险。

## 4. BRCB、double-head 与成对 head 处理

问题：逐行标量广播和过细粒度操作增加 Scalar 控制与短 Vector API 开销。

机制：用 BRCB/repeat-stride 组织广播；一次准备两个 head 的独立向量工作，让编译器和 Vector pipeline 有更多可交叠指令。

适用条件：两个 head 的数据和输出仍保持独立，只共享调度框架或可安全复用的输入。

边界：double-head 不意味着把两个 head 的数学状态混在一起；移除 barrier 前必须重新证明数据依赖。

## 5. 固定形状 direct MMAD

问题：目标 Cube 形状固定为 `64×64×128/256`，通用 Matmul 框架可能带来额外调度与布局成本。

机制：手写 GM→L1→L0A/L0B→MMAD→L0C→workspace 数据流；K128 使用 L1 ping-pong，K256 使用四个物理 stage 的保守路径。

适用条件：形状与平台固定，能够持续验证 L0 生命周期、事件所有权和编译器生成结果。

失败经验：

- 低阶 API 不会自动更快。
- L0B 被下一个 K-part 提前覆盖，会在第三段/跨 task 才暴露。
- 单 task 正确不代表多 task 事件复用安全。
- 最终稳定原则是让 L0 事件在 task 内闭合，并用 fail-closed Host 路由控制范围。

## 6. 指数因子化与数值边界

在 mask 区域中：

```text
exp(g_i-g_j) = exp(g_i-m) * exp(m-g_j)
```

机制：把二维逐元素 Exp 拆为两个一维因子，再通过广播组合，以减少 Exp 调用规模。

适用条件：必须保持 `g_i-g_j<0` 的 mask 语义，并覆盖极端动态范围、接近零差值和非单调输入。

边界：数学等价不等于有限精度下的溢出/下溢行为完全等价；它需要专门的数值回归。

## 7. 延迟输出与 Phase C

问题：KKT 已消费后立即 MTE3 写回，可能与下一任务的 MTE2/Vector 工作争用。

机制：把“可归还 KKT slot”和“何时回收/写回输出 buffer”解耦；保留有限输出窗口，在后续任务更合适的阶段发射 MTE3。

实验证据：历史记录中 Phase C 优于更早的 Phase B 发射；更激进的 KKT double-buffer/next-task prefetch 略有退化。

边界：更早写回、更多 buffer、更多 overlap 都不是单调收益。必须对任务规模与 MTE 争用实测。

## 8. Chunk-pair 与工作负载调度

问题：小/中/大 task 数的固定 blockDim 和循环分配会产生不同的启动成本、负载均衡与局部性。

机制：

- chunk-pair 沿时间方向聚合相邻 chunk，减少部分循环/状态开销。
- R=3 等后期路径尝试 chunk-local 连续任务。
- 当前版本对特定 `Hg`、K、R、task count 范围采用 8/16 core 或 group-aligned 调度，其余回退到通用 cyclic scheduler。

来源边界：本节的任务连续性思路参考了 2026 年 7 月冠军公开实现；没有复制其完整 kernel。冠军实现的阈值依赖其自身结构，因此本项目只在适合自身 direct-MMAD/AIV 流水的范围内采用，并对 `taskCount>400` 保留原调度回退。当前组合是已通过审核的最终版本；若修改这些阈值，仍需重新进行完整 case 回归。

## 9. 没有进入主线的路线

- Beta-on-AIC：数学上可写成行缩放矩阵乘法，但高阶 API 路线历史上仅 12/15，并使多数普通 case 慢约 6–7 倍，暂停。
- 更激进 KKT 预取：局部增加 overlap，但出现轻微退化，未保留。
- 只看一条 trace 的局部收益：不等于 Judge 或完整 case 集合收益。
- 把多个局部候选拼成“all optimizations”：若没有同一源码的编译、正确性和性能闭环，就不是可发布基线。

## 10. 推荐的优化验证闭环

1. 固定基线源码哈希、环境、case 和计时参数。
2. 每次只引入一个可证伪机制。
3. 先跑语义/边界正确性，再跑对应热点 case。
4. 用 msprof/simulator 解释引擎 span、busy union、wait 和数据搬运。
5. 对完整目标 case 统计回退数量，不拼接局部最优。
6. 只有同一完整源码通过正确性与性能门槛，才晋级为新基线。
