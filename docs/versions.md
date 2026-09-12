# 关键源码版本

本仓库没有把所有过程文件原样上传，而是选取四个可解释的历史里程碑；根目录是第五个、也是作者指定的当前发布基线。每个历史目录都成套保留匹配的 `op_host/` 与 `op_kernel/`，避免 Host tiling 和 kernel ABI 混用。

| 版本 | 位置 | 主线变化 | 状态边界 |
| --- | --- | --- | --- |
| v01 Cube-first baseline | `versions/v01_cube_first_baseline` | 早期 Cube 主导实现，通用 Matmul 与 VECOUT 中转 | 用于展示起点，不代表最后性能基线 |
| v02 MIX pipeline V6.1 | `versions/v02_mix_pipeline_v6_1` | AIC 生产一次 Gram，两个 AIV 消费；双 slot READY/FREE 流水 | 代表架构转折点 |
| v03 direct-MMAD stable | `versions/v03_direct_mmad_stable` | K128 L1 ping-pong、K256 保守 direct MMAD、输出窗口与跨 task 预取的稳定组合 | 原始 manifest 一并保留；历史整包状态，不等于当前最优 |
| v04 Phase-C/chunk-pair | `versions/v04_phase_c_chunk_pair` | 后期 AIV Phase C、chunk pair、指数因子化与 direct MMAD 组合 | 任务连续分配策略引入前的直接前身 |
| v05 release baseline | 仓库根目录 | 在 v04 上加入参考 2026 年 7 月冠军的活动 group 选择与连续任务分配 | 作者最终版本，已通过赛事审核 |

## 源码指纹

| 版本 | kernel header SHA-256 | Host tiling SHA-256 |
| --- | --- | --- |
| v01 | `112efbb62f83d891b71183cdc520eb58ecf489232028ea0a0bbda149741068b7` | `02cefb94a48e79c0ed0348bf9259bda1b8442898afd5728747a1baecbc95c0f3` |
| v02 | `4f2b0b69d4e3f4accd269f40e449d3d70cf8bd10123db05c48f9700615ddb618` | `b8b55a6d2c76693ba0b0e83edfc87d3f08d05aa00c2eb5c6195bbe4605013d53` |
| v03 | `b07ab363a501b8868d1d51169b3080521bb17ac496bedb76b727ade325665ac6` | `b8b55a6d2c76693ba0b0e83edfc87d3f08d05aa00c2eb5c6195bbe4605013d53` |
| v04 | `7526f63cb3fa3fbb07b084d1ac437e7becf9aa9fca59ba01b0ed31acd8306e09` | `1e9d11e0b4dd2d05a0548f32ec3775caa483dd06c638e5f16246efb6b8bff559` |
| v05 | `7d416f68bdf9b329b7ce331c421f8f5f1c839f777b6995d84455aa205b18d03f` | `eb3df2aa7df231517394e71bd8fe808e04ea87b91f0530163ec178302d4dc748` |

这些哈希对应整理时的核心源码快照。之后若修复根目录代码，v05 哈希自然会变化；历史版本目录原则上保持只读。

## 为什么没有纳入其他目录

- 大量 `build/`、dump、数据库和数百 MB trace 属于可再生过程文件，不适合源码仓库。
- 只有单文件、缺少匹配 Host 侧或状态不明的候选不作为里程碑。
- `code3_all_optimizations` 是独立实验分支，缺少编译/上板/Judge 验证，不能冒充发布版本。
- 历史局部最优不能拼成一个未经验证的“全优化版”。
