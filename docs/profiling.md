# Profiling 与 simulator 指南

## 上板 msprof

先确认正确性 case 通过，然后对单个 case 做最小、可重复采集：

```bash
bash scripts/run_kkt.sh prof perf_official_like_k128
```

默认 wrapper 使用：

```text
msprof op --warm-up=10 --launch-count=1 --output=<repo>/results/profile_runs/...
```

测试二进制收到 `--warmup 1 --repeat 1 --no-check`。msprof 的 warm-up/launch-count 与测试程序自己的 warmup/repeat 是两层控制，可用环境变量或 `prof-run` 显式调整：

```bash
MSPROF_WARMUP=5 MSPROF_LAUNCH_COUNT=1 \
  bash scripts/run_kkt.sh prof-run \
  --case perf_many_kv_h32_hg32_k256 --warmup 0 --repeat 1 --no-check
```

重点读取：

- `OpBasicInfo.csv`: `Task Duration`, `Block Dim`, `Mix Block Dim`
- `PipeUtilization.csv`: 各 pipeline 活跃度
- `Memory*.csv`, `L2Cache.csv`: 搬运与 cache 现象
- `ArithmeticUtilization.csv`: Cube/Vector 算术利用
- `ResourceConflictRatio.csv`: 资源冲突信号

不要只依据一列 utilization 下结论。先定位端到端 kernel span，再结合关键引擎的 busy 区间、等待与搬运解释变化。

## 历史硬件样本

`results/device_profiles/` 只保留三次历史采集的日志与小型 CSV，去除了 dump、数据库和大体积原始文件：

| case | 测试程序 device avg | msprof Task Duration | BlockDim |
| --- | ---: | ---: | ---: |
| `perf_official_like_k128` | 日志未保留 | 575.780 µs | 20 |
| `perf_high_reuse_h128_hg1` | 4908.460 µs | 4217.640 µs | 20 |
| `perf_many_kv_h32_hg32_k256` | 2846.000 µs | 2256.840 µs | 20 |

这些采集发生于 2026-07-23，来自优化过程中的早期源码。测试程序计时和 msprof 的单次 task duration 口径不同，不能相互替换，也不能据此声称最终版达到这些数字。

## simulator 采集

```bash
bash scripts/run_simulator.sh trace_ratio3_full64_k128
```

默认使用 `Ascend910B3`、core 0、launch-count 1，输出到 `results/simulator_runs/`。可覆盖：

```bash
SIM_SOC=Ascend910B3 SIM_CORE_ID=0 SIM_DIR=/data/sim \
  bash scripts/run_simulator.sh trace_ratio4_full64_k128
```

## 历史 trace 的离线分析

如果有一个已有的 Chrome trace JSON，可使用保留的分析脚本；先看脚本帮助以确认参数：

```bash
python3 tools/trace/analyze_ascend_trace_multicore.py --help
python3 tools/trace/analyze_trace_overlap.py --help
```

`results/simulator/` 保留两组已生成汇总。它们的全局 span 分别约 39.640 µs 与 40.520 µs，均观察到 20 个 AIC、40 个 AIV 和 314 个 MMAD。它们是历史 trace 的分析结果，不绑定当前最终源码。

## 解释规则

- 多核与多 pipeline 并行，按指令聚合的 `sum µs` 可以远大于 wall-clock span。
- CSV 中 BAR/WAIT 的累计时长不是可以直接删除的延迟；需要确认依赖和资源生命周期。
- AIV 的 cross-core wait 可能表示生产—消费关系，而非单纯浪费。
- 比较优化前后时必须保持 case、SoC、CANN、采集参数一致，并保留原始命令和源码哈希。
- simulator 用于解释机制；是否晋级仍以目标板正确性和完整性能 case 为准。
