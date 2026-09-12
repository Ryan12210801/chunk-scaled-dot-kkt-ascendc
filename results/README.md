# Curated historical results

本目录只保留可审阅的小型历史证据：

- `device_profiles/`: 三次 2026-07-23 上板 `msprof op` 的控制台日志与 CSV 指标。
- `simulator/`: 两组历史 trace 已生成的多核/流水汇总，不包含数百 MB 原始 trace。

这些结果来自优化过程中的历史源码，不是根目录最终版的复测成绩。具体口径与解释规则见 `../docs/profiling.md`。

新的运行结果默认写到被 `.gitignore` 排除的 `test_runs/`、`profile_runs/` 和 `simulator_runs/`，避免误提交大体积或机器相关文件。
