# 开源发布审计

## 已完成

- 从 `best_chunk_scaled_dot_kkt_real` 提取当前发布基线，未修改原始比赛目录。
- 选择四个成套历史里程碑，加上根目录最终版共五阶段。
- 将增强版 aclnn/CPU-golden runner 作为维护中的本地测试入口。
- 修复测试、msprof 与 simulator 脚本中的比赛机绝对路径和输出目录问题。
- 将构建声明收紧到 Ascend 910B，避免对未验证 SoC 作兼容性承诺。
- 移出过时 UT 模板，并清楚标记它与当前 tiling ABI 不兼容。
- 仅保留小型 CSV、日志和 trace 汇总；排除 build、dump、数据库和巨型 trace。
- 补齐环境、测试、profiling、版本、优化机制与 provenance 文档。
- 已按仓库所有者确认加入 MIT License。
- 当前发布基线已通过赛事审核。
- 已明确记录：只参考 2026 年 7 月冠军实现的任务连续分配策略，其余核心实现为独立开发。

## 后续维护建议

1. **环境变更**：更换 CANN 或 SoC 后，应重新运行正确性和性能套件。
2. **性能数据口径**：历史 profile 用于展示分析过程；新增最终版成绩时，应记录源码哈希、设备状态、warmup/repeat 和完整 case 结果。
3. **任务调度改动**：修改活动 group、`blockDim` 或连续分配阈值后，应重新做完整回归。

## 建议发布门禁

```bash
bash build.sh -j8
bash examples/run.sh --list
bash scripts/run_kkt.sh correctness
bash scripts/run_kkt.sh perf --warmup 10 --repeat 50
bash scripts/run_kkt.sh prof perf_official_like_k128
```

当前版本已通过审核。后续发布新的性能结果时，应确保结果属于同一完整源码，不拼接不同版本的局部最优值。

## 未纳入仓库

- 原始 build/install 输出
- dump、SQLite/db、device 二进制
- 41–422 MB 的原始 simulator trace
- 状态不明或缺少匹配 Host/kernel 的散落候选
- 未经编译/上板/Judge 验证的 `code3_all_optimizations` 实验
