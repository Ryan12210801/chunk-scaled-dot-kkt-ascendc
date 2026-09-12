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

## 发布前阻塞项

1. **目标机复验**：需要在 Ascend 910B + CANN 9.0.0 上干净构建、安装并运行正确性套件。
2. **最终版性能闭环**：历史 profile 不能替代最终源码复测。至少运行 6 个 perf cases，并记录源码哈希、设备状态、warmup/repeat 和结果。
3. **上游归属确认**：`NOTICE.md` 已标明 champion-derived 调度来源，后续仍应确认是否需要更具体的作者/提交 attribution。

## 建议发布门禁

```bash
bash build.sh -j8
bash examples/run.sh --list
bash scripts/run_kkt.sh correctness
bash scripts/run_kkt.sh perf --warmup 10 --repeat 50
bash scripts/run_kkt.sh prof perf_official_like_k128
```

仓库可以作为带明确验证边界的源码与过程记录公开；只有在正确性全部通过、性能数据属于同一完整源码且归属确认完成后，才应创建性能 release 或宣布最终成绩。

## 未纳入仓库

- 原始 build/install 输出
- dump、SQLite/db、device 二进制
- 41–422 MB 的原始 simulator trace
- 状态不明或缺少匹配 Host/kernel 的散落候选
- 未经编译/上板/Judge 验证的 `code3_all_optimizations` 实验
