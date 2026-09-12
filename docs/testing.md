# 本地测试指南

## 测试结构

维护中的测试入口是 `examples/test_aclnn_chunk_scaled_dot_kkt.cpp`。它通过 aclnn 调用自定义算子，并在主机端生成 CPU golden。默认精度标准为：

```text
abs(actual - expected) <= 1e-6
或
relative_error <= 1e-5
```

小 case 做完整输出比对；大 case 在计时区间之外做采样比对。输出的 padding 区也会检查为 0。

## 构建与列出用例

先安装算子，再执行：

```bash
bash examples/run.sh --list
# 或已构建后：
bash scripts/run_kkt.sh list
```

## 正确性回归

```bash
bash scripts/run_kkt.sh correctness
```

11 个 case 覆盖：

- K=128 与 K=256
- `numRepeat=1` 与多个 query heads 复用同一 KV head
- full64、tail63、tail33、tail32、单 token
- 单调、非单调、全相等、接近零的 `g_i-g_j`
- row0 active 场景，防止误加位置下三角
- 多序列 full/tail 混合
- `H=128,Hg=1` 高复用与 `Hg=32` 大 KV-head 数

单独定位 case：

```bash
bash scripts/run_kkt.sh case tail33_k256
bash scripts/run_kkt.sh case many_heads_high_reuse_small_k128
```

## 性能回归

```bash
bash scripts/run_kkt.sh perf
bash scripts/run_kkt.sh perf --warmup 10 --repeat 50 --samples 24
```

6 个性能 case 包含 K128/K256、单/多序列、大 T、高 head 复用与多 KV heads。默认每个 case 自带 warmup/repeat 值；命令行参数可统一覆盖。

计时使用设备事件；wall time 仅辅助观察。比较两个版本时必须使用同一机器、同一 CANN/频率状态、相同输入、warmup、repeat 和校验方式，并报告完整 case 集合，不能拼接每个 case 的局部最佳值。

## 压力测试

```bash
bash scripts/run_kkt.sh stress
```

当前 stress case 使用 `T=262144,H=8,Hg=2,K=128` 和四段序列。它可能占用较多设备与主机内存，应在正确性套件通过后单独运行。

## 原始二进制 CLI

```bash
examples/build/bin/test_aclnn_chunk_scaled_dot_kkt --help
examples/build/bin/test_aclnn_chunk_scaled_dot_kkt --suite correctness
examples/build/bin/test_aclnn_chunk_scaled_dot_kkt --case perf_official_like_k128
examples/build/bin/test_aclnn_chunk_scaled_dot_kkt --case trace_ratio3_full64_k128 \
  --warmup 0 --repeat 1 --no-check
```

可用参数：`--suite`、`--case`、`--device`、`--warmup`、`--repeat`、`--samples`、`--no-check`、`--list`。

## 结果与停止条件

- 正确性：进程返回 0，所有选中 case 输出 `[ OK ]`，`mismatch=0`、`paddingMismatch=0`、`nonFinite=0`。
- 性能：至少先通过对应输入的校验；随后才可用 `--no-check` 采集。
- 任意 hang、runtime error、非有限值或 padding 错误都阻止版本晋级。
- 单次变快或 simulator span 变短都不足以宣布总体优化；必须对完整目标 case 复测。

## 历史 UT 的状态

原工作区中的 `tests/ut` 是早期框架 UT 模板，其中存在 TODO，且使用了旧 tiling 字段；它与当前发布基线的 tiling ABI 不兼容，因此没有纳入发布包，也没有接入 `build.sh`。不要把它的编译/运行结果作为当前实现的验收证据。
