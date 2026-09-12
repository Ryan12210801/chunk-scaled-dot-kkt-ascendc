# Historical source milestones

这些目录是从原始比赛工作区中挑选的成套源码快照，不是可独立构建的完整仓库。若要复现某一版，请复制该目录中的 `op_host/` 与 `op_kernel/` 覆盖到一个干净的工程副本中，再重新构建；不要只替换 kernel header。

版本含义、状态与 SHA-256 见 `../docs/versions.md`。根目录当前源码是 v05 release baseline。

在 Linux 上可检查核心快照是否被意外改动：

```bash
bash scripts/check_source_snapshots.sh
```
