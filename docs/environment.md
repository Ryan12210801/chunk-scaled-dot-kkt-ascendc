# 环境与构建

## 已恢复的开发环境

下面的信息来自保留下来的 CMake cache、构建脚本和 simulator 脚本，不是对所有环境的兼容性承诺。

| 项目 | 已恢复值 | 证据/说明 |
| --- | --- | --- |
| OS | Linux 5.10，aarch64 | 历史 `CMakeSystem.cmake` |
| CANN | 9.0.0 | 历史 `ASCEND_CANN_PACKAGE_PATH=/home/developer/Ascend/cann-9.0.0` |
| CMake | 历史生成目录为 3.20.5；工程最低 3.16 | 生成文件与根 `CMakeLists.txt` |
| C++ | `/usr/bin/c++`，C++17 | 历史 cache 与工程配置 |
| 目标 | Ascend 910B / NPU arch 2201 | Host 注册、kernel 编译保护 |
| simulator SoC | `Ascend910B3` | 历史及整理后的 simulator 脚本 |

本次开源整理发生在 macOS，因此只做了脚本、文档、目录和源码静态检查。CANN 编译、算子安装、正确性和性能均应在目标 Linux/NPU 环境重新验证。

## 依赖

- CANN Toolkit 9.0.0，并能找到 `set_env.sh`
- 支持 Ascend 910B 的编译环境与 NPU 驱动
- CMake 3.16+
- C++17 编译器
- 运行测试时需要 `acl`, `nnopbase`, `opapi` 等 CANN runtime 库
- profiling 需要 CANN 提供的 `msprof`

## 构建和安装算子

```bash
cd /path/to/chunk-scaled-dot-kkt
source /home/developer/Ascend/cann-9.0.0/set_env.sh
export ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0
bash build.sh -j8
```

`build.sh` 固定传入 `-DASCEND_COMPUTE_UNIT=ascend910b`，并构建 `all binary package install`。成功时会打印 kernel object 与 `custom_opp_*.run` 包路径。

如 CANN 不在历史路径，通过环境变量覆盖：

```bash
export ASCEND_HOME_PATH=/opt/Ascend/cann
export ASCEND_ENV=/opt/Ascend/cann/set_env.sh
source "$ASCEND_ENV"
bash build.sh
```

## 清理

```bash
bash build.sh --make_clean
```

该命令只清理仓库内的 `build/` 与 `build_out/`。

## 构建测试程序

算子包安装成功后：

```bash
bash examples/run.sh --list
```

这会配置 `examples/build/`，查找安装后的 `aclnn_chunk_scaled_dot_kkt.h` 与 `libcust_opapi.so`，构建二进制并将 `--list` 传给它。

如果二进制位于其他位置：

```bash
KKT_BIN=/absolute/path/to/test_aclnn_chunk_scaled_dot_kkt \
  bash scripts/run_kkt.sh list
```

## 故障定位顺序

1. `source "$ASCEND_ENV"` 后确认 `msprof`、CMake 和 runtime 库在 PATH/LD_LIBRARY_PATH 中。
2. 确认自定义包已安装，并能找到生成的 aclnn 头文件与 `libcust_opapi.so`。
3. 确认设备是 Ascend 910B；其他 SoC 会由工程配置主动拒绝。
4. 先跑 `--list`，再跑一个小正确性 case，最后才跑 perf/stress。
5. 如果更换 Host tiling 或历史 kernel，必须整套替换匹配的 `op_host/` 与 `op_kernel/`，不要混用不同 tiling ABI。
