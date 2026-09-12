#!/usr/bin/env bash
# chunk_scaled_dot_kkt 算子调用示例执行脚本

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "========================================"
echo "chunk_scaled_dot_kkt 算子调用示例"
echo "========================================"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    export ASCEND_HOME_PATH=/usr/local/Ascend/cann
fi

export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake ..
cmake --build . -- -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"

echo "执行调用示例..."
"${BUILD_DIR}/bin/test_aclnn_chunk_scaled_dot_kkt" "$@"

echo "========================================"
echo "执行完成"
echo "========================================"
