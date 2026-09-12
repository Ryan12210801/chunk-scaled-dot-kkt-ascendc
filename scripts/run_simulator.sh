

#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

KKT_BIN="${KKT_BIN:-${REPO_DIR}/examples/build/bin/test_aclnn_chunk_scaled_dot_kkt}"
ASCEND_ENV="${ASCEND_ENV:-${ASCEND_HOME_PATH:-/home/developer/Ascend/cann-9.0.0}/set_env.sh}"
SIM_CASE="${1:-trace_ratio3_full64_k128}"
SIM_SOC="${SIM_SOC:-Ascend910B3}"
SIM_CORE_ID="${SIM_CORE_ID:-0}"
SIM_DIR="${SIM_DIR:-${REPO_DIR}/results/simulator_runs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${SIM_DIR}/${SIM_CASE}_${TIMESTAMP}"

if [[ -f "${ASCEND_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${ASCEND_ENV}"
fi

command -v msprof >/dev/null 2>&1 || {
    echo "[ERROR] msprof was not found in PATH" >&2
    exit 1
}

[[ -x "${KKT_BIN}" ]] || {
    echo "[ERROR] Test binary is missing or not executable: ${KKT_BIN}" >&2
    echo "[INFO] Build it first with: bash examples/run.sh --list" >&2
    exit 1
}

mkdir -p "${OUTPUT_DIR}"

echo "[INFO] Simulator case: ${SIM_CASE}"
echo "[INFO] SoC model: ${SIM_SOC}; core: ${SIM_CORE_ID}"
echo "[INFO] Output: ${OUTPUT_DIR}"

msprof op simulator \
    --soc-version="${SIM_SOC}" \
    --launch-count=1 \
    --core-id="${SIM_CORE_ID}" \
    --output="${OUTPUT_DIR}" \
    "${KKT_BIN}" \
    --case "${SIM_CASE}" \
    --warmup 0 \
    --repeat 1 \
    --no-check
