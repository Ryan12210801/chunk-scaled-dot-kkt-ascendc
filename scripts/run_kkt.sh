#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# chunk_scaled_dot_kkt unified runner
#
# Examples:
#   bash scripts/run_kkt.sh list
#   bash scripts/run_kkt.sh correctness
#   bash scripts/run_kkt.sh perf
#   bash scripts/run_kkt.sh case perf_official_like_k128
#   bash scripts/run_kkt.sh prof perf_official_like_k128
#   bash scripts/run_kkt.sh run --suite perf --warmup 5 --repeat 20
#
# Environment overrides:
#   KKT_BIN=/path/to/test_binary bash scripts/run_kkt.sh perf
#   ASCEND_ENV=/path/to/set_env.sh bash scripts/run_kkt.sh perf
#   LOG_DIR=./my_logs bash scripts/run_kkt.sh perf
#   PROF_DIR=./my_prof bash scripts/run_kkt.sh prof perf_official_like_k128
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

KKT_BIN="${KKT_BIN:-${REPO_DIR}/examples/build/bin/test_aclnn_chunk_scaled_dot_kkt}"
ASCEND_ENV="${ASCEND_ENV:-${ASCEND_HOME_PATH:-/home/developer/Ascend/cann-9.0.0}/set_env.sh}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/results/test_runs}"
PROF_DIR="${PROF_DIR:-${REPO_DIR}/results/profile_runs}"
MSPROF_WARMUP="${MSPROF_WARMUP:-10}"
MSPROF_LAUNCH_COUNT="${MSPROF_LAUNCH_COUNT:-1}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

usage() {
    cat <<'USAGE'
Usage:
  bash scripts/run_kkt.sh <mode> [arguments]

Modes:
  list
      List all available test cases.

  correctness [extra test arguments]
      Run the correctness suite.

  perf [extra test arguments]
      Run the performance suite.

  stress [extra test arguments]
      Run the stress suite.

  case <case_name> [extra test arguments]
      Run one selected case.

  run [test arguments]
      Pass all arguments directly to the test binary.

  prof <case_name> [extra test arguments]
      Profile one case with msprof.
      Default test arguments: --warmup 1 --repeat 1 --no-check

  prof-run [test arguments]
      Run msprof and pass all supplied arguments directly to the test binary.

  help
      Show this message.

Examples:
  bash scripts/run_kkt.sh correctness
  bash scripts/run_kkt.sh perf --warmup 10 --repeat 50
  bash scripts/run_kkt.sh case perf_official_like_k128
  bash scripts/run_kkt.sh prof perf_high_reuse_h128_hg1
  bash scripts/run_kkt.sh run --suite perf --no-check

Environment variables:
  KKT_BIN      Test executable path
  ASCEND_ENV   Ascend set_env.sh path
  LOG_DIR      Normal test log directory
  PROF_DIR     Profiling output directory
  MSPROF_WARMUP       msprof operator warm-up count (default: 10)
  MSPROF_LAUNCH_COUNT msprof captured launches (default: 1)
USAGE
}

error() {
    echo "[ERROR] $*" >&2
    exit 1
}

info() {
    echo "[INFO] $*"
}

load_ascend_env() {
    if [[ -f "${ASCEND_ENV}" ]]; then
        # shellcheck disable=SC1090
        source "${ASCEND_ENV}"
    else
        info "Ascend environment script not found: ${ASCEND_ENV}"
        info "Continuing with the current shell environment."
    fi
}

check_binary() {
    [[ -f "${KKT_BIN}" ]] || error "Test binary not found: ${KKT_BIN}"
    [[ -x "${KKT_BIN}" ]] || error "Test binary is not executable: ${KKT_BIN}"
}

run_test() {
    local log_name="$1"
    shift

    mkdir -p "${LOG_DIR}"
    local log_file="${LOG_DIR}/${log_name}_${TIMESTAMP}.log"

    info "Binary: ${KKT_BIN}"
    info "Command: ${KKT_BIN} $*"
    info "Log: ${log_file}"
    echo

    "${KKT_BIN}" "$@" 2>&1 | tee "${log_file}"

    echo
    info "Finished. Log saved to: ${log_file}"
}

run_profile() {
    local profile_name="$1"
    shift

    command -v msprof >/dev/null 2>&1 || error "msprof was not found in PATH"

    mkdir -p "${PROF_DIR}"
    local output_dir="${PROF_DIR}/${profile_name}_${TIMESTAMP}"
    local log_file="${output_dir}.log"

    info "Binary: ${KKT_BIN}"
    info "Profile output: ${output_dir}"
    info "Command: msprof op --warm-up=${MSPROF_WARMUP} --launch-count=${MSPROF_LAUNCH_COUNT} --output=${output_dir} ${KKT_BIN} $*"
    echo

    msprof op \
        --warm-up="${MSPROF_WARMUP}" \
        --launch-count="${MSPROF_LAUNCH_COUNT}" \
        --output="${output_dir}" \
        "${KKT_BIN}" "$@" 2>&1 | tee "${log_file}"

    echo
    info "Profiling finished."
    info "Profile data: ${output_dir}"
    info "Console log: ${log_file}"

    local summary
    summary="$(find "${output_dir}" -type f \
        \( -name 'op_summary*.csv' \
        -o -name '*PipeUtilization*.csv' \
        -o -name 'task_time*.csv' \) \
        2>/dev/null | head -n 20 || true)"

    if [[ -n "${summary}" ]]; then
        echo
        info "Key profiling files:"
        echo "${summary}"
    fi
}

main() {
    local mode="${1:-help}"
    if [[ $# -gt 0 ]]; then
        shift
    fi

    load_ascend_env

    case "${mode}" in
        help|-h|--help)
            usage
            ;;

        list)
            check_binary
            "${KKT_BIN}" --list
            ;;

        correctness)
            check_binary
            run_test "correctness" --suite correctness "$@"
            ;;

        perf)
            check_binary
            run_test "perf" --suite perf "$@"
            ;;

        stress)
            check_binary
            run_test "stress" --suite stress "$@"
            ;;

        case)
            check_binary
            [[ $# -ge 1 ]] || error "Missing case name. Example: bash scripts/run_kkt.sh case perf_official_like_k128"
            local case_name="$1"
            shift
            run_test "case_${case_name}" --case "${case_name}" "$@"
            ;;

        run)
            check_binary
            run_test "custom" "$@"
            ;;

        prof)
            check_binary
            [[ $# -ge 1 ]] || error "Missing case name. Example: bash scripts/run_kkt.sh prof perf_official_like_k128"
            local case_name="$1"
            shift
            run_profile "${case_name}" \
                --case "${case_name}" \
                --warmup 1 \
                --repeat 1 \
                --no-check \
                "$@"
            ;;

        prof-run)
            check_binary
            run_profile "custom" "$@"
            ;;

        *)
            usage
            echo
            error "Unknown mode: ${mode}"
            ;;
    esac
}

main "$@"
