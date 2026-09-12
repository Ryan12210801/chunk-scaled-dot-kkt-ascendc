#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_DIR}"
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum --check versions/SOURCE_SNAPSHOTS.sha256
elif command -v shasum >/dev/null 2>&1; then
    shasum --algorithm 256 --check versions/SOURCE_SNAPSHOTS.sha256
else
    echo "[ERROR] Neither sha256sum nor shasum is available" >&2
    exit 1
fi
