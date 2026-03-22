#!/usr/bin/env bash
# run_single.sh — manually (re)run a single experiment by name.
#
# Usage: bash runner/run_single.sh <run_name> [--config configs/experiments.yaml]

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <run_name> [--config <path>]" >&2
  exit 1
fi

RUN_NAME="$1"
CONFIG="${2:-configs/experiments.yaml}"

mkdir -p "results/${RUN_NAME}"

echo "Starting run: ${RUN_NAME}"
echo "Config:       ${CONFIG}"
echo "Output:       results/${RUN_NAME}/console.log"
echo ""

python runner/experiment_wrapper.py --config "${CONFIG}" --run-name "${RUN_NAME}" \
  2>&1 | tee "results/${RUN_NAME}/console.log"
