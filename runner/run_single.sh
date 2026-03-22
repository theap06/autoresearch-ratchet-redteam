#!/usr/bin/env bash
# run_single.sh — manually (re)run a single experiment by name.
# Starts Ollama if it isn't already running.
#
# Usage: bash runner/run_single.sh <run_name> [configs/experiments.yaml]

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <run_name> [<config>]" >&2
  exit 1
fi

RUN_NAME="$1"
CONFIG="${2:-configs/experiments.yaml}"

# Read Ollama settings from config
OLLAMA_HOST=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c.get('ollama_host','http://localhost:11434'))")
OLLAMA_GPU=$(python3  -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c.get('ollama_gpu', 0))")
OLLAMA_MODEL=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c.get('ollama_model','qwen2.5-coder:72b'))")

# Start Ollama if not already running
if ! curl -sf "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
  echo "Starting Ollama on GPU ${OLLAMA_GPU} …"
  CUDA_VISIBLE_DEVICES="${OLLAMA_GPU}" OLLAMA_HOST="${OLLAMA_HOST}" ollama serve &
  OLLAMA_PID=$!
  for i in $(seq 1 12); do
    sleep 5
    curl -sf "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1 && break
    [ "$i" -eq 12 ] && echo "ERROR: Ollama failed to start." >&2 && exit 1
  done
  echo "Ollama ready."
else
  echo "Ollama already running at ${OLLAMA_HOST}."
fi

# Pull model if needed
if ! OLLAMA_HOST="${OLLAMA_HOST}" ollama list 2>/dev/null | grep -q "${OLLAMA_MODEL%%:*}"; then
  echo "Pulling ${OLLAMA_MODEL} …"
  CUDA_VISIBLE_DEVICES="${OLLAMA_GPU}" OLLAMA_HOST="${OLLAMA_HOST}" ollama pull "${OLLAMA_MODEL}"
fi

mkdir -p "results/${RUN_NAME}"
echo ""
echo "Run:    ${RUN_NAME}"
echo "Config: ${CONFIG}"
echo "Model:  ${OLLAMA_MODEL}"
echo "Output: results/${RUN_NAME}/console.log"
echo ""

python runner/experiment_wrapper.py --config "${CONFIG}" --run-name "${RUN_NAME}" \
  2>&1 | tee "results/${RUN_NAME}/console.log"
