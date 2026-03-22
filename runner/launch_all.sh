#!/usr/bin/env bash
# launch_all.sh — start all 8 experimental runs in parallel tmux windows.
# Starts a shared Ollama server first, then launches each run in its own window.
#
# Usage: bash runner/launch_all.sh [configs/experiments.yaml]

set -euo pipefail

CONFIG="${1:-configs/experiments.yaml}"
SESSION="autoresearch"

# Ensure dependencies are available
for cmd in tmux ollama python3; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "ERROR: '$cmd' is not installed or not on PATH." >&2
    exit 1
  fi
done

# Warn if Anthropic key is missing (required for llm_* monitored runs)
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "WARNING: ANTHROPIC_API_KEY is not set. LLM-monitored runs will fail."
  echo "  Set it with: export ANTHROPIC_API_KEY=sk-ant-..."
  echo "  Continuing in 5 seconds — Ctrl-C to abort."
  sleep 5
else
  echo "ANTHROPIC_API_KEY is set (${#ANTHROPIC_API_KEY} chars)."
fi

# Parse config via Python
read -r OLLAMA_MODEL OLLAMA_GPU OLLAMA_HOST < <(python3 - <<'EOF'
import yaml
with open("configs/experiments.yaml") as f:
    cfg = yaml.safe_load(f)
print(
    cfg.get("ollama_model", "qwen2.5-coder:72b"),
    cfg.get("ollama_gpu", 7),
    cfg.get("ollama_host", "http://localhost:11434"),
)
EOF
)

RUNS=$(python3 - <<'EOF'
import yaml
with open("configs/experiments.yaml") as f:
    cfg = yaml.safe_load(f)
for r in cfg["runs"]:
    print(r["name"], r["gpu"])
EOF
)

# ---------------------------------------------------------------------------
# 1. Start the Ollama server (if not already running)
# ---------------------------------------------------------------------------

OLLAMA_PORT="${OLLAMA_HOST##*:}"
OLLAMA_PORT="${OLLAMA_PORT%%/*}"

if curl -sf "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
  echo "Ollama server already running at ${OLLAMA_HOST}."
else
  echo "Starting Ollama server on GPU ${OLLAMA_GPU} with model ${OLLAMA_MODEL} …"
  tmux kill-session -t "${SESSION}_ollama" 2>/dev/null || true
  tmux new-session -d -s "${SESSION}_ollama" -n "ollama" \
    "CUDA_VISIBLE_DEVICES=${OLLAMA_GPU} OLLAMA_HOST=${OLLAMA_HOST} ollama serve 2>&1 | tee results/ollama.log"

  # Wait for the server to become ready (up to 120s)
  echo -n "Waiting for Ollama to start"
  for i in $(seq 1 24); do
    sleep 5
    if curl -sf "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
      echo " ready."
      break
    fi
    echo -n "."
    if [ "$i" -eq 24 ]; then
      echo ""
      echo "ERROR: Ollama server did not start within 120 seconds." >&2
      exit 1
    fi
  done
fi

# ---------------------------------------------------------------------------
# 2. Pull the model if not already present
# ---------------------------------------------------------------------------

if CUDA_VISIBLE_DEVICES="${OLLAMA_GPU}" OLLAMA_HOST="${OLLAMA_HOST}" \
   ollama list 2>/dev/null | grep -q "${OLLAMA_MODEL%%:*}"; then
  echo "Model ${OLLAMA_MODEL} already available."
else
  echo "Pulling model ${OLLAMA_MODEL} (this may take a while) …"
  CUDA_VISIBLE_DEVICES="${OLLAMA_GPU}" OLLAMA_HOST="${OLLAMA_HOST}" \
    ollama pull "${OLLAMA_MODEL}"
fi

# ---------------------------------------------------------------------------
# 3. Launch each experimental run in a tmux window
# ---------------------------------------------------------------------------

tmux kill-session -t "$SESSION" 2>/dev/null || true

mkdir -p results/
FIRST=1
while IFS=" " read -r RUN_NAME GPU; do
  mkdir -p "results/${RUN_NAME}"
  # Export ANTHROPIC_API_KEY into each window so llm_* monitors can call Claude
  CMD="export ANTHROPIC_API_KEY='${ANTHROPIC_API_KEY:-}'; python runner/experiment_wrapper.py --config ${CONFIG} --run-name ${RUN_NAME} 2>&1 | tee results/${RUN_NAME}/console.log"
  if [ "$FIRST" -eq 1 ]; then
    tmux new-session -d -s "$SESSION" -n "$RUN_NAME" "bash -c '${CMD}'"
    FIRST=0
  else
    tmux new-window -t "$SESSION" -n "$RUN_NAME" "bash -c '${CMD}'"
  fi
  echo "  [GPU ${GPU}] ${RUN_NAME} — tmux window '${RUN_NAME}'"
done <<< "$RUNS"

echo ""
echo "Ollama server : tmux session '${SESSION}_ollama' (GPU ${OLLAMA_GPU}, model ${OLLAMA_MODEL})"
echo "Experiment runs: tmux session '${SESSION}'"
echo ""
echo "Attach with:    tmux attach -t ${SESSION}"
echo "Switch windows: Ctrl-b <number> or Ctrl-b w"
echo "Monitor Ollama: tmux attach -t ${SESSION}_ollama"
echo ""
echo "Heartbeat files: results/<run_name>/heartbeat.txt"
