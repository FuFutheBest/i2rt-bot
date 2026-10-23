#!/usr/bin/env bash
set -euo pipefail

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
PROJECT="$HOME/SuperInference.lingbotva"
LOG_DIR="$PROJECT/logs"
LOG_FILE="$LOG_DIR/pull_dualearphone_8x8x1_8000pre.log"
LOCK_FILE="$HOME/.cache/pull_dualearphone_8x8x1_8000pre.lock"
mkdir -p "$LOG_DIR" "$(dirname "$LOCK_FILE")"

{
  echo "===== $(date '+%F %T %Z %z') pull dualearphone_8x8x1_8000pre start ====="
  flock -n 9 || { echo "Another pull task is still running; skip."; exit 0; }
  cd "$PROJECT"
  REMOTE_HOST="PAI-4051" \
  REMOTE_RUN_DIR="/mnt/workspace/shenyibo/nmx_vla/results/dualearphone_8x8x1_8000pre" \
  LOCAL_ROOT="$HOME/lingbotva_ckpts" \
  LOCAL_RUN_DIR="$HOME/lingbotva_ckpts/dualearphone_8x8x1_8000pre" \
  PRETRAINED_MODEL_PATH="$HOME/lingbot-va-pretrained-ckpt" \
    bash scripts/lingbotva/pull_checkpoint.sh
  echo "===== $(date '+%F %T %Z %z') pull dualearphone_8x8x1_8000pre done ====="
} 9>"$LOCK_FILE" >> "$LOG_FILE" 2>&1
