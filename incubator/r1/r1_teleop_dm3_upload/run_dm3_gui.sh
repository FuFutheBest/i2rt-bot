#!/usr/bin/env bash
# Launch the DM3 upload GUI with the calibrated upload runtime profile.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_NAME="${SUPERINFERENCE_UPLOAD_CONFIG_NAME:-quest3_collect_r1}"

cd "$REPO_ROOT"
exec python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$CONFIG_NAME" --role upload -- \
  -m incubator.r1.r1_teleop_dm3_upload.gui "$@"
