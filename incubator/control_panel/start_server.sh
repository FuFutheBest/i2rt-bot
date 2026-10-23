#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
RELEASE_ROOT="$(CDPATH= cd -- "$PROJECT_ROOT/.." && pwd)"
HOST="${SUPERINFERENCE_CONTROL_PANEL_HOST:-0.0.0.0}"
PORT="${SUPERINFERENCE_CONTROL_PANEL_PORT:-18088}"

if [[ -f "$RELEASE_ROOT/manifest/release.json" \
      && -x "$RELEASE_ROOT/bin/superinference-control-panel" ]]; then
  exec "$RELEASE_ROOT/bin/superinference-control-panel" \
    --host "$HOST" --port "$PORT" "$@"
fi

PYTHON="${SUPERINFERENCE_CONTROL_PANEL_PYTHON:-python3}"
cd "$PROJECT_ROOT"
exec "$PYTHON" -m incubator.control_panel --host "$HOST" --port "$PORT" "$@"
