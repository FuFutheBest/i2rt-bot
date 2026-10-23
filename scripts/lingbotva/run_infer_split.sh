#!/usr/bin/env bash
#
# run_infer_split.sh - Launch the OpenPi UMI dual-arm real-robot inference using
# two calibrated Python profiles that communicate only via named shared memory.
#
# Why two environments:
#   - device_starter.py drives the RizonRobot device and needs the profile bound
#     to the selected config's `device` role.
#   - consumer/inference/openpi_policy_infer.py loads OpenPI from the profile
#     bound to the selected config's `policy` role.
#
# Usage:
#   scripts/lingbotva/run_infer_split.sh devices [CONFIG_NAME]   # terminal A: start FIRST, wait for device init
#   scripts/lingbotva/run_infer_split.sh policy  [CONFIG_NAME]    # terminal B: start AFTER devices are ready
#
# Shutdown order: Ctrl+C terminal B (policy) first, then terminal A (devices).
#
# Author: SuperInference
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_CLI="${REPO_ROOT}/scripts/diagnostics/runtime_profiles.py"
DEVICE_PYTHON="${DEVICE_PYTHON:-${SUPERINFERENCE_DEVICE_PYTHON:-}}"
POLICY_PYTHON="${POLICY_PYTHON:-${SUPERINFERENCE_INFERENCE_PYTHON:-}}"
CONFIG_NAME="${2:-policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi}"

usage() {
    cat <<'EOF'
Usage: scripts/lingbotva/run_infer_split.sh <side> [CONFIG_NAME]

  side         devices | policy
                 devices -> device_starter.py using the bound `device` profile
                 policy  -> openpi_policy_infer using the bound `policy` profile
  CONFIG_NAME  Hydra config name (default: policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi)

Env overrides:
  DEVICE_PYTHON / POLICY_PYTHON  explicit Python executables, preferred when set

Configure persistent workstation choices with scripts/diagnostics/runtime_profiles.py.

Start `devices` first, wait for device initialization, then start `policy`.
EOF
}

side="${1:-}"
if [[ -z "${side}" || "${side}" == "-h" || "${side}" == "--help" ]]; then
    usage
    exit 0
fi

cd "${REPO_ROOT}"

case "${side}" in
    devices)
        if [[ -n "${DEVICE_PYTHON}" ]]; then
            echo "[run_infer_split] device side -> python=${DEVICE_PYTHON}, config=${CONFIG_NAME}"
            exec "${DEVICE_PYTHON}" device_starter.py --config-name "${CONFIG_NAME}"
        fi
        echo "[run_infer_split] device side -> runtime profile, config=${CONFIG_NAME}"
        exec python3 "${RUNTIME_CLI}" --repo-root "${REPO_ROOT}" run \
            --config-name "${CONFIG_NAME}" --role device -- \
            "${REPO_ROOT}/device_starter.py" --config-name "${CONFIG_NAME}"
        ;;
    policy)
        if [[ -n "${POLICY_PYTHON}" ]]; then
            echo "[run_infer_split] policy side -> python=${POLICY_PYTHON}, config=${CONFIG_NAME}"
            exec "${POLICY_PYTHON}" -m consumer.inference.openpi_policy_infer --config-name "${CONFIG_NAME}"
        fi
        echo "[run_infer_split] policy side -> runtime profile, config=${CONFIG_NAME}"
        exec python3 "${RUNTIME_CLI}" --repo-root "${REPO_ROOT}" run \
            --config-name "${CONFIG_NAME}" --role policy -- \
            -m consumer.inference.openpi_policy_infer --config-name "${CONFIG_NAME}"
        ;;
    *)
        echo "Unknown side: ${side}" >&2
        usage
        exit 1
        ;;
esac
