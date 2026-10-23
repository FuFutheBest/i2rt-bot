#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /absolute/path/to/python312-venv /absolute/path/to/nedf3-wheelhouse" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd "$script_dir/../../.." && pwd -P)
venv=$1
wheelhouse=$2
python_bin=${PYTHON_BIN:-python3.12}
uv_bin=${UV_BIN:-$(command -v uv || true)}

if [[ -z "$uv_bin" ]]; then
  echo "uv is required; set UV_BIN or install uv first" >&2
  exit 2
fi
if [[ "$venv" != /* ]]; then
  echo "The environment path must be absolute: $venv" >&2
  exit 2
fi
if [[ -e "$venv" ]]; then
  echo "Refusing to replace an existing path: $venv" >&2
  exit 2
fi
if [[ "$wheelhouse" != /* || ! -d "$wheelhouse" || -L "$wheelhouse" ]]; then
  echo "Wheelhouse must be an absolute, non-symlink directory: $wheelhouse" >&2
  exit 2
fi
for required in \
  "$repo_root/third_party/openpi/uv.lock" \
  "$repo_root/third_party/i2rt/pyproject.toml" \
  "$repo_root/third_party/lktech_motor_sdk/pyproject.toml"; do
  if [[ ! -f "$required" ]]; then
    echo "Required third_party source is missing: $required" >&2
    exit 2
  fi
done

"$uv_bin" venv --python "$python_bin" "$venv"
UV_PROJECT_ENVIRONMENT="$venv" "$uv_bin" sync --python "$python_bin" \
  --frozen --no-dev --no-install-package mujoco \
  --project "$repo_root/third_party/openpi"
cd "$repo_root"
"$uv_bin" pip install --python "$venv/bin/python" \
  --build-constraints incubator/control_panel/release/requirements/build_constraints.txt \
  --find-links "$wheelhouse" \
  -r incubator/control_panel/release/requirements/main_py312.txt
# i2rt currently pins NumPy 2.2.6 and rerun-sdk >=0.32 while OpenPi's validated
# profile uses NumPy 1.26 and rerun-sdk 0.23. The control path imports without
# either i2rt-only dependency upgrade, so install local integration sources
# without replacing the OpenPi ABI. main_py312.txt already owns LKTech's pyserial
# dependency. Real Yam validation remains mandatory before promotion.
"$uv_bin" pip install --python "$venv/bin/python" --no-deps \
  -e "$repo_root/third_party/i2rt" \
  -e "$repo_root/third_party/lktech_motor_sdk"

set +e
dependency_check=$("$uv_bin" pip check --python "$venv/bin/python" 2>&1)
dependency_status=$?
set -e
if [[ $dependency_status -ne 0 ]]; then
  actual_conflicts=$(printf '%s\n' "$dependency_check" | grep -c '^The package ' || true)
  for documented in \
    'The package `gym-aloha` requires `mujoco' \
    'The package `i2rt` requires `numpy' \
    'The package `i2rt` requires `rerun-sdk'; do
    if ! grep -Fq "$documented" <<<"$dependency_check"; then
      echo "$dependency_check" >&2
      echo "Unified profile dependency check changed unexpectedly" >&2
      exit 1
    fi
  done
  if [[ $actual_conflicts -ne 3 ]]; then
    echo "$dependency_check" >&2
    echo "Unified profile has undocumented dependency conflicts" >&2
    exit 1
  fi
  echo "Dependency check contains only the 3 documented compatibility overrides"
fi

"$venv/bin/python" - <<'PY'
from importlib import metadata
import can
import cv2
import i2rt
import jax
import lktech_motor_sdk
import mujoco
import numpy
import openpi
from ruamel.yaml import YAML
import serial
import torch

assert numpy.__version__ == "1.26.4", numpy.__version__
assert tuple(map(int, cv2.__version__.split(".")[:2])) == (4, 11)
assert metadata.version("i2rt")
assert YAML
print("Unified Python 3.12 source profile import probe passed")
PY
