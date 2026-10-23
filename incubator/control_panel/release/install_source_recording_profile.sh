#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /absolute/path/to/python /absolute/path/to/nedf3-wheelhouse" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd "$script_dir/../../.." && pwd -P)
python_path=$1
wheelhouse=$2
uv_bin=${UV_BIN:-$(command -v uv || true)}

if [[ -z "$uv_bin" ]]; then
  echo "uv is required; set UV_BIN or install uv first" >&2
  exit 2
fi
if [[ "$python_path" != /* || ! -x "$python_path" ]]; then
  echo "Python must be an absolute executable path: $python_path" >&2
  exit 2
fi
if [[ "$wheelhouse" != /* || ! -d "$wheelhouse" || -L "$wheelhouse" ]]; then
  echo "Wheelhouse must be an absolute, non-symlink directory: $wheelhouse" >&2
  exit 2
fi

"$python_path" - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"NEDF3 source profile requires Python 3.12, got {sys.version}")
PY

for wheel_pattern in 'dm3_core-0.2.18-*.whl' 'nmx_msg-2.2.0-*.whl'; do
  shopt -s nullglob
  matches=("$wheelhouse"/$wheel_pattern)
  shopt -u nullglob
  if [[ ${#matches[@]} -ne 1 || -L "${matches[0]:-}" ]]; then
    echo "Expected exactly one regular wheel matching $wheel_pattern in $wheelhouse" >&2
    exit 2
  fi
done

cd "$repo_root"
"$uv_bin" pip install --python "$python_path" \
  --build-constraints incubator/control_panel/release/requirements/build_constraints.txt \
  --find-links "$wheelhouse" \
  -r incubator/control_panel/release/requirements/recording.txt

"$python_path" - <<'PY'
from importlib import metadata

import rerun

from consumer.io.async_infer_data_saver_NEDF3 import InferenceDataSaverConsumer
from dm3_core.mcap import McapFileType, ReaderFactory, WriterFactory
from dm3_core.nedf import NEDFFactory, NEDFReaderBase, NEDFType, NEDFWriterBase
from nmx_msg.Arm_pb2 import ArmEndForce, ArmState, ArmTcpCmd, ArmTcpState
from nmx_msg.Hand_pb2 import HandCmd, HandState
from nmx_msg.Image_pb2 import RGBD
from nmx_msg.Metadata_pb2 import Metadata

expected = {
    "dm3-core": "0.2.18",
    "nmx-msg": "2.2.0",
    "rerun-sdk": "0.22.1",
}
actual = {name: metadata.version(name) for name in expected}
assert actual == expected, (actual, expected)
assert hasattr(NEDFType, "NEDF3")
for owner, method in (
    (NEDFFactory, "get_writer"),
    (NEDFFactory, "get_reader"),
    (NEDFWriterBase, "write_message"),
    (NEDFReaderBase, "iter_decoded_messages"),
    (NEDFReaderBase, "get_raw_topics_timestamps"),
):
    assert callable(getattr(owner, method, None)), f"missing {owner.__name__}.{method}"
assert ReaderFactory and WriterFactory and McapFileType
assert InferenceDataSaverConsumer
assert rerun
for message_type in (
    RGBD, ArmTcpCmd, ArmTcpState, ArmState, ArmEndForce,
    HandCmd, HandState, Metadata,
):
    assert hasattr(message_type, "SerializeToString")
print("NEDF3 recording profile import probe passed", actual)
PY
