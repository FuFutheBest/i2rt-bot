#!/usr/bin/env python3
# ruff: noqa: E501
"""Generate a standalone joint14 target-versus-measured RoboTwin trace page."""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path
from typing import Any

JOINT_NAMES = (
    "left_j0",
    "left_j1",
    "left_j2",
    "left_j3",
    "left_j4",
    "left_j5",
    "left_gripper",
    "right_j0",
    "right_j1",
    "right_j2",
    "right_j3",
    "right_j4",
    "right_j5",
    "right_gripper",
)


def _load_records(trace_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {trace_path}:{line_number}") from exc
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _vector(record: dict[str, Any], key: str) -> list[float] | None:
    value = record.get(key)
    if not isinstance(value, list) or len(value) != 14:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _time_ns(record: dict[str, Any]) -> int | None:
    value = record.get("timestamp_ns", record.get("event_ns"))
    try:
        timestamp_ns = int(value)
    except (TypeError, ValueError):
        return None
    return timestamp_ns if timestamp_ns > 0 else None


def _build_data(records: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = [record for record in records if record.get("event_type") in {"sim_action_applied", "sim_measured_state"}]
    timestamps = [timestamp for record in relevant if (timestamp := _time_ns(record)) is not None]
    origin = min(timestamps) if timestamps else 0

    def point(record: dict[str, Any], vector: list[float]) -> list[Any] | None:
        timestamp = _time_ns(record)
        if timestamp is None:
            return None
        return [(timestamp - origin) / 1_000_000_000.0, vector]

    targets: list[list[Any]] = []
    measured: list[list[Any]] = []
    measured_targets: list[list[Any]] = []
    latencies_ms: list[float] = []
    for record in relevant:
        event_type = record.get("event_type")
        if event_type == "sim_action_applied":
            target = _vector(record, "target_qpos")
            if target is not None and (item := point(record, target)) is not None:
                targets.append(item)
            try:
                latency = (int(record["executed_action_timestamp_ns"]) - int(record["control_write_timestamp_ns"])) / 1e6
            except (KeyError, TypeError, ValueError):
                continue
            latencies_ms.append(latency)
        elif event_type == "sim_measured_state":
            qpos = _vector(record, "measured_qpos")
            target = _vector(record, "target_qpos")
            if qpos is not None and (item := point(record, qpos)) is not None:
                measured.append(item)
            if target is not None and (item := point(record, target)) is not None:
                measured_targets.append(item)

    return {
        "jointNames": JOINT_NAMES,
        "targets": targets,
        "measured": measured,
        "measuredTargets": measured_targets,
        "summary": {
            "applied": len(targets),
            "measured": len(measured),
            "latencyMs": {
                "min": min(latencies_ms) if latencies_ms else None,
                "max": max(latencies_ms) if latencies_ms else None,
                "mean": sum(latencies_ms) / len(latencies_ms) if latencies_ms else None,
            },
        },
    }


def _html(data: dict[str, Any], trace_path: Path) -> str:
    serialized = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    title = f"RoboTwin Joint Trace - {trace_path.name}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light; font-family: Arial, sans-serif; background: #f4f6f7; color: #1d252c; }}
body {{ margin: 0; }}
main {{ max-width: 1160px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 22px; margin: 0 0 6px; }}
p {{ margin: 0; color: #53606b; font-size: 14px; }}
.toolbar {{ display: flex; gap: 16px; align-items: center; margin: 20px 0 12px; flex-wrap: wrap; }}
select {{ padding: 7px 9px; border: 1px solid #abb5bc; border-radius: 4px; background: white; }}
.legend {{ display: flex; gap: 14px; font-size: 13px; }}
.legend span::before {{ content: ""; display: inline-block; width: 16px; height: 3px; margin-right: 5px; vertical-align: middle; }}
.target::before {{ background: #d05a36; }} .actual::before {{ background: #167c80; }} .held::before {{ background: #8c6b26; }}
.chart {{ border: 1px solid #c5cdd1; background: white; min-height: 430px; }}
canvas {{ display: block; width: 100%; height: 430px; }}
.stats {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1px; background: #c5cdd1; margin-top: 16px; border: 1px solid #c5cdd1; }}
.stat {{ background: white; padding: 12px; }} .stat b {{ display: block; font-size: 13px; color: #53606b; font-weight: normal; }} .stat strong {{ font-size: 19px; }}
@media (max-width: 620px) {{ main {{ padding: 14px; }} .stats {{ grid-template-columns: 1fr; }} canvas {{ height: 340px; }} .chart {{ min-height: 340px; }} }}
</style>
</head>
<body><main>
<h1>RoboTwin Joint Trace</h1><p>{trace_path}</p>
<div class="toolbar"><label>Joint <select id="joint"></select></label><div class="legend"><span class="target">applied target</span><span class="actual">measured</span><span class="held">held target</span></div></div>
<div class="chart"><canvas id="chart"></canvas></div>
<section class="stats"><div class="stat"><b>Applied setpoints</b><strong id="applied"></strong></div><div class="stat"><b>Measured samples</b><strong id="measured"></strong></div><div class="stat"><b>Control write to physical apply</b><strong id="latency"></strong></div></section>
</main><script>
const data = {serialized};
const select = document.getElementById('joint');
data.jointNames.forEach((name, index) => {{ const option = new Option(name, index); select.add(option); }});
document.getElementById('applied').textContent = data.summary.applied;
document.getElementById('measured').textContent = data.summary.measured;
const latency = data.summary.latencyMs.mean;
document.getElementById('latency').textContent = latency === null ? 'n/a' : `${{latency.toFixed(2)}} ms mean`;
const canvas = document.getElementById('chart'); const context = canvas.getContext('2d');
function drawLine(points, joint, color, scale) {{
  if (!points.length) return; context.strokeStyle = color; context.lineWidth = 1.7; context.beginPath();
  points.forEach((point, index) => {{ const x = scale.x(point[0]); const y = scale.y(point[1][joint]); index ? context.lineTo(x,y) : context.moveTo(x,y); }}); context.stroke();
}}
function draw() {{
  const ratio = window.devicePixelRatio || 1; const width = canvas.clientWidth; const height = canvas.clientHeight;
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio); context.setTransform(ratio,0,0,ratio,0,0); context.clearRect(0,0,width,height);
  const joint = Number(select.value); const all = [...data.targets, ...data.measured, ...data.measuredTargets];
  if (!all.length) {{ context.fillStyle='#53606b'; context.font='14px Arial'; context.fillText('No sim_action_applied or sim_measured_state records found.', 22, 34); return; }}
  const times = all.map(point => point[0]); const values = all.map(point => point[1][joint]);
  let minX = Math.min(...times), maxX = Math.max(...times), minY = Math.min(...values), maxY = Math.max(...values);
  if (minX === maxX) {{ minX -= .5; maxX += .5; }} if (minY === maxY) {{ minY -= .1; maxY += .1; }}
  const pad = {{ left: 58, right: 18, top: 20, bottom: 42 }}; const graphW=width-pad.left-pad.right, graphH=height-pad.top-pad.bottom;
  const scale = {{ x: value => pad.left + (value-minX)/(maxX-minX)*graphW, y: value => pad.top + (maxY-value)/(maxY-minY)*graphH }};
  context.strokeStyle='#e0e5e7'; context.lineWidth=1; context.fillStyle='#53606b'; context.font='12px Arial';
  for(let i=0;i<=4;i++) {{ const y=pad.top+i*graphH/4; const v=maxY-(maxY-minY)*i/4; context.beginPath(); context.moveTo(pad.left,y); context.lineTo(width-pad.right,y); context.stroke(); context.fillText(v.toFixed(3),4,y+4); }}
  context.fillText(`${{minX.toFixed(2)}} s`,pad.left,height-14); context.fillText(`${{maxX.toFixed(2)}} s`,width-pad.right-48,height-14);
  drawLine(data.measuredTargets,joint,'#8c6b26',scale); drawLine(data.targets,joint,'#d05a36',scale); drawLine(data.measured,joint,'#167c80',scale);
}}
select.addEventListener('change', draw); window.addEventListener('resize', draw); draw();
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="Bridge ActionStateXYZ JSONL trace")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--open", action="store_true", help="Open the generated HTML in the default browser")
    args = parser.parse_args()
    trace_path = args.trace.expanduser().resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(f"RoboTwin trace not found: {trace_path}")
    output_path = args.output.expanduser().resolve() if args.output else trace_path.with_suffix(".robotwin.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_html(_build_data(_load_records(trace_path)), trace_path), encoding="utf-8")
    print(output_path)
    if args.open:
        webbrowser.open(output_path.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
