"""Stream a saved I2RT NEDF3 episode into Rerun."""

from __future__ import annotations

import argparse
import json
import math
import socket
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from consumer.io.nedf3_i2rt_contract import (
    NEDF3_CAMERA_TOPICS,
    NEDF3_I2RT_HAND_TOPICS,
    NEDF3_I2RT_JOINT_TOPICS,
    NEDF3_I2RT_LEGACY_BGR_JPEG_VERSION,
    NEDF3_METADATA_TOPIC,
    NEDF3_RECORD_TOPICS,
)

JOINT_NAMES = tuple(f"joint{index}" for index in range(6))
CAMERA_TOPIC_INFO = {topic: role for role, topic in NEDF3_CAMERA_TOPICS.items()}
JOINT_TOPIC_INFO = {
    topic: (role, side)
    for role, side_topics in NEDF3_I2RT_JOINT_TOPICS.items()
    for side, topic in side_topics.items()
}
HAND_TOPIC_INFO = {
    topic: (role, side)
    for role, side_topics in NEDF3_I2RT_HAND_TOPICS.items()
    for side, topic in side_topics.items()
}


@dataclass(frozen=True)
class EpisodeSummary:
    episode_dir: Path
    shards: tuple[Path, ...]
    started_at_ns: int
    ended_at_ns: int
    topic_counts: dict[str, int]
    contract_version: str

    @property
    def duration_s(self) -> float:
        return max(0, self.ended_at_ns - self.started_at_ns) / 1e9


@dataclass(frozen=True)
class I2RTRerunRecord:
    timestamp_ns: int
    topic: str
    kind: str
    role: str
    side: str | None = None
    values: tuple[float, ...] = ()
    image: bytes = b""
    image_width: int = 0
    image_height: int = 0


class _EmbeddedProtobufDecoder:
    def __init__(self) -> None:
        self._message_classes: dict[tuple[str, bytes], Any] = {}

    def decode(self, schema: Any, payload: bytes) -> Any:
        if schema.encoding != "protobuf":
            raise ValueError(
                f"Unsupported schema encoding {schema.encoding!r} for {schema.name!r}"
            )
        key = (str(schema.name), bytes(schema.data))
        message_class = self._message_classes.get(key)
        if message_class is None:
            message_class = self._build_message_class(schema.name, schema.data)
            self._message_classes[key] = message_class
        return message_class.FromString(payload)

    @staticmethod
    def _build_message_class(message_name: str, descriptor_data: bytes) -> Any:
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

        descriptor_set = descriptor_pb2.FileDescriptorSet.FromString(descriptor_data)
        pool = descriptor_pool.DescriptorPool()
        pending = list(descriptor_set.file)
        while pending:
            remaining = []
            for descriptor in pending:
                try:
                    pool.Add(descriptor)
                except Exception:
                    remaining.append(descriptor)
            if len(remaining) == len(pending):
                names = ", ".join(descriptor.name for descriptor in remaining)
                raise ValueError(f"Could not resolve protobuf descriptors: {names}")
            pending = remaining

        descriptor = pool.FindMessageTypeByName(str(message_name))
        factory = message_factory.MessageFactory(pool)
        if hasattr(factory, "GetPrototype"):
            return factory.GetPrototype(descriptor)
        return message_factory.GetMessageClass(descriptor)


def episode_mcap_paths(episode_dir: Path) -> tuple[Path, ...]:
    """Resolve ordered MCAP shards from a finalized NEDF3 episode."""
    episode_dir = episode_dir.expanduser().resolve()
    metadata_path = episode_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"NEDF3 metadata not found: {metadata_path}")

    data_dir = episode_dir / "data"
    index_path = data_dir / "mcap_index.json"
    paths: list[Path] = []
    if index_path.is_file():
        try:
            entries = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid MCAP index: {index_path}") from exc
        if not isinstance(entries, list):
            raise ValueError(f"MCAP index must contain a list: {index_path}")
        for entry in entries:
            filename = entry.get("filename") if isinstance(entry, dict) else None
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"MCAP index contains an invalid filename: {index_path}")
            path = (data_dir / filename).resolve()
            if path.parent != data_dir.resolve():
                raise ValueError(f"MCAP index filename escapes the data directory: {filename}")
            paths.append(path)
    else:
        paths = sorted(data_dir.glob("*.mcap"))

    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MCAP shard not found: {missing[0]}")
    if not paths:
        raise FileNotFoundError(f"No MCAP shards found under: {data_dir}")
    return tuple(paths)


def _iter_messages(shards: Sequence[Path]) -> Iterator[tuple[Any, Any, Any]]:
    from mcap.reader import make_reader

    for shard in shards:
        with shard.open("rb") as stream:
            yield from make_reader(stream).iter_messages()


def inspect_episode(episode_dir: Path) -> EpisodeSummary:
    """Read timestamps and topic counts without decoding image payloads."""
    episode_dir = Path(episode_dir).expanduser().resolve()
    shards = episode_mcap_paths(episode_dir)
    try:
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid NEDF3 metadata: {episode_dir / 'metadata.json'}") from exc
    nested_extra = metadata.get("extra")
    contract_version = str(
        metadata.get("contract_version")
        or (
            nested_extra.get("contract_version")
            if isinstance(nested_extra, dict)
            else ""
        )
        or ""
    )
    topic_counts: Counter[str] = Counter()
    started_at_ns: int | None = None
    ended_at_ns: int | None = None
    for _schema, channel, message in _iter_messages(shards):
        timestamp_ns = int(message.log_time)
        topic_counts[str(channel.topic)] += 1
        started_at_ns = (
            timestamp_ns if started_at_ns is None else min(started_at_ns, timestamp_ns)
        )
        ended_at_ns = (
            timestamp_ns if ended_at_ns is None else max(ended_at_ns, timestamp_ns)
        )
    if started_at_ns is None or ended_at_ns is None:
        raise ValueError(f"Episode contains no MCAP messages: {episode_dir}")
    if not any(topic_counts[topic] for topic in NEDF3_RECORD_TOPICS):
        raise ValueError(f"Episode contains no canonical I2RT NEDF3 topics: {episode_dir}")
    return EpisodeSummary(
        episode_dir=Path(episode_dir).expanduser().resolve(),
        shards=shards,
        started_at_ns=started_at_ns,
        ended_at_ns=ended_at_ns,
        topic_counts=dict(topic_counts),
        contract_version=contract_version,
    )


def decode_legacy_v1_camera_rgb(image: bytes) -> np.ndarray:
    """Recover source RGB values from a v1 JPEG encoded with RGB/BGR swapped."""
    import cv2

    decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("OpenCV could not decode the legacy camera JPEG")
    # v1 passed RGB bytes to a BGR encoder. OpenCV's BGR decode therefore
    # numerically reconstructs the original RGB channel order expected by Rerun.
    return np.ascontiguousarray(decoded)


def _joint_values(decoded: Any, topic: str) -> tuple[float, ...]:
    values = {str(joint.name): float(joint.pos) for joint in decoded.joints}
    missing = [name for name in JOINT_NAMES if name not in values]
    if missing:
        raise ValueError(f"{topic} is missing joints: {missing}")
    result = tuple(values[name] for name in JOINT_NAMES)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{topic} contains a non-finite joint value")
    return result


def _gripper_value(decoded: Any, topic: str) -> float:
    values = {
        str(joint.name): float(joint.pos)
        for joint in decoded.joints
        if str(joint.name) == "gripper"
    }
    if "gripper" not in values or not math.isfinite(values["gripper"]):
        raise ValueError(f"{topic} does not contain a finite gripper value")
    return values["gripper"]


def iter_i2rt_records(episode_dir: Path) -> Iterator[I2RTRerunRecord]:
    """Yield decoded camera, joint, and gripper records in MCAP order."""
    decoder = _EmbeddedProtobufDecoder()
    for schema, channel, message in _iter_messages(episode_mcap_paths(episode_dir)):
        topic = str(channel.topic)
        if topic == NEDF3_METADATA_TOPIC:
            continue
        if (
            topic not in CAMERA_TOPIC_INFO
            and topic not in JOINT_TOPIC_INFO
            and topic not in HAND_TOPIC_INFO
        ):
            continue
        decoded = decoder.decode(schema, message.data)
        timestamp_ns = int(message.log_time)
        if topic in CAMERA_TOPIC_INFO:
            rgb = decoded.rgb
            image = bytes(rgb.data)
            if not image:
                continue
            yield I2RTRerunRecord(
                timestamp_ns=timestamp_ns,
                topic=topic,
                kind="camera",
                role=CAMERA_TOPIC_INFO[topic],
                image=image,
                image_width=int(rgb.cols),
                image_height=int(rgb.rows),
            )
        elif topic in JOINT_TOPIC_INFO:
            role, side = JOINT_TOPIC_INFO[topic]
            yield I2RTRerunRecord(
                timestamp_ns=timestamp_ns,
                topic=topic,
                kind="joints",
                role=role,
                side=side,
                values=_joint_values(decoded, topic),
            )
        else:
            role, side = HAND_TOPIC_INFO[topic]
            yield I2RTRerunRecord(
                timestamp_ns=timestamp_ns,
                topic=topic,
                kind="gripper",
                role=role,
                side=side,
                values=(_gripper_value(decoded, topic),),
            )


def _blueprint(rrb: Any) -> Any:
    return rrb.Blueprint(
        rrb.Grid(
            rrb.Spatial2DView(origin="/cameras/right", name="Right wrist"),
            rrb.Spatial2DView(origin="/cameras/left", name="Left wrist"),
            rrb.Spatial2DView(origin="/cameras/head", name="Head"),
            rrb.TimeSeriesView(origin="/joints/leader/left", name="Leader left"),
            rrb.TimeSeriesView(origin="/joints/follower/left", name="Follower left"),
            rrb.TimeSeriesView(origin="/joints/leader/right", name="Leader right"),
            rrb.TimeSeriesView(origin="/joints/follower/right", name="Follower right"),
            rrb.TimeSeriesView(origin="/grippers", name="Grippers"),
            grid_columns=3,
        ),
        collapse_panels=True,
    )


def _log_series_styles(rr: Any) -> None:
    joint_colors = (
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (23, 190, 207),
    )
    for role in ("leader", "follower"):
        for side in ("left", "right"):
            for name, color in zip(JOINT_NAMES, joint_colors):
                rr.log(
                    f"/joints/{role}/{side}/{name}",
                    rr.SeriesLine(name=name, color=color),
                    static=True,
                )
    gripper_colors = {
        ("leader", "left"): (31, 119, 180),
        ("follower", "left"): (23, 190, 207),
        ("leader", "right"): (255, 127, 14),
        ("follower", "right"): (214, 39, 40),
    }
    for (role, side), color in gripper_colors.items():
        rr.log(
            f"/grippers/{role}/{side}",
            rr.SeriesLine(name=f"{role} {side}", color=color),
            static=True,
        )


def _available_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def visualize_episode(
    episode_dir: Path,
    *,
    output_rrd: Path | None = None,
    memory_limit: str = "4GB",
) -> EpisodeSummary:
    """Stream one episode to a viewer or save it as a Rerun recording."""
    import rerun as rr
    import rerun.blueprint as rrb

    summary = inspect_episode(episode_dir)
    blueprint = _blueprint(rrb)
    rr.init(
        f"i2rt_nedf3_{summary.episode_dir.name}",
        recording_id=str(uuid.uuid4()),
        spawn=False,
    )
    if output_rrd is None:
        from utils.rerun_spawn import spawn_rerun_viewer

        spawn_rerun_viewer(
            port=_available_local_port(),
            memory_limit=memory_limit,
            default_blueprint=blueprint,
        )
    else:
        output_rrd = output_rrd.expanduser().resolve()
        output_rrd.parent.mkdir(parents=True, exist_ok=True)
        rr.save(output_rrd, default_blueprint=blueprint)

    _log_series_styles(rr)
    repair_legacy_camera_order = (
        summary.contract_version == NEDF3_I2RT_LEGACY_BGR_JPEG_VERSION
    )
    for record in iter_i2rt_records(summary.episode_dir):
        rr.set_time_seconds(
            "episode_time", (record.timestamp_ns - summary.started_at_ns) / 1e9
        )
        if record.kind == "camera":
            image = (
                rr.Image(decode_legacy_v1_camera_rgb(record.image))
                if repair_legacy_camera_order
                else rr.EncodedImage(contents=record.image, media_type="image/jpeg")
            )
            rr.log(
                f"/cameras/{record.role}",
                image,
            )
        elif record.kind == "joints":
            assert record.side is not None
            for name, value in zip(JOINT_NAMES, record.values):
                rr.log(f"/joints/{record.role}/{record.side}/{name}", rr.Scalar(value))
        else:
            assert record.side is not None
            rr.log(
                f"/grippers/{record.role}/{record.side}", rr.Scalar(record.values[0])
            )
    rr.disconnect()
    return summary


def _episode_finished_at_ns(path: Path) -> int:
    stats_path = path / "episode_stats.json"
    if stats_path.is_file():
        try:
            value = json.loads(stats_path.read_text(encoding="utf-8")).get("ended_at_ns")
            return int(value)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return path.stat().st_mtime_ns


def latest_episode(data_dir: Path) -> Path:
    """Return the newest finalized, non-quarantined episode below a data root."""
    data_dir = data_dir.expanduser().resolve()
    candidates = [
        path
        for path in data_dir.rglob("episode_*")
        if path.is_dir()
        and not path.name.endswith("_inprogress")
        and "quarantine" not in path.parts
        and (path / "metadata.json").is_file()
        and (path / "data").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"No finalized NEDF3 episodes found under: {data_dir}")
    return max(candidates, key=lambda path: (_episode_finished_at_ns(path), path.name))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episode",
        nargs="?",
        type=Path,
        help="Finalized episode directory. The newest episode is used when omitted.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data root searched when no episode is supplied (default: data).",
    )
    parser.add_argument("--output", type=Path, help="Write an .rrd file instead of opening Rerun.")
    parser.add_argument("--memory-limit", default="4GB", help="Rerun viewer memory limit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    episode = args.episode.expanduser().resolve() if args.episode else latest_episode(args.data_dir)
    summary = visualize_episode(
        episode,
        output_rrd=args.output,
        memory_limit=args.memory_limit,
    )
    destination = f" -> {args.output.expanduser().resolve()}" if args.output else ""
    print(
        f"Visualized {summary.episode_dir} ({summary.duration_s:.2f}s, "
        f"{sum(summary.topic_counts.values())} messages){destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
