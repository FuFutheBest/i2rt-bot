import json
import os
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

from incubator.control_panel.recording_control import (
    RecordingControlPlane,
    WholeEpisodeCommandConsumer,
    _canonical_json,
    atomic_write_json,
    quarantine_episode,
    preflight_recorder_python,
    resolve_recorder_python,
    write_conversion_source_receipt,
    write_incomplete_metadata,
    write_recording_decision,
)


def _home_target(candidate_id: str = "left-01") -> dict:
    return {
        "schema_version": 1,
        "source": "home_pose_candidates",
        "config_name": "demo",
        "config_hash": "a" * 64,
        "arms": {
            "left": {
                "candidate_id": candidate_id,
                "joints": [0.1] * 7,
                "collision_checked": False,
            },
            "right": {
                "candidate_id": "right-01",
                "joints": [-0.1] * 7,
                "collision_checked": False,
            },
        },
    }


class RecordingControlPlaneTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.control = RecordingControlPlane(
            self.root / "run",
            "run-test-1",
            owner_pid=os.getpid(),
            status_flush_interval_s=0.0,
        )
        self.episode_dir = self.root / "episodes" / "episode_0000_temp"
        self.episode_dir.mkdir(parents=True)
        self.episode_id = self.control.begin_episode(
            index=0, episode_dir=self.episode_dir, episode_id="episode-test-1"
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def command(
        self,
        command_id: str,
        action: str,
        *,
        idempotency_key: str = None,
        expected_revision: int = None,
    ) -> dict:
        if expected_revision is None:
            expected_revision = self.control.snapshot()["revision"]
        return {
            "schema_version": 1,
            "run_id": self.control.run_id,
            "command_id": command_id,
            "idempotency_key": idempotency_key or command_id,
            "action": action,
            "episode_id": self.episode_id,
            "expected_revision": expected_revision,
            "created_at": "2026-07-13T00:00:00+00:00",
        }

    def spool(self, command: dict) -> None:
        atomic_write_json(
            self.control.commands_dir / f"{command['command_id']}.json", command
        )

    def test_dense_segments_use_committed_reference_frame_boundaries_and_crc(self) -> None:
        for timestamp in (1_000_000_000, 1_010_000_000, 1_020_000_000):
            self.control.note_reference_commit(timestamp)

        first = self.command("segment-1", "segment_accept")
        self.spool(first)
        self.control.process_commands(
            lambda command: self.control.append_segment("accept", command)
        )
        self.control.note_reference_commit(1_030_000_000)
        self.control.note_reference_commit(1_040_000_000)
        second = self.command("segment-2", "segment_reject")
        self.spool(second)
        self.control.process_commands(
            lambda command: self.control.append_segment("reject", command)
        )

        lines = [
            json.loads(line)
            for line in (self.episode_dir / "dense_labels.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual([(line["start_frame"], line["end_frame"]) for line in lines], [(0, 3), (3, 5)])
        self.assertEqual([line["label"] for line in lines], ["accept", "reject"])
        for line in lines:
            crc = line.pop("crc32")
            self.assertEqual(crc, f"{zlib.crc32(_canonical_json(line)) & 0xFFFFFFFF:08x}")

        status = json.loads(self.control.status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["episode"]["frame_count"], 5)
        self.assertEqual(status["episode"]["untagged_start_frame"], 5)
        self.assertEqual(status["episode"]["checkpoint_revision"], 2)
        self.assertEqual(status["reference"]["name"], "RizonRobot_2_control")

    def test_no_new_reference_frame_rejects_without_advancing_checkpoint(self) -> None:
        self.control.note_reference_commit(100)
        applied = self.command("segment-applied", "segment_accept")
        self.spool(applied)
        self.control.process_commands(
            lambda command: self.control.append_segment("accept", command)
        )
        before = self.control.snapshot()

        empty = self.command("segment-empty", "segment_reject")
        self.spool(empty)
        self.control.process_commands(
            lambda command: self.control.append_segment("reject", command)
        )
        ack = json.loads(
            (self.control.acks_dir / "segment-empty.json").read_text(encoding="utf-8")
        )
        after = self.control.snapshot()
        self.assertEqual(ack["status"], "rejected")
        self.assertEqual(ack["error"]["code"], "no_new_reference_frame")
        self.assertEqual(after["episode"]["checkpoint_revision"], before["episode"]["checkpoint_revision"])
        self.assertEqual(len(after["segments"]), len(before["segments"]))

    def test_idempotency_replay_does_not_duplicate_dense_line(self) -> None:
        self.control.note_reference_commit(100)
        original = self.command(
            "segment-original", "segment_accept", idempotency_key="same-request"
        )
        self.spool(original)
        self.control.process_commands(
            lambda command: self.control.append_segment("accept", command)
        )

        replay = self.command(
            "segment-replay", "segment_accept", idempotency_key="same-request"
        )
        self.spool(replay)
        self.control.process_commands(
            lambda command: self.control.append_segment("accept", command)
        )
        lines = (self.episode_dir / "dense_labels.jsonl").read_text(encoding="utf-8").splitlines()
        ack = json.loads(
            (self.control.acks_dir / "segment-replay.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(lines), 1)
        self.assertTrue(ack["replayed"])
        self.assertFalse((self.control.commands_dir / "segment-replay.json").exists())

    def test_revision_conflict_is_rejected(self) -> None:
        command = self.command(
            "stale-command", "segment_accept", expected_revision=999999
        )
        self.spool(command)
        self.control.process_commands(
            lambda item: self.control.append_segment("accept", item)
        )
        ack = json.loads(
            (self.control.acks_dir / "stale-command.json").read_text(encoding="utf-8")
        )
        self.assertEqual(ack["status"], "rejected")
        self.assertEqual(ack["error"]["code"], "revision_conflict")
        self.assertEqual(ack["owner_pid"], os.getpid())

    def test_empty_poll_does_not_rewrite_status_at_collection_frequency(self) -> None:
        before = self.control.status_path.stat().st_mtime_ns
        self.assertEqual(self.control.process_commands(lambda command: None), 0)
        after = self.control.status_path.stat().st_mtime_ns
        self.assertEqual(after, before)

    def test_episode_accept_closes_only_the_untagged_tail(self) -> None:
        for timestamp in (10, 20):
            self.control.note_reference_commit(timestamp)
        first = self.command("first", "segment_reject")
        self.control.append_segment("reject", first)
        self.control.note_reference_commit(30)
        self.control.note_reference_commit(40)
        whole = self.command("whole", "episode_accept")
        tail = self.control.accept_tail(whole)
        self.assertEqual((tail["start_frame"], tail["end_frame"]), (2, 4))
        self.assertEqual(tail["label"], "accept")

    def test_episode_reject_closes_only_the_untagged_tail(self) -> None:
        for timestamp in (10, 20):
            self.control.note_reference_commit(timestamp)
        first = self.command("first-accept", "segment_accept")
        self.control.append_segment("accept", first)
        self.control.note_reference_commit(30)
        self.control.note_reference_commit(40)
        whole = self.command("whole-reject", "episode_discard")
        tail = self.control.reject_tail(whole)
        self.assertEqual((tail["start_frame"], tail["end_frame"]), (2, 4))
        self.assertEqual(tail["label"], "reject")
        self.assertEqual(tail["source_action"], "episode_discard")

    def test_whole_episode_command_is_not_consumed_by_saver_owner(self) -> None:
        command = self.command("whole-owned", "episode_accept")
        self.spool(command)
        saver_handler = mock.Mock()
        processed = self.control.process_commands(
            saver_handler,
            allowed_actions={"segment_accept", "segment_reject", "shutdown"},
        )
        self.assertEqual(processed, 0)
        saver_handler.assert_not_called()
        self.assertTrue((self.control.commands_dir / "whole-owned.json").exists())
        self.assertFalse((self.control.acks_dir / "whole-owned.json").exists())

    def test_policy_owner_acks_only_after_next_episode_is_recording(self) -> None:
        command = self.command("whole-transition", "episode_accept")
        command["home_target"] = _home_target()
        self.spool(command)
        whole_owner = WholeEpisodeCommandConsumer(
            self.control.control_dir,
            self.control.run_id,
            owner_pid=os.getpid(),
        )
        transition = mock.Mock()

        def perform_transition(item: dict) -> None:
            transition(item)
            self.control.set_state("finalizing", ready=True, active=True)
            self.control.arm()
            next_dir = self.root / "episodes" / "episode_0001_temp"
            next_dir.mkdir()
            self.control.begin_episode(
                index=1,
                episode_dir=next_dir,
                episode_id="episode-test-2",
            )
            whole_owner.wait_for_next_episode(
                self.episode_id, auto_restart=True, timeout_s=1.0
            )

        processed = whole_owner.process_commands(perform_transition)
        self.assertEqual(processed, 1)
        transition.assert_called_once()
        self.assertEqual(transition.call_args.args[0]["home_target"], _home_target())
        ack = json.loads(
            (self.control.acks_dir / "whole-transition.json").read_text(encoding="utf-8")
        )
        self.assertEqual(ack["status"], "applied")
        self.assertTrue(ack["recording"])
        self.assertFalse((self.control.commands_dir / "whole-transition.json").exists())

    def test_whole_episode_idempotency_fingerprint_includes_home_target(self) -> None:
        command = self.command(
            "whole-home-first",
            "episode_accept",
            idempotency_key="same-home-request",
        )
        command["home_target"] = _home_target()
        self.spool(command)
        whole_owner = WholeEpisodeCommandConsumer(
            self.control.control_dir,
            self.control.run_id,
            owner_pid=os.getpid(),
        )

        def complete(_item: dict) -> None:
            self.control.set_state("finalizing", ready=True, active=True)
            self.control.arm()
            next_dir = self.root / "episodes" / "episode_0001_temp"
            next_dir.mkdir()
            self.control.begin_episode(
                index=1,
                episode_dir=next_dir,
                episode_id="episode-test-2",
            )

        self.assertEqual(whole_owner.process_commands(complete), 1)

        replay = self.command(
            "whole-home-changed",
            "episode_accept",
            idempotency_key="same-home-request",
        )
        replay["episode_id"] = "episode-test-2"
        replay["home_target"] = _home_target(candidate_id="left-02")
        self.spool(replay)
        self.assertEqual(whole_owner.process_commands(lambda _item: None), 1)
        ack = json.loads(
            (self.control.acks_dir / "whole-home-changed.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(ack["status"], "rejected")
        self.assertEqual(ack["error"]["code"], "idempotency_conflict")

    def test_policy_owner_ignores_dense_command(self) -> None:
        command = self.command("dense-owned", "segment_accept")
        self.spool(command)
        whole_owner = WholeEpisodeCommandConsumer(
            self.control.control_dir, self.control.run_id
        )
        handler = mock.Mock()
        self.assertEqual(whole_owner.process_commands(handler), 0)
        handler.assert_not_called()
        self.assertTrue((self.control.commands_dir / "dense-owned.json").exists())


class IncompleteEpisodeTest(unittest.TestCase):
    def test_recording_decision_is_separate_from_conversion_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            episode = Path(temp) / "episode_0001"
            episode.mkdir()
            (episode / "metadata.json").write_text(
                json.dumps({"data_format": "NEDF3"}), encoding="utf-8"
            )
            (episode / "episode_result.json").write_text(
                json.dumps({"episode_result": "success"}), encoding="utf-8"
            )
            receipt = write_conversion_source_receipt(
                episode,
                episode_id="episode-id",
                episode_index=1,
                result="success",
            )
            decision = write_recording_decision(
                receipt,
                {
                    "action": "episode_accept",
                    "command_id": "command-id",
                    "idempotency_key": "idempotency-id",
                },
            )
            self.assertEqual(decision["status"], "accepted")
            self.assertEqual(
                json.loads((episode / ".conversion_source.json").read_text())["fingerprint"],
                receipt["fingerprint"],
            )

    def test_incomplete_metadata_and_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode = root / "episode_0007_temp"
            episode.mkdir()
            (episode / "data.mcap").write_bytes(b"partial")
            metadata = write_incomplete_metadata(
                episode,
                episode_id="episode-id",
                episode_index=7,
                reason="shutdown",
            )
            target, renamed = quarantine_episode(
                episode,
                root,
                episode_index=7,
                episode_id="episode-id",
            )
            self.assertTrue(renamed)
            self.assertFalse(episode.exists())
            self.assertTrue((target / metadata.name).exists())
            self.assertEqual((target / "data.mcap").read_bytes(), b"partial")

    def test_quarantine_rename_failure_preserves_temp_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode = root / "episode_0001_temp"
            episode.mkdir()
            (episode / "data.mcap").write_bytes(b"must-survive")
            with mock.patch(
                "incubator.control_panel.recording_control.os.replace",
                side_effect=OSError("injected"),
            ):
                target, renamed = quarantine_episode(
                    episode,
                    root,
                    episode_index=1,
                    episode_id="episode-id",
                )
            self.assertFalse(renamed)
            self.assertEqual(target, episode)
            self.assertEqual((episode / "data.mcap").read_bytes(), b"must-survive")


class RecorderPythonProfileTest(unittest.TestCase):
    def test_server_profile_overrides_stale_config_path(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"SUPERINFERENCE_RECORDER_PYTHON": sys.executable},
        ):
            selected = resolve_recorder_python(
                "/home/other-user/missing/python", "/missing/fallback/python"
            )
        self.assertEqual(selected, sys.executable)

    def test_preflight_reports_dependency_import_failure(self) -> None:
        failed = subprocess.CompletedProcess(
            [sys.executable], returncode=1, stdout="", stderr="No module named nmx_msg"
        )
        with mock.patch(
            "incubator.control_panel.recording_control.subprocess.run", return_value=failed
        ):
            ok, error = preflight_recorder_python(sys.executable)
        self.assertFalse(ok)
        self.assertIn("nmx_msg", error)

    def test_preflight_accepts_profile_when_import_probe_succeeds(self) -> None:
        passed = subprocess.CompletedProcess(
            [sys.executable], returncode=0, stdout="", stderr=""
        )
        with mock.patch(
            "incubator.control_panel.recording_control.subprocess.run", return_value=passed
        ):
            ok, error = preflight_recorder_python(sys.executable)
        self.assertTrue(ok)
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
