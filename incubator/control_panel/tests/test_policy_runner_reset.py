from pathlib import Path
import tempfile
import unittest
from unittest import mock

from manager.base_policy_runner import BasePolicyRunner


class PolicyRunnerResetTest(unittest.TestCase):
    def make_runner(self, reset_path: Path) -> BasePolicyRunner:
        runner = BasePolicyRunner.__new__(BasePolicyRunner)
        runner._policy_runner_reset_command_path = str(reset_path)
        runner.last_reset_timestamp = 0
        runner.action_step_interval_ns = 10_000_000
        runner.chunk_manager = mock.Mock()
        runner._pending_chunk_switch_observation_timestamp_ns = 123
        runner.device_action_queues = {"robot": mock.Mock()}
        runner.device_first_actions = {"robot": object()}
        runner._clear_all_shm = mock.Mock(return_value=True)
        runner._write_action_index = mock.Mock(return_value=True)
        return runner

    def test_reset_file_is_removed_only_after_runner_state_is_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reset_path = Path(temporary) / "runner-reset"
            reset_path.touch()
            runner = self.make_runner(reset_path)
            sequence = mock.Mock()
            sequence.attach_mock(runner.chunk_manager.reset, "chunk_reset")
            sequence.attach_mock(runner.device_action_queues["robot"].clear, "queue_clear")
            sequence.attach_mock(runner._clear_all_shm, "clear_shm")
            sequence.attach_mock(runner._write_action_index, "write_index")

            runner._check_and_handle_reset_command()

            self.assertFalse(reset_path.exists())
            self.assertEqual(
                sequence.mock_calls,
                [
                    mock.call.chunk_reset(),
                    mock.call.queue_clear(),
                    mock.call.clear_shm(),
                    mock.call.write_index(0),
                ],
            )
            self.assertIsNone(runner._pending_chunk_switch_observation_timestamp_ns)
            self.assertIsNone(runner.device_first_actions["robot"])

    def test_failed_reset_keeps_command_for_fail_closed_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reset_path = Path(temporary) / "runner-reset"
            reset_path.touch()
            runner = self.make_runner(reset_path)
            runner._clear_all_shm.return_value = False

            runner._check_and_handle_reset_command()

            self.assertTrue(reset_path.exists())
            runner._write_action_index.assert_not_called()


if __name__ == "__main__":
    unittest.main()
