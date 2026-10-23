"""Tests for chunk-manager action index tracking."""

import unittest

import numpy as np

from manager.utils.action_manager import BaseChunkManager


class BaseChunkManagerTest(unittest.TestCase):
    def test_tracks_last_dispatched_original_index(self) -> None:
        """RTC receives the index returned by get(), without a one-step offset."""
        manager = BaseChunkManager()
        chunk = {
            "left": np.array([[10.0], [11.0], [12.0]]),
            "right": np.array([[20.0], [21.0], [22.0]]),
        }

        self.assertIsNone(manager.get_original_current_step())

        manager.put(chunk, observation_timestamp=100)
        self.assertIsNone(manager.get_original_current_step())

        first_action = manager.get()
        self.assertIsNotNone(first_action)
        np.testing.assert_array_equal(first_action["left"], np.array([10.0]))
        self.assertEqual(manager.current_step, 1)
        self.assertEqual(manager.get_original_current_step(), 0)

        second_action = manager.get()
        self.assertIsNotNone(second_action)
        np.testing.assert_array_equal(second_action["right"], np.array([21.0]))
        self.assertEqual(manager.get_original_current_step(), 1)

        manager.put(chunk, observation_timestamp=200)
        self.assertIsNone(manager.get_original_current_step())
        self.assertIsNotNone(manager.get())
        self.assertEqual(manager.get_original_current_step(), 0)

        manager.reset()
        self.assertIsNone(manager.get_original_current_step())


if __name__ == "__main__":
    unittest.main()
