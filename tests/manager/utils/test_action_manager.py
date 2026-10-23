import numpy as np

from manager.utils.action_manager import BaseChunkManager


def test_base_chunk_manager_reports_latest_original_action_index():
    manager = BaseChunkManager()
    chunk = {"robot": np.arange(12).reshape(3, 4)}

    manager.put(chunk, observation_timestamp=123)
    assert manager.get_original_current_step() is None

    np.testing.assert_array_equal(manager.get()["robot"], chunk["robot"][0])
    assert manager.get_original_current_step() == 0

    np.testing.assert_array_equal(manager.get()["robot"], chunk["robot"][1])
    assert manager.get_original_current_step() == 1

    np.testing.assert_array_equal(manager.get()["robot"], chunk["robot"][2])
    assert manager.get_original_current_step() == 2

    assert manager.get() is None
    assert manager.get_original_current_step() == 2


def test_base_chunk_manager_clears_action_index_for_new_chunk_and_reset():
    manager = BaseChunkManager()
    chunk = {"robot": np.zeros((2, 4))}

    manager.put(chunk, observation_timestamp=123)
    manager.get()
    assert manager.get_original_current_step() == 0

    manager.put(chunk, observation_timestamp=456)
    assert manager.get_original_current_step() is None

    manager.get()
    manager.reset()
    assert manager.get_original_current_step() is None


def test_base_chunk_manager_only_skips_steps_when_replacing_old_chunk():
    manager = BaseChunkManager(drop_first_actions=2)
    first_chunk = {"robot": np.arange(16).reshape(4, 4)}
    second_chunk = {"robot": np.arange(20, 36).reshape(4, 4)}

    manager.put(first_chunk, observation_timestamp=123)
    np.testing.assert_array_equal(manager.get()["robot"], first_chunk["robot"][0])
    assert manager.get_original_current_step() == 0

    manager.put(second_chunk, observation_timestamp=456)
    assert manager.get_original_current_step() is None
    np.testing.assert_array_equal(manager.get()["robot"], second_chunk["robot"][2])
    assert manager.get_original_current_step() == 2

    manager.reset()
    manager.put(first_chunk, observation_timestamp=789)
    np.testing.assert_array_equal(manager.get()["robot"], first_chunk["robot"][0])
    assert manager.get_original_current_step() == 0


def test_base_chunk_manager_rejects_negative_drop_count():
    try:
        BaseChunkManager(drop_first_actions=-1)
    except ValueError as exc:
        assert str(exc) == "drop_first_actions must be non-negative"
    else:
        raise AssertionError("Expected negative drop_first_actions to raise ValueError")
