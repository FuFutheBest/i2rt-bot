from consumer.base import BaseConsumer


class _SummaryHandle:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.fail:
            raise OSError('injected close failure')


def test_disconnect_is_safe_before_base_init_completes() -> None:
    consumer = BaseConsumer.__new__(BaseConsumer)

    consumer.disconnect()
    consumer.__del__()

    assert getattr(consumer, 'summary_shm', None) is None


def test_disconnect_clears_handle_before_close_and_is_idempotent() -> None:
    consumer = BaseConsumer.__new__(BaseConsumer)
    handle = _SummaryHandle(fail=True)
    consumer.summary_shm = handle

    consumer.disconnect()
    consumer.disconnect()

    assert handle.close_calls == 1
    assert consumer.summary_shm is None
