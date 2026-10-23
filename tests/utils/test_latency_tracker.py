from utils.latency_tracker import LatencyTracker


def test_compute_inference_delay_steps_uses_p95_and_execution_rate():
    tracker = LatencyTracker(maxlen=8)
    for latency in (0.080, 0.100, 0.120, 0.200):
        tracker.add(latency)

    # p95 is below 0.2s; at 30Hz that still occupies six control steps.
    assert tracker.compute_inference_delay_steps(execution_fps=30.0) == 6


def test_compute_inference_delay_steps_uses_default_without_samples():
    tracker = LatencyTracker()

    assert tracker.compute_inference_delay_steps(
        execution_fps=30.0,
        default_delay_steps=4,
    ) == 4
