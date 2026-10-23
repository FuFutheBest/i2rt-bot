---
name: fastwam-video-visualization
description: Maintain optional FastWAM predicted-video transport and dashboard artifacts.
---

# FastWAM Video Visualization

Video prediction is optional and belongs to the model Server. The policy client
only requests it when a visualizer is active or `save_predictions` is enabled.
The response carries JPEG frames, the checkpoint camera-concatenation mode,
input frames after preprocessing, and the source observation timestamp.

Keep the visualizer's live camera SHM readers independent from policy inference.
Predicted-video generation may run an extra synchronous `infer_joint` diffusion
pass and therefore can add substantial response latency. Disable it for
latency-sensitive RTC profiles, or use a conservative request interval.

When saving is enabled, each request gets an isolated artifact directory with
the combined MP4, optional preprocessed camera PNGs, and JSON metadata. These
files are diagnostic artifacts, not a substitute for rollout recording. Check
the `fastwam-inference-runtime` profile before enabling video on a real robot.
Saved MP4 artifacts must use browser-compatible H.264/AVC with `yuv420p` and
fast-start metadata. Do not use OpenCV's `mp4v` writer: it creates MPEG-4 Part 2
streams that commonly fail in Chrome and Edge even when served as `video/mp4`.
The runtime therefore requires an FFmpeg build with the `libx264` encoder when
video artifact saving is requested.
