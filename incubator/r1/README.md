# R1 Incubator

R1 Quest3/Flexiv/YAM operator tooling is kept here because it requires
workstation hardware, Tk, ZED/GStreamer, DM3 upload services, and optional
vendor SDKs. The runtime device implementations remain in the core package:

- `incubator/r1/r1_teleoperator.py` provides Quest Vuer and replay producers.
- `incubator/r1/r1_rizon.py` provides the LK force-gated Rizon and YAM devices.
- `incubator/r1/async_data_saver_NEDF3_r1.py` provides the R1 NEDF3 recorder.

Run the R1 configs with `device_starter.py --config-path configs --config-name quest3_collect_r1` on
the target workstation after installing the optional dependencies and checking
the `third_party/quest_vuer` submodule.
