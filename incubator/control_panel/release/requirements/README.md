# Release dependency profiles

These files are owned by the control-panel release builder. Run install commands
from the repository root so editable `third_party/` paths resolve consistently.

- `main_py312.txt` is the device/control-panel overlay applied after OpenPi's
  frozen `uv.lock` by `../create_source_py312_env.sh`. It creates one shared
  Python 3.12 source profile for the web panel, devices, OpenPi policy, LKTech,
  i2rt, NEDF recording, and conversion. The profile pins the CUDA 12.8
  compiler used by JAX so it never falls back to an older host `ptxas`.
- `build_constraints.txt` pins build-isolation tools.
- `policy.txt` defines the legacy split OpenPi Python 3.12 profile with the
  same Torch cu128 and JAX CUDA 12 contract as the unified profile.
- `recording.txt` pins the NEDF3 dependencies merged into the
  shared Python 3.12 profile, including the three private SDK distributions.
- `flexiv_rdk_*.txt` are explicit firmware-compatible RDK choices for source
  environments. The binary release bundles all reviewed choices.

`policy.txt` and the root `requirements_device.txt` remain available for
legacy split-profile source environments. The unified source builder requires
the official cp312 Linux wheelhouse as its second argument or via
`NEDF_WHEELHOUSE`.

There are intentionally no Elements-version aliases. Elements UI version does
not determine the RDK version accepted by robot firmware.
