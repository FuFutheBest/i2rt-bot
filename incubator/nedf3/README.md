# NEDF3 Incubator

This directory contains the hardware- and workstation-specific NEDF3 tools
carried over from the `NEDF3` branch:

- `Vive_DM3/` is the Windows DM3 collection and upload application.
- `I2RT_DM3/` is the Linux I2RT application with login, collection, and upload
  tabs. It intentionally omits the Vive calibration and diagnostic workflow.
- `windows_DM3/` contains its platform support modules.
- `nedf3_processer/` contains offline alignment and gripper-width processing.

The stable topic and metadata contract lives in
`consumer/io/nedf3_contract.py`; these tools must consume that contract rather
than duplicating topic definitions in core runtime modules.

Start the I2RT application from the repository root:

```bash
.venv/bin/python -m incubator.nedf3.I2RT_DM3.gui
```

The browser interface uses the same controller and process ownership layer:

```bash
.venv/bin/python -m incubator.nedf3.I2RT_DM3.web
```

Open `http://127.0.0.1:8080`. The server binds to loopback by default and
requires `--allow-remote` when a non-loopback `--host` is requested. Closing a
browser tab does not stop collection; stopping the server performs the managed
saver and device shutdown sequence.

The I2RT upload controller requests its STS token with an empty
`resource_type`. The existing Vive application retains its configured resource
type.
