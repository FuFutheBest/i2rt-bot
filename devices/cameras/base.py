#!/usr/bin/env python3
"""Shared camera adapter foundation.

Author: OpenAI Assistant
"""

from devices.base import BaseDevice
from devices.contracts import DeviceKind


class BaseCameraDevice(BaseDevice):
    """Base class for cameras with the common RGB state contract.

    Camera subclasses must declare ``state_feedback`` in their public
    constructor. ``StateFeedbackConfig`` validates ``[H, W, 3]``, ``uint8``
    (or another supported SHM dtype), and ``type: rgb`` before capture starts.
    """

    device_kind = DeviceKind.CAMERA
