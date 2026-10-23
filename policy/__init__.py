#!/usr/bin/env python3
"""
Policy package - Pluggable policies for action inference.

Author: Jun Lv, Zheng Wang
"""

from .base import BasePolicy
from .tiny_mlp import TinyMLPPolicy
from .robotwin_act_policy import RoboTwinACTPolicy
from .robotwin_replay_policy import RoboTwinReplayPolicy
from .robotwin_xpolicylab_policy import RoboTwinXPolicyLabPolicy
try:
    from .openpi_policy import OpenPiPolicy
except ImportError:
    OpenPiPolicy = None
try:
    from .diffusion_policy import DiffusionPolicy
except ImportError:
    DiffusionPolicy = None
try:
    from .openpi_policy import OpenPiPolicy
except ImportError:
    OpenPiPolicy = None
try:
    from .openpi_history_policy import OpenPiHistoryPolicy
except ImportError:
    OpenPiHistoryPolicy = None
try:
    from .diffusion_policy_iphone_umi import DiffusionPolicyIPhoneUMI
except Exception:
    DiffusionPolicyIPhoneUMI = None
try:
    from .openpi_aloha_policy import OpenPiAlohaPolicy
except ImportError:
    OpenPiAlohaPolicy = None
try:
    from .openpi_aloha_history_policy import OpenPiAlohaHistoryPolicy
except ImportError:
    OpenPiAlohaHistoryPolicy = None
try:
    from .nmx_vla_policy import NMXVLAPolicy
except Exception:
    NMXVLAPolicy = None
try:
    from .lingbot_va_policy import LingBotVAPolicy
except Exception:
    LingBotVAPolicy = None
try:
    from .openpi_umi_policy import OpenPiUmiPolicy
except ImportError:
    OpenPiUmiPolicy = None
try:
    from .openpi_umi_history_policy import OpenPiUmiHistoryPolicy
except ImportError:
    OpenPiUmiHistoryPolicy = None
try:
    # FastWAM's websocket client is optional for lightweight/device-only runs.
    from .fastwam_policy import FastWAMPolicy
except Exception:
    FastWAMPolicy = None

# Policy class mapping - add new policy classes here
POLICY_CLASSES = {
    'TinyMLPPolicy': TinyMLPPolicy,
    'DiffusionPolicy': DiffusionPolicy,
    'DiffusionPolicyIPhoneUMI': DiffusionPolicyIPhoneUMI,
    "OpenPiPolicy": OpenPiPolicy,
    "OpenPiAlohaPolicy": OpenPiAlohaPolicy,
    "OpenPiAlohaHistoryPolicy": OpenPiAlohaHistoryPolicy,
    "OpenPiHistoryPolicy": OpenPiHistoryPolicy,
    "NMXVLAPolicy": NMXVLAPolicy,
    "LingBotVAPolicy": LingBotVAPolicy,
    "OpenPiUmiPolicy": OpenPiUmiPolicy,
    "OpenPiUmiHistoryPolicy": OpenPiUmiHistoryPolicy,
    "FastWAMPolicy": FastWAMPolicy,
    "RoboTwinACTPolicy": RoboTwinACTPolicy,
    "RoboTwinReplayPolicy": RoboTwinReplayPolicy,
    "RoboTwinXPolicyLabPolicy": RoboTwinXPolicyLabPolicy,
}

__all__ = [
    'BasePolicy',
    'TinyMLPPolicy',
    'DiffusionPolicy',
    'DiffusionPolicyIPhoneUMI',
    'POLICY_CLASSES',
    'OpenPiPolicy', 'OpenPiAlohaPolicy', 'OpenPiAlohaHistoryPolicy', 'OpenPiHistoryPolicy',
    'NMXVLAPolicy',
    'LingBotVAPolicy',
    'OpenPiUmiPolicy',
    'OpenPiUmiHistoryPolicy',
    'FastWAMPolicy',
    'RoboTwinACTPolicy',
    'RoboTwinReplayPolicy',
    'RoboTwinXPolicyLabPolicy',
]
