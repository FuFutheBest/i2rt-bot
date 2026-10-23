"""Optional NEDF3 recording, DM3 operator UI, and offline processors.

This package is deliberately outside the core runtime. Importing it does not
load tkinter, OpenCV, AV, or the vendor NEDF SDK; launch the concrete tools
from their module paths on a recording workstation.
"""

__all__ = []
