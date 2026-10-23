"""Shared-memory-native web control panel for SuperInference.

The package initializer intentionally avoids importing hardware registries or
Flask.  Import a focused submodule without pulling in the whole application.
"""

from typing import Any


def create_app(*args: Any, **kwargs: Any) -> Any:
    """Lazily construct the Flask app."""
    from .server import create_app as factory

    return factory(*args, **kwargs)


__all__ = ['create_app']
