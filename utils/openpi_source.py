"""Select one local OpenPi source tree before importing its shared package name."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Optional


def _activate_package(package_name: str, source_path: Path, *, index: int) -> None:
    package_init = source_path / package_name / "__init__.py"
    if not package_init.is_file():
        raise FileNotFoundError(
            f"Configured source does not contain {package_name}/__init__.py: {source_path}"
        )

    loaded_package = sys.modules.get(package_name)
    if loaded_package is not None:
        loaded_file = getattr(loaded_package, "__file__", None)
        loaded_path = Path(loaded_file).resolve() if loaded_file else None
        if loaded_path is None or source_path not in loaded_path.parents:
            raise RuntimeError(
                f"A different {package_name} package was imported before its configured "
                f"source could be activated: loaded={loaded_file}, requested={source_path}"
            )
        return

    source_str = str(source_path)
    if source_str in sys.path:
        sys.path.remove(source_str)
    sys.path.insert(index, source_str)

    spec = importlib.util.find_spec(package_name)
    resolved_origin = Path(spec.origin).resolve() if spec is not None and spec.origin else None
    if resolved_origin is None or source_path not in resolved_origin.parents:
        raise ImportError(
            f"Could not resolve {package_name} from configured source {source_path}; "
            f"resolved origin={resolved_origin}"
        )


def activate_openpi_source(
    source: Optional[str], *, repo_root: Optional[Path] = None
) -> Optional[Path]:
    """Prepend an OpenPi ``src`` directory and verify that it wins resolution.

    Multiple OpenPi checkouts in this repository expose the same top-level
    ``openpi`` package. This function must therefore run before the first OpenPi
    import. Existing callers that pass ``None`` retain normal environment-based
    package resolution.
    """
    if not source:
        return None

    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        if repo_root is None:
            raise ValueError("repo_root is required when source is relative")
        source_path = repo_root / source_path
    source_path = source_path.resolve()
    _activate_package("openpi", source_path, index=0)

    # A source checkout keeps the runtime client in a sibling workspace package.
    # Pin it too when present so core OpenPi and openpi_client cannot come from
    # different forks/installations.
    checkout_root = source_path.parent
    client_source = checkout_root / "packages" / "openpi-client" / "src"
    if client_source.is_dir():
        _activate_package("openpi_client", client_source.resolve(), index=1)
    return source_path
