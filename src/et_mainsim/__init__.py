"""ET reference workflow application.

The package root intentionally avoids importing simulation backends or assets.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("et-mainsim")
except PackageNotFoundError:
    __version__ = "0.1.0"


__all__ = ["__version__"]
