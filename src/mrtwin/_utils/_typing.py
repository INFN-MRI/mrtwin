"""Custom typing for type hints."""

__all__ = ["CacheDirType", "PhantomType"]

from pathlib import Path
from typing import Any, Union

CacheDirType = Union[Path, None]
PhantomType = Any # TODO: find something more useful