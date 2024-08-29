"""Custom typing for type hints."""

__all__ = ["CacheDirType"]

from pathlib import Path
from typing import Union

CacheDirType = Union[Path, None]