"""Default path handling."""

__all__ = ["get_mrtwin_dir"]

import os

from pathlib import Path

from ._typing import CacheDirType

# Directory where data will be stored
def get_mrtwin_dir(cache_dir: CacheDirType = None) -> Path:
    """Get the MRTWIN directory.

    Adapted from brainweb_dl._brainweb.get_brainweb_dir

    Parameters
    ----------
    osf_dir : os.PathLike
       osf_directory to download the data.

    Returns
    -------
    os.PathLike
        Path to osf_dir

    Notes
    -----
    The osf directory is set in the following order:
    - The cache_dir passed as argument.
    - The environment variable MRTWIN_DIR.
    - The default osf_directory ~/.cache/cache_dir.
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
    elif "MRTWIN_DIR" in os.environ:
        cache_dir = Path(os.environ["MRTWIN_DIR"])
    else:
        cache_dir = Path.home() / ".cache" / "mrtwin"
    os.makedirs(Path(cache_dir), exist_ok=True)
    return Path(cache_dir)