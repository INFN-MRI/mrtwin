"""OSF Phantom sub-package."""

__all__ = ["osf_phantom"]

from typing import Sequence
from .._utils import CacheDirType, PhantomType

from ._osf import NumericOSFPhantom


def osf_phantom(
    ndim: int,
    subject: int,
    shape: int | Sequence[int] = None,
    output_res: float | Sequence[float] = None,
    B0: float = 3.0,
    cache: bool = True,
    cache_dir: CacheDirType = None,
    osf_dir: CacheDirType = None,
    force: bool = False,
    verify: bool = True,
) -> PhantomType:
    """
    Get OSF phantom.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions. If ndim == 2, use a single slice
        (central axial slice).
    subject : int
        Subject id to download.
    shape: int | Sequence[int] | None, optional
        Shape of the output data, the data will be interpolated to the given shape.
        If int, assume isotropic matrix. The default is ``None`` (original shape).
    output_res: float | Sequence[float] | None, optional
        Resolution of the output data, the data will be rescale to the given resolution.
        If scalar, assume isotropic resolution. The default is ``None``
        (estimate from shape assuming same fov).
    B0 : float, optional
        Static field strength in [T].
        The default is `3.0`.
    cache : bool, optional
        If ``True``, cache the phantom. The default is ``True``.
    cache_dir : CacheDirType, optional
        cache_directory for phantom caching.
        The default is ``None`` (``~/.cache/mrtwin``).
    osf_dir : CacheDirType, optional
        osf_directory for brainweb segmentation caching.
        The default is ``None`` (``~/.cache/osf``).
    force : bool, optional
        Force download even if the file already exists.
        The default is ``False``.
    verify : bool, optional
        Enable SSL verification.
        DO NOT DISABLE (i.e., ``verify=False``) IN PRODUCTION.
        The default is ``True``.

    Returns
    -------
    PhantomType
        OSF phantom.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from mrtwin import osf_phantom

    We can generate a dense single slice 2D phantom with a matrix size
    of `(256, 256)` at `1.0625` mm isotropic resolution for the
    OSF subject `n=1` as:

    >>> phantom = osf_phantom(ndim=2, subject=1)

    Phantom T1 and T2 maps, can be accessed as:

    >>> fig, ax = plt.subplots(2, 1)

    >>> im1 = ax[0].imshow(phantom.T1, cmap="magma", vmin=0, vmax=3500)
    >>> ax[0].axis("off"), ax[0].set_title("T1 [ms]")
    >>> fig.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)

    >>> im2 = ax[1].imshow(phantom.T2, cmap="viridis", vmin=0, vmax=250)
    >>> ax[1].axis("off"), ax[1].set_title("T2 [ms]")
    >>> fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    """
    return NumericOSFPhantom(
        ndim,
        subject,
        shape,
        output_res,
        B0,
        cache,
        cache_dir,
        osf_dir,
        force,
        verify,
    )
