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
    B0: float = 1.5,
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
    shape: int | Sequence[int | None, optional
        Shape of the output data, the data will be interpolated to the given shape.
        If int, assume isotropic matrix. The default is ``None`` (original shape).
    output_res: float | Sequence[float] | None, optional
        Resolution of the output data, the data will be rescale to the given resolution.
        If scalar, assume isotropic resolution. The default is ``None``
        (estimate from shape assuming same fov).
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
