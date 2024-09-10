"""B0 field maps generation routines."""

__all__ = ["b0field"]


from typing import Sequence


import numpy as np


from .._utils import fftc, ifftc


def b0field(
    chi: np.ndarray,
    b0range: Sequence[float] | None = None,
    mask: np.ndarray | None = None,
    B0: float = 1.5,
):
    """
    Simulate inhomogeneous B0 fields.

    Output field units is ``[Hz]``. The field
    is created by convolving the dipole kernel with an input
    magnetic susceptibility map.

    Parameters
    ----------
    chi : np.ndarray
        Object magnetic susceptibility map in ``[ppb]`` of
        shape ``(ny, nx)`` (2D) or ``(nz, ny, nx)`` (3D).
    b0range : Sequence[float] | None, optional
        Range of B0 field in ``[Hz]``. The default is ``None``
        (do not force a range).
    mask : np.ndarray, optional
        Region of support of the object of
        shape ``(ny, nx)`` (2D) or ``(nz, ny, nx)`` (3D).
        The default is ``None``.
    B0 : float, optional
        Static field strength in [T]. Used to compute
        B0 scaling (assuming 1H imaging)
        if `b0range` is not provided.
        The default is `1.5`.

    Returns
    -------
    B0map : np.ndarray
        Spatially varying B0 maps of shape ``(ny, nx)`` (2D)
        or ``(nz, ny, nx)`` (3D) in ``[Hz]``, arising from the object susceptibility.

    Example
    -------
    >>> from mrtwin import shepplogan_phantom, b0field

    We can generate a 2D B0 field map of shape ``(ny=128, nx=128)`` starting from a
    magnetic susceptibility distribution:

    >>> chi = shepplogan_phantom(2, 128, segtype=False).Chi
    >>> b0map = b0field(chi)

    B0 values range can be specified using ``b0range`` argument:

    >>> b0map = b0field(chi, b0range=(-500, 500))

    """
    # get input shape
    ishape = chi.shape

    # get number of spatial dims
    ndim = len(ishape)

    # get k space coordinates
    kgrid = [
        np.arange(-ishape[n] // 2, ishape[n] // 2, dtype=np.float32)
        for n in range(len(ishape))
    ]
    kgrid = np.meshgrid(*kgrid, indexing="ij")
    kgrid = np.stack(kgrid, axis=-1)

    knorm = (kgrid**2).sum(axis=-1) + np.finfo(np.float32).eps
    dipole_kernel = 1 / 3 - (kgrid[..., 0] ** 2 / knorm)

    # apply convolution
    B0map = ifftc(
        dipole_kernel * fftc(chi, ax=range(-ndim, 0)), ax=range(-ndim, 0)
    ).real

    # rescale
    if b0range is not None:
        B0map = B0map - B0map.min()  # (min, max) -> (0, max - min)
        B0map = B0map / B0map.max()  # (0, max - min) -> (0, 1)
        B0map = (
            B0map * (b0range[1] - b0range[0]) + b0range[0]
        )  # (0, 1) -> (b0range[0], b0range[1])
    else:
        gamma = 42.58 * 1e6  # Hz / T
        scale = gamma * B0
        B0map = scale * B0map

    # mask
    if mask is not None:
        mask = mask != 0
        B0map = mask * B0map

    return B0map.astype(np.float32)
