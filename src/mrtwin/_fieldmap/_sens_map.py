"""Coil sensitivity maps generation routines."""

__all__ = ["sensmap"]


import os


from typing import Sequence


import numpy as np


from .._utils import CacheDirType, get_mrtwin_dir


from ._birdcage import _birdcage


def sensmap(
    shape: Sequence[int],
    coil_width: float = 2.0,
    shift: Sequence[int] | None = None,
    dphi: float = 0.0,
    nrings: int = None,
    cache: bool | None = None,
    cache_dir: CacheDirType = None,
):
    """
    Simulate birdcage coils.

    Adapted from SigPy [1].

    Parameters
    ----------
    shape : Iterable[int]
        Size of the matrix ``(ncoils, ny, nx)`` (2D)
        or ``(ncoils, nz, ny, nx)`` (3D) for the sensitivity coils.
    coil_width : float, optional
        Width of the coil, with respect to image dimension.
        The default is ``2.0``.
    shift : Sequence[int] | None, optional
        Displacement of the coil center with respect to matrix center.
        The default is ``(0, 0)`` / ``(0, 0, 0)``.
    dphi : float, optional
        Bulk coil angle in ``[deg]``.
        The default is ``0.0°``.
    nrings : int | None, optional
        Number of rings for a cylindrical hardware set-up.
        The default is ``ncoils // 4``.
    cache : bool | None, optional
        If ``True``, cache the phantom.
        The default is ``True`` for 3D phantoms
        and ``False`` for single-slice 2D.
    cache_dir : CacheDirType, optional
        cache_directory for phantom caching.
        The default is ``None`` (``~/.cache/mrtwin``).

    Returns
    -------
    smap : torch.Tensor
        Complex spatially varying sensitivity maps of shape ``(nmodes, ny, nx)`` (2D)
        or ``(nmodes, nz, ny, nx)`` (3D). If ``nmodes = 1``, the first dimension is squeezed.

    Example
    -------
    >>> from mrtwin import sensmap

    We can generate a set of ``nchannels=8`` 2D sensitivity maps of shape ``(ny=128, nx=128)`` by:

    >>> smap = sensmap((8, 128, 128))

    Coil center and rotation can be modified by ``shift`` and ``dphi`` arguments:

    >>> smap = sensmap((8, 128, 128), shift=(-3, 5), dphi=30.0) # center shifted by (dy, dx) = (-3, 5) pixels and rotated by 30.0 degrees.

    Similarly, ``nchannels=8`` 3D sensitivity maps can be generated as:

    >>> smap = sensmap((8, 128, 128, 128))

    Beware that this will require more memory.

    References
    ----------
    [1] https://github.com/mikgroup/sigpy/tree/main

    """
    # Default values
    if shift is None:
        shift = [0.0 for ax in range(len(shape) - 1)]
    if nrings is None:
        nrings = np.max((shape[0] // 4, 1))
    if cache is None and len(shape) == 3:  # (nc, ny, nx) -> 2D
        cache = False
    elif cache is None and len(shape) == 4:  # (nc, nz, ny, nx) -> 3D
        cache = True

    # Get filename for caching
    shape_str = [str(di) for di in shape]
    shape_str = "x".join(tuple(shape_str))

    shift_str = [str(di) for di in shift]
    shift_str = "x".join(tuple(shift_str))

    file_name = f"sensmap{shape_str}mtx_{coil_width}width_{shift_str}shift_{nrings}rings_{dphi}deg.npy"

    # Get base directory
    cache_dir = get_mrtwin_dir(cache_dir)

    # Get file path
    file_path = os.path.join(cache_dir, file_name)

    # Try to load
    if os.path.exists(file_path):
        return np.load(file_path)

    # Generate map
    smap = _birdcage(shape, coil_width, nrings, shift, np.deg2rad(dphi))

    # Normalize
    rss = sum(abs(smap) ** 2, 0) ** 0.5
    smap /= rss

    # Cache the result
    if cache and os.path.exists(file_path) is False:
        np.save(file_path, smap)

    return smap
