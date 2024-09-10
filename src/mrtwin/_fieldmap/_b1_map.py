"""B1+ field maps generation routines."""

__all__ = ["b1field"]


import math
import os


from typing import Sequence


import numpy as np


from .._utils import CacheDirType, get_mrtwin_dir


from ._birdcage import _birdcage


def b1field(
    shape: Sequence[int],
    nmodes: int = 1,
    b1range: Sequence[float] = (0.5, 2.0),
    shift: Sequence[int] | None = None,
    dphi: float = 0.0,
    coil_width: float = 1.1,
    ncoils: int = 2,
    nrings: int | None = None,
    mask: np.ndarray | None = None,
    cache: bool | None = None,
    cache_dir: CacheDirType = None,
):
    """
    Simulate inhomogeneous B1+ fields.

    Adapted from SigPy [1].

    Parameters
    ----------
    shape : Sequence[int]
        Size of the matrix ``(ny, nx)`` (2D) or
        ``(nz, ny, nx)`` (3D) for the B1+ field.
    nmodes : int, optional
        Number of B1+ modes. First mode is ``CP`` mode, second
        is ``gradient`` mode, and so on. The default is ``1``.
    b1range : Sequence[float], optional
        Range of B1+ magnitude. The default is ``(0.5, 2.0)``.
    shift : Sequence[int], optional
        Displacement of the coil center with respect to matrix center.
        The default is ``(0, 0)`` / ``(0, 0, 0)``.
    dphi : float, optional
        Bulk coil angle in ``[deg]``.
        The default is ``0.0°``.
    coil_width : float, optional
        Width of the coil, with respect to image dimension.
        The default is ``1.1``.
    ncoils : int, optional
        Number of transmit coil elements. Standard coils have ``2`` channels
        operating in quadrature. To support multiple modes (e.g., PTX), increase this
        number. The default is ``4``.
    nrings : int, optional
        Number of rings for a cylindrical hardware set-up.
        The default is ``ncoils // 4``.
    mask : np.ndarray | None, optional
        Region of support of the object of
        shape ``(ny, nx)`` (2D) or ``(nz, ny, nx)`` (3D).
        The default is ``None``.
    cache : bool | None, optional
        If ``True``, cache the phantom.
        The default is ``True`` for 3D phantoms
        and ``False`` for single-slice 2D.
    cache_dir : CacheDirType, optional
        cache_directory for phantom caching.
        The default is ``None`` (``~/.cache/mrtwin``).

    Returns
    -------
    smap : np.ndarray
        Complex spatially varying b1+ maps of shape ``(nmodes, ny, nx)`` (2D)
        or ``(nmodes, nz, ny, nx)`` (3D). Magnitude of the map represents
        the relative flip angle scaling (wrt to the nominal).

    Example
    -------
    >>> from mrtwin import b1field

    We can generate a 2D B1+ field map of shape ``(ny=128, nx=128)`` by:

    >>> b1map = b1field((128, 128))

    Field center and rotation can be modified by ``shift`` and ``dphi`` arguments:

    >>> b1map = b1field((128, 128), shift=(-3, 5), dphi=30.0) # center shifted by (dy, dx) = (-3, 5) pixels and rotated by 30.0 degrees.

    B1+ values range and steepness of variation can be specified using ``b1range`` and ``coil_width`` arguments:

    >>> # transmit coil is 4 times bigger than FOV (e.g., body coil) and
    >>> # B1+ scalings are between (0.8, 1.2) the nominal flip angle (e.g., 3T scanner)
    >>> b1map3T = b1field((128, 128), b1range=(0.8, 1.2), coil_width=4.0)
    >>>
    >>> # transmit coil is 1.1 times bigger than FOV (e.g., head coil) and
    >>> # B1+ scalings are between (0.5, 2.0) the nominal flip angle (e.g., 7T scanner)
    >>> b1map7T = b1field((128, 128), b1range=(0.5, 2.0), coil_width=1.1)

    Multiple orthogonal modes can be simulated by ``nmodes`` argument.
    For example, ``CP`` mode and ``gradient`` mode can be obtained as:

    >>> b1map = b1field((128, 128), nmodes=2) # b1map[0] is CP, b1map[1] is gradient mode.

    Three dimensional B1+ maps of shape ``(nz, ny, nx)`` can be obtained as:

    >>> b1map = b1field((128, 128, 128))

    Beware that this will require more memory.

    References
    ----------
    [1] https://github.com/mikgroup/sigpy/tree/main

    """
    ncoils *= 2

    # check we can do quadrature
    assert (
        ncoils >= 2
    ), f"We support circular polarization only - found {ncoils} transmit elements."
    assert ncoils >= nmodes, f"Need ncoils (={ncoils}) to be >= nmodes (={nmodes})."

    # Default values
    if shift is None:
        shift = [0.0 for ax in range(len(shape))]
    if nrings is None:
        nrings = np.max((shape[0] // 4, 1))
    if cache is None and len(shape) == 2:  # (ny, nx) -> 2D
        cache = False
    elif cache is None and len(shape) == 3:  # (nz, ny, nx) -> 3D
        cache = True

    # Get filename for caching
    shape_str = [str(di) for di in shape]
    shape_str = "x".join(tuple(shape_str))

    shift_str = [str(di) for di in shift]
    shift_str = "x".join(tuple(shift_str))

    b1range_str = [str(value) for value in b1range]
    b1range_str = "-".join(tuple(b1range_str))

    if mask is None:
        is_masked = False
    else:
        is_masked = True

    file_name = f"b1map{shape_str}mtx_{nmodes}modes_{b1range_str}range_{coil_width}width_{ncoils}coils_{shift_str}shift_{nrings}rings_{dphi}deg_{is_masked}.npy"

    # Get base directory
    cache_dir = get_mrtwin_dir(cache_dir)

    # Get file path
    file_path = os.path.join(cache_dir, file_name)

    # Try to load
    if os.path.exists(file_path):
        return np.load(file_path)

    # Generate coils
    smap = _birdcage(
        [ncoils] + list(shape), coil_width, nrings, shift, np.deg2rad(dphi)
    )

    # Normalize
    rss = sum(abs(smap) ** 2, 0) ** 0.5
    smap /= rss

    # Combine
    dalpha = 2 * math.pi / ncoils
    alpha = np.arange(ncoils) * dalpha
    mode = np.arange(nmodes)
    phafu = np.exp(1j * mode[:, None] * alpha[None, :])  # (nmodes, nchannels)

    # Get modes
    smap = smap.T  # (nc, ...) -> (..., nc)
    smap = [(abs(smap) * phafu[n]).sum(axis=-1) for n in range(nmodes)]
    smap = np.stack(smap, axis=-1)  # (..., nmodes)
    smap = smap.T  # (..., nmodes) -> (nmodes, ...)

    # Rescale
    phase = smap / abs(smap)
    smap = abs(smap)
    smap = smap - smap.min()  # (min, max) -> (0, max - min)
    smap = smap / smap.max()  # (0, max - min) -> (0, 1)
    smap = (
        smap * (b1range[1] - b1range[0]) + b1range[0]
    )  # (0, 1) -> (b1range[0], b1range[1])
    smap = smap * phase

    if nmodes == 1:
        smap = abs(smap[0]).astype(np.float32)
    else:
        smap = smap.astype(np.complex64)

    # Mask
    if mask is not None:
        smap = mask * smap

    # Cache the result
    if cache and os.path.exists(file_path) is False:
        np.save(file_path, smap)

    return smap
