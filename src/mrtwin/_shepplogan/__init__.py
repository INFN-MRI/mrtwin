"""Shepp-Logan Phantom sub-package."""

__all__ = ["shepplogan_phantom"]

from typing import Sequence
from .._utils import CacheDirType, PhantomType

from ._shepplogan import NumericSheppLoganPhantom, CrispSheppLoganPhantom
from ._shepplogan_mw import NumericMWSheppLoganPhantom, CrispMWSheppLoganPhantom
from ._shepplogan_mt import NumericMTSheppLoganPhantom, CrispMTSheppLoganPhantom
from ._shepplogan_mwmt import NumericMWMTSheppLoganPhantom, CrispMWMTSheppLoganPhantom

VALID_MODELS = ["single-pool", "mt-model", "mw-model", "mwmt-model"]
VALID_SEGMENTATION = ["crisp"]


def shepplogan_phantom(
    ndim: int,
    shape: int | Sequence[int],
    model: str = "single-pool",
    segtype: str | bool = "crisp",
    B0: float = 1.5,
    cache: bool = None,
    cache_dir: CacheDirType = None,
) -> PhantomType:
    """
    Get SheppLogan phantom.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions. If ndim == 2, use a single slice
        (central axial slice).
    shape: int | Sequence[int]
        Shape of the output data. If int, assume isotropic matrix.
    model : str, optional
        String selecting one of the built-in
        tissue models. Valid entries are:

        * ``"single-pool"``: Single pool tissue model.
        * ``"mw-model"``: Myelin Water (MW) + Free Water (Intra-Extracellular, IEW)
        * ``"mt-model"``: Macromolecular pool + Free Water (IEW + MW)
        * ``"mwmt-model"``: Macromolecular pool + MW + IEW

        The default is ``"single-pool"``.
    segtype : str | bool, optional
        Phantom type. If it is a string (``"crisp"``)
        select crisp segmentation.
        If it is ``False``, return a dense numeric phantom.
        The default is ``crisp``.
    cache : bool | None, optional
        If ``True``, cache the phantom.
        The default is ``True`` for 3D phantoms
        and ``False`` for single-slice 2D.
    cache_dir : CacheDirType, optional
        cache_directory for phantom caching.
        The default is ``None`` (``~/.cache/mrtwin``).

    Returns
    -------
    PhantomType
        Shepp-Logan phantom.
        
    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from mrtwin import shepplogan_phantom
    
    We can generate a dense single slice 2D phantom with a matrix size
    of `(200, 200)` as:
    
    >>> phantom = shepplogan_phantom(ndim=2, shape=200, segtype=False)
    
    Phantom T1 and T2 maps, can be accessed as:
    
    >>> fig, ax = plt.subplots(2, 1)

    >>> im1 = ax[0].imshow(phantom.T1, cmap="magma", vmax=1500)
    >>> ax[0].axis("off"), ax[0].set_title("T1 [ms]")
    >>> fig.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)

    >>> im2 = ax[1].imshow(phantom.T2, cmap="viridis", vmax=150)
    >>> ax[1].axis("off"), ax[1].set_title("T2 [ms]")
    >>> fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    """
    # check validity
    assert model in VALID_MODELS, ValueError(f"model must be one of {VALID_MODELS}")
    assert not (segtype) or segtype in VALID_SEGMENTATION, ValueError(
        f"segtype must be either False or one of {VALID_SEGMENTATION}"
    )

    # initialize model
    params = {
        "ndim": ndim,
        "shape": shape,
        "B0": B0,
        "cache": cache,
        "cache_dir": cache_dir,
    }
    if model == "single-pool":
        if segtype == "crisp":
            return CrispSheppLoganPhantom(**params)
        if segtype is False:
            return NumericSheppLoganPhantom(**params)
    if model == "mw-model":
        if segtype == "crisp":
            return CrispMWSheppLoganPhantom(**params)
        if segtype is False:
            return NumericMWSheppLoganPhantom(**params)
    if model == "mt-model":
        if segtype == "crisp":
            return CrispMTSheppLoganPhantom(**params)
        if segtype is False:
            return NumericMTSheppLoganPhantom(**params)
    if model == "mwmt-model":
        if segtype == "crisp":
            return CrispMWMTSheppLoganPhantom(**params)
        if segtype is False:
            return NumericMWMTSheppLoganPhantom(**params)
