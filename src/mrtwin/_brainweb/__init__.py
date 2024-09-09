"""Brainweb Phantom sub-package."""

__all__ = ["brainweb_phantom"]

from typing import Sequence
from .._utils import CacheDirType, PhantomType

from ._brainweb import (
    NumericBrainwebPhantom,
    CrispBrainwebPhantom,
    FuzzyBrainwebPhantom,
)
from ._brainweb_mw import (
    NumericMWBrainwebPhantom,
    CrispMWBrainwebPhantom,
    FuzzyMWBrainwebPhantom,
)
from ._brainweb_mt import (
    NumericMTBrainwebPhantom,
    CrispMTBrainwebPhantom,
    FuzzyMTBrainwebPhantom,
)
from ._brainweb_mwmt import (
    NumericMWMTBrainwebPhantom,
    CrispMWMTBrainwebPhantom,
    FuzzyMWMTBrainwebPhantom,
)

VALID_MODELS = ["single-pool", "mt-model", "mw-model", "mwmt-model"]
VALID_SEGMENTATION = ["crisp", "fuzzy"]


def brainweb_phantom(
    ndim: int,
    subject: int,
    shape: int | Sequence[int] = None,
    model: str = "single-pool",
    segtype: str | bool = "crisp",
    output_res: float | Sequence[float] = None,
    B0: float = 1.5,
    cache: bool = True,
    cache_dir: CacheDirType = None,
    brainweb_dir: CacheDirType = None,
    force: bool = False,
    verify: bool = True,
) -> PhantomType:
    """
    Get BrainWeb phantom.

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
    model : str, optional
        String selecting one of the built-in
        tissue models. Valid entries are:

        * ``"single-pool"``: Single pool tissue model.
        * ``"mw-model"``: Myelin Water (MW) + Free Water (Intra-Extracellular, IEW)
        * ``"mt-model"``: Macromolecular pool + Free Water (IEW + MW)
        * ``"mwmt-model"``: Macromolecular pool + MW + IEW

        The default is ``"single-pool"``.
    segtype : str | bool, optional
        Phantom type. If it is a string (``"fuzzy"`` or ``"crisp"``)
        select fuzzy and crisp segmentation, respectively.
        If it is ``False``, return a dense numeric phantom.
        The default is ``crisp``.
    output_res: float | Sequence[float] | None, optional
        Resolution of the output data, the data will be rescaled to the given resolution.
        If scalar, assume isotropic resolution. The default is ``None``
        (estimate from shape assuming same fov).
    cache : bool, optional
        If ``True``, cache the phantom. The default is ``True``.
    cache_dir : CacheDirType, optional
        cache_directory for phantom caching.
        The default is ``None`` (``~/.cache/mrtwin``).
    brainweb_dir : CacheDirType, optional
        Brainweb_directory for brainweb segmentation caching.
        The default is ``None`` (``~/.cache/brainweb``).
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
        Brainweb phantom.

    """
    # check validity
    assert model in VALID_MODELS, ValueError(f"model must be one of {VALID_MODELS}")
    assert not (segtype) or segtype in VALID_SEGMENTATION, ValueError(
        f"segtype must be either False or one of {VALID_SEGMENTATION}"
    )

    # initialize model
    params = {
        "ndim": ndim,
        "subject": subject,
        "shape": shape,
        "output_res": output_res,
        "B0": B0,
        "cache": cache,
        "cache_dir": cache_dir,
        "brainweb_dir": brainweb_dir,
        "force": force,
        "verify": verify,
    }
    if model == "single-pool":
        if segtype == "fuzzy":
            return FuzzyBrainwebPhantom(**params)
        if segtype == "crisp":
            return CrispBrainwebPhantom(**params)
        if segtype is False:
            return NumericBrainwebPhantom(**params)
    if model == "mw-model":
        if segtype == "fuzzy":
            return FuzzyMWBrainwebPhantom(**params)
        if segtype == "crisp":
            return CrispMWBrainwebPhantom(**params)
        if segtype is False:
            return NumericMWBrainwebPhantom(**params)
    if model == "mt-model":
        if segtype == "fuzzy":
            return FuzzyMTBrainwebPhantom(**params)
        if segtype == "crisp":
            return CrispMTBrainwebPhantom(**params)
        if segtype is False:
            return NumericMTBrainwebPhantom(**params)
    if model == "mwmt-model":
        if segtype == "fuzzy":
            return FuzzyMWMTBrainwebPhantom(**params)
        if segtype == "crisp":
            return CrispMWMTBrainwebPhantom(**params)
        if segtype is False:
            return NumericMWMTBrainwebPhantom(**params)
