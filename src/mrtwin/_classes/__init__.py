"""Tissue model for different experiments."""

__all__ = ["tissue_map"]

import os
import sys

import numpy as np

if sys.version_info > (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files

from pathlib import Path

from brainweb_dl._brainweb import _load_tissue_map

BUILT_IN_MAPS = ["single-pool", "mt-model", "mw-model", "mwmt-model"]


class TissueMap:
    """Tissue map files for the brainweb dataset."""

    single: Path = files("mrtwin._classes") / "single_pool.csv"  # type: ignore
    mt: Path = files("mrtwin._classes") / "mt_model.csv"  # type: ignore
    mw: Path = files("mrtwin._classes") / "mw_model.csv"  # type: ignore
    mwmt: Path = files("mrtwin._classes") / "mwmt_model.csv"  # type: ignore


def tissue_map(path_or_dict: str | os.PathLike | dict) -> list[dict]:
    """
    Build dictionary of tissue properties.

    Parameters
    ----------
    path_or_dict : str | os.PathLike | dict
        This can be either a list of dictionaries
        each describing a tissue class (e.g., WM, GM),
        a path to a cvs file storing the tissue classes
        descriptions or a string selecting one of the built-in
        tissue models. For the latter case, valid entries are:

            * "single-pool": Single pool tissue model.
            * "mw-model": Myelin Water (MW) + Free Water (Intra-Extracellular, IEW)
            * "mt-model": Macromolecular pool + Free Water (IEW + MW)
            * "mwmt-model": Macromolecular pool + MW + IEW

    Notes
    -----
    Each configuration file  field / dict must at least contain
    "Tissue Type", "ID" and "Label" fields. The other can be
    designed by the user to accomodate different properties (e.g., Diffusion).

    Returns
    -------
    list[dict]
        List of dictionaries each describing a tissue class.

    """
    # parse properties
    if isinstance(path_or_dict, str) and path_or_dict in BUILT_IN_MAPS:
        if path_or_dict == "single-pool":
            tissue_dict = _load_tissue_map(TissueMap.single)
        if path_or_dict == "mt-model":
            tissue_dict = _load_tissue_map(TissueMap.mw)
        if path_or_dict == "mw-model":
            tissue_dict = _load_tissue_map(TissueMap.mt)
        if path_or_dict == "mwmt-model":
            tissue_dict = _load_tissue_map(TissueMap.mwmt)
    else:
        if isinstance(path_or_dict, list):
            tissue_dict = path_or_dict
        else:
            tissue_dict = _load_tissue_map(path_or_dict)

    # check validity of dictionary
    assert (
        "Tissue Type" in tissue_dict and "ID" in tissue_dict and "Label" in tissue_dict
    ), KeyError("Tissue Type, ID and Label fields must be defined.")

    # iterate and cast string to float / int
    for item in tissue_dict:
        item["ID"] = int(item["Label"])
        for key in item.keys() - {"Tissue Type", "ID", "Label"}:
            item[key] = float(item[key])
            if item[key] == -1:
                item[key] = np.nan

    return tissue_dict


def get_t1(A: float, C: float, B0: float) -> float:
    """
    Calculate T1 for a given tissue at a specific field strength.

    Parameters
    ----------
    A : float
        Base value in [ms].
    C : float
        Growth order.
    B0 : float
        Static field strength in [T].

    Returns
    -------
    float
        T1 value in [ms].

    """
    return A * (B0**C)


def get_t2star(T2: float, Chi: float, B0: float) -> float:
    """
    Calculate T2* for a given tissue at a specific field strength.

    Parameters
    ----------
    T2 : float
        Transverse relaxation time in [ms].
    Chi : float
        Magnetic susceptibility.
    B0 : float
        Static field strength in [T].

    Returns
    -------
    float
        T2* value in [ms].

    """
    gamma0 = 267.52219  # 10^6 rad⋅s−1⋅T⋅−1
    return 1 / (1 / T2 + gamma0 * np.abs(B0 * Chi))


def extrapolate_t1(T1start: float, B0start: float, B0end: float) -> float:
    """
    Extrapolate T1 values at a given field strength.

    Parameters
    ----------
    T1start : float
        Initial T1 in [ms].
    B0start : float
        Initial field strength in [T].
    B0end : float
        Desired field strength in [T].

    Returns
    -------
    float
        Final T1 in [ms].

    """
    scale = (B0end / B0start) ** 0.5
    return scale * T1start


def extrapolate_t2star(
    T2s_start: float, T2: float, B0start: float, B0end: float
) -> float:
    """
    Extrapolate T2* values at a given field strength.

    Parameters
    ----------
    T2s_start : float
        Initial T2* in [ms].
    T2 : float
        Transverse relaxation in [ms].
    B0start : float
        Initial field strength in [T].
    B0end : float
        Desired field strength in [T].

    Returns
    -------
    float
        Final T2* in [ms].

    """
    scale = B0end / B0start
    R2p_start = 1 / T2s_start - 1 / T2
    R2p_end = scale * R2p_start
    return 1 / (R2p_end + 1 / T2)
