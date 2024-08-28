"""Wrapper around phantominator to extract MR tissue segmentation."""

__all__ = []

import importlib

from typing import Dict

import numpy as np
import numpy.typing as npt

mr_shepp_logan = importlib.import_module('phantominator.mr_shepp_logan')

def _mr_relaxation_parameters() -> Dict[str, npt.ArrayLike]:
    """Return MR relaxation parameters for certain tissues.

    Returns
    -------
    params : dict
        Gives entries as [A, C, (t1), t2, chi]

    Notes
    -----
    If t1 is None, the model T1 = A*B0^C will be used.  If t1 is not
    np.nan, then specified t1 will be used.
    """
    # params['tissue-name'] = [A, C, (t1 value if explicit), t2, chi]
    params = dict()
    params['scalp'] = [np.nan, np.nan, 9, .07, -7.5e-6]
    params['marrow'] = [np.nan, np.nan, 11, .05, -8.85e-6]
    params['csf'] = [np.nan, np.nan, 1, 1.99, -9e-6]
    params['blood-clot'] = [np.nan, np.nan, 8, .2, -9e-6]
    params['gray-matter'] = [np.nan, np.nan, 2, .1, -9e-6]
    params['white-matter'] = [np.nan, np.nan, 3, .08, -9e-6]
    params['tumor'] = [np.nan, np.nan, 8, .1, -9e-6]
    return params


def mr_ellipsoid_parameters() -> npt.ArrayLike:
    """Return parameters of ellipsoids.

    Returns
    -------
    E : array_like
        Parameters for the ellipsoids used to construct the phantom.
    """
    params = _mr_relaxation_parameters()

    E = np.zeros((15, 13))
    # [:, [x, y, z, a, b, c, theta, m0, A, C, (t1), t2, chi]]
    E[0, :] = [
        0, 0, 0, .72, .95, .93, 0, .8,
        *params['scalp']]
    E[1, :] = [
        0, 0, 0, .69, .92, .9, 0, .12,
        *params['marrow']]
    E[2, :] = [
        0, -.0184, 0, .6624, .874, .88, 0, .98,
        *params['gray-matter']]
    E[3, :] = [
        0, -.0184, 0, .6024, .814, .82, 0, .745,
        *params['white-matter']]
    E[4, :] = [
        -.3, 0, 0, .41, .16, .21, np.deg2rad(-72), .98,
        *params['csf']]
    E[5, :] = [
        .25, 0, 0, .31, .11, .22, np.deg2rad(72), .98,
        *params['csf']]
    E[6, :] = [
        0, .35, 0, .15, .15, .15, 0, .617,
        *params['tumor']]
    E[7, :] = [
        0, .1, 0, .046, .046, .046, 0, .95,
        *params['tumor']]
    E[8, :] = [
        -.08, -.605, 0, .046, .023, .02, 0, .95,
        *params['tumor']]
    E[9, :] = [
        .06, -.605, 0, .046, .023, .02, np.deg2rad(-90), .95,
        *params['tumor']]
    E[10, :] = [
        0, -.1, 0, .046, .046, .046, 0, .95,
        *params['tumor']]
    E[11, :] = [
        0, -.605, 0, .023, .023, .023, 0, .95,
        *params['tumor']]
    E[12, :] = [
        .06, -.105, .3125, .056, .04, .1, np.deg2rad(-90), .93,
        *params['tumor']]
    E[13, :] = [
        0, .1, .3125, .056, .056, .1, 0, .98,
        *params['csf']]
    E[14, :] = [
        .56, -.4, .25, .2, .03, .1, np.deg2rad(70), .85,
        *params['blood-clot']]
    

    # Need to subtract some ellipses here...
    Eneg = np.zeros(E.shape)
    for ii in range(E.shape[0]):

        # Ellipsoid geometry
        Eneg[ii, :7] = E[ii, :7]

        # Tissue property differs after 4th subtracted ellipsoid
        if ii > 3:
            Eneg[ii, 7:] = E[3, 7:]
        else:
            Eneg[ii, 7:] = E[ii-1, 7:]

    # Throw out first as we skip this one in the paper's table
    Eneg = Eneg[1:, :]

    # Spin density is negative for subtraction
    Eneg[:, 7] *= -1

    # Paper doesn't use last blood-clot ellipsoid
    E = E[:-1, :]
    Eneg = Eneg[:-1, :]

    # Put both ellipsoid groups together
    E = np.concatenate((E, Eneg), axis=0)

    return E

# Monkey patch
mr_shepp_logan.mr_ellipsoid_parameters = mr_ellipsoid_parameters

def get_shepp_logan(
    ndim: int,
    shape: int | tuple[int, int] | tuple[int, int, int],
    ):
    """
    Get crisp Shepp-Logan tissue segmentation.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions. If ndim == 2, create a single slice
        2D phantom.
    shape: int | tuple[int, int] | tuple[int, int, int]
        Shape of the output data, the data will be interpolated to the given shape.
        If int, assume isotropic matrix.
        
    Returns
    -------
    np.ndarray.
        Shepp-Logan segmentation.

    """
    assert ndim == 2 or ndim == 3, ValueError(f"Number of spatial dimensions (={ndim}) must be either 2 or 3.")
    
    # default params
    if np.isscalar(shape):
        shape = shape * np.ones(ndim, dtype=int)
    elif ndim == 3:
        shape = [shape[1], shape[2], shape[0]]
        
    # cast to list
    shape = list(shape)
    assert len(shape) == ndim, ValueError("If shape must be either a scalar or a ndim-length sequence.")
    
    if ndim == 2:
        shape = shape + [5]
        zlims = (0, .5)
    else:
        zlims = (-1, 1)
        
    # build segmentation
    _, data, _ = mr_shepp_logan.mr_shepp_logan(shape, zlims=zlims)
    
    if ndim == 2:
        data = data[:, :, 0]
    else:
        data = data.transpose(-1, 0, 1)
        
    return np.ascontiguousarray(data)#.astype(int)
