"""Set spatial prescription for a given phantom."""

__all__ = ["set_prescription"]

from typing import Sequence

import numpy as np

from . import _resample

def set_prescription(
        data: np.ndarray, 
        orig_res: Sequence[float], 
        orig_shape: Sequence[int], 
        output_res: Sequence[float],
        output_shape: Sequence[int] | None = None, 
        ):
    """
    Set prescription (fov and resolution) for an input dataset.

    Parameters
    ----------
    data : np.ndarray
        Input dataset of shape (..., (nz0), ny0, nx0).
    orig_res : Sequence[float]
        Input resolution ((dz0), dy0, dx0), same units as output_res.
    orig_shape : Sequence[int]
        Input shape ((nz0), ny0, nx0).
    output_res : Sequence[float]
        Output resolution ((dz1), dy1, dx1), same units as orig_res.
    output_shape : Sequence[int] | None, optional
        Output shape ((nz1), ny1, nx1). If not provided,
        calculate shape to preserve original FoV 
        ((dz0 * nz0), dy0 * ny0, dx0 * nx0).

    Returns
    -------
    data : np.ndarray
        Resampled data to ((nz1), ny1, nx1) so that
        output resolution is ((dz1), dy1, dx1).

    """  
    # c0nvert to array
    orig_res = np.asarray(orig_res)
    output_res = np.asarray(output_res)
    
    # get fov
    orig_fov = np.asarray(orig_shape) * orig_res
    
    # default output shape
    if output_shape is None:
        output_shape = orig_fov / output_res
    
    # get output fov
    output_fov = output_shape * output_res
    
    # select region of interest
    scale = output_fov / orig_fov
    roi = scale * np.asarray(orig_shape)
    roi = np.round(roi).astype(int)
    
    # crop or pad
    data = _resample.resize(data, roi)
    
    # resample to desired output resolution
    data = _resample.resample(data, output_shape)
    
    return data
