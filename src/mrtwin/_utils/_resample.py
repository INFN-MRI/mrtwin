"""Array resampling routines."""

__all__ = ["resample"]

import numpy as np


from scipy.ndimage import zoom


from ._broadcasting import _expand_shapes


def resample(input, oshape):
    """
    Resample a n-dimensional signal.

    Parameters
    ----------
    input : np.ndarray
        Input array of shape ``(..., ishape)``.
    oshape : Sequence
        Output shape.

    Returns
    -------
    output : np.ndarray
        Resampled tensor of shape ``(..., oshape)``.

    """
    if isinstance(oshape, int):
        oshape = [oshape]

    # get initial and final shapes
    ishape1, oshape1 = _expand_shapes(input.shape, oshape)

    # interpolate
    scale = np.asarray(oshape1) / np.asarray(ishape1)

    return zoom(input, scale, order=1)
