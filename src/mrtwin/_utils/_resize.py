"""Array shape manipulation routines."""

__all__ = ["resize"]

import numpy as np


from ._broadcasting import _expand_shapes


def resize(input, oshape):
    """
    Resize with zero-padding or cropping.

    Adapted from SigPy [1].

    Parameters
    ----------
    input : np.ndarray
        Input array of shape ``(..., ishape)``.
    oshape : Sequence
        Output shape.

    Returns
    -------
    output : np.ndarray
        Zero-padded or cropped array of shape ``(..., oshape)``.

    Examples
    --------
    >>> import numpy as np
    >>> import mrtwin

    We can pad tensors to desired shape:

    >>> x = np.asarray([0, 1, 0])
    >>> y = mrtwin.resize(x, [5])
    >>> y
    array([0, 0, 1, 0, 0])

    Batch dimensions are automatically expanded (pad will be applied starting from rightmost dimension):

    >>> x = np.asarray([0, 1, 0])[None, ...]
    >>> x.shape
    (1, 3)
    >>> y = mrtwin.resize(x, [5]) # len(oshape) == 1
    >>> y.shape
    (1, 5)

    Similarly, if oshape is smaller than ishape, the tensor will be cropped:

    >>> x = np.asarray([0, 0, 1, 0, 0])
    >>> y = mrtwin.resize(x, [3])
    >>> y
    tensor([0, 1, 0])

    Again, batch dimensions are automatically expanded:

    >>> x = np.asarray([0, 0, 1, 0, 0])[None, ...]
    >>> x.shape
    (1, 5)
    >>> y = mrtwin.resize(x, [3]) # len(oshape) == 1
    >>> y.shape
    (1, 3)

    References
    ----------
    [1] https://github.com/mikgroup/sigpy/blob/main/sigpy/util.py

    """
    if isinstance(oshape, int):
        oshape = [oshape]

    ishape1, oshape1 = _expand_shapes(input.shape, oshape)

    if ishape1 == oshape1:
        return input

    # shift not supported for now
    ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape1, oshape1)]
    oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape1, oshape1)]

    copy_shape = [
        min(i - si, o - so) for i, si, o, so in zip(ishape1, ishift, oshape1, oshift)
    ]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    output = np.zeros(oshape1, dtype=input.dtype)
    input = input.reshape(ishape1)
    output[oslice] = input[islice]

    return output
