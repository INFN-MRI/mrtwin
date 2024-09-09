"""Shape broadcasting sub-routines."""

__all__ = ["_expand_shapes"]

import numpy as np


def _expand_shapes(*shapes):  # noqa
    shapes = [list(shape) for shape in shapes]
    max_ndim = max(len(shape) for shape in shapes)

    shapes_exp = [np.asarray([1] * (max_ndim - len(shape)) + shape) for shape in shapes]
    shapes_exp = np.stack(shapes_exp, axis=0)  # (nshapes, max_ndim)
    shapes_exp = np.max(shapes_exp, axis=0)

    # restore original shape in non-padded portions
    shapes_exp = [list(shapes_exp[: -len(shape)]) + shape for shape in shapes]

    return tuple(shapes_exp)
