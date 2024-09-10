"""Birdcage map generation."""

__all__ = ["_birdcage"]

import math


import numpy as np


def _birdcage(shape, coil_width, nrings, shift, dphi):  # noqa
    # default
    if shift is None:
        shift = [0.0 for ax in range(len(shape) - 1)]
    if nrings is None:
        nrings = np.max((shape[0] // 4, 1))

    # coil width and radius
    c_width = coil_width * min(shape[-2:])
    c_rad = 0.5 * c_width

    if len(shape) == 3:
        nc, ny, nx = shape
        phi = np.arange(nc) * (2 * math.pi / nc) + dphi
        y, x = np.mgrid[:ny, :nx]

        x0 = c_rad * np.cos(phi) + shape[-1] / 2.0 + shift[-1]
        y0 = c_rad * np.sin(phi) + shape[-2] / 2.0 + shift[-2]

        x_co = x[None, ...] - x0[:, None, None]
        y_co = y[None, ...] - y0[:, None, None]

        # coil magnitude
        rr = np.sqrt(x_co**2 + y_co**2) / (2 * c_width)

        # coil phase
        phi = np.arctan2(x_co, -y_co) - phi[:, None, None]

    elif len(shape) == 4:
        nc, nz, ny, nx = shape
        phi = np.arange(nc) * (2 * math.pi / (nc + nrings)) + dphi
        z, y, x = np.mgrid[:nz, :ny, :nx]

        x0 = c_rad * np.cos(phi) + shape[-1] / 2.0 + shift[-1]
        y0 = c_rad * np.sin(phi) + shape[-2] / 2.0 + shift[-2]
        z0 = (
            np.floor(np.arange(nc) / nrings)
            - 0.5 * (np.ceil(np.arange(nc) / nrings) - 1)
            + shape[-3] / 2.0
            + shift[-3]
        )

        x_co = x[None, ...] - x0[:, None, None, None]
        y_co = y[None, ...] - y0[:, None, None, None]
        z_co = z[None, ...] - z0[:, None, None, None]

        # coil magnitude
        rr = np.sqrt(x_co**2 + y_co**2 + z_co**2) / (2 * c_width)

        # coil phase
        phi = np.arctan2(x_co, -y_co) - phi[:, None, None, None]
    else:
        raise ValueError("Can only generate shape with length 3 or 4")

    # build coils
    rr[rr == 0.0] = 1.0
    smap = (1.0 / rr) * np.exp(1j * phi)

    return smap.astype(np.complex64)
