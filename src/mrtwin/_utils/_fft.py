"""Centered ND FFT subroutines."""

__all__ = ["fftc", "ifftc"]

import numpy as np


def fftc(x, ax):
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax
    )


def ifftc(x, ax):  # noqa
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax
    )
