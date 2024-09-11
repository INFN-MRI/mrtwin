"""Gradient Impulse Response Function generation routines."""

__all__ = ["generate_girf"]

from typing import Sequence


import numpy as np


def generate_girf(
    dt: float, N: int, fwhm: float | Sequence[float], delay: float | Sequence[float]
):
    """
    Generate an approximate Gradient Impulse Response Function (GIRF).

    The GIRF is approximated as a complex-valued array, whose magnitude (modeled)
    as a Gaussian of the specified Full Width Half Maximum, which dampens
    high-frequency components of the k-space trajectory, and whose phase
    is a linear ramp representing a non-integer shift of the waveform.

    The response is modeled independently for the three physical axes ``z, y, x``.

    Parameters
    ----------
    dt : float
        Gradient raster time (in seconds).
    N : int
        Length of the GIRF.
    fwhm : float | Sequence[float]
        Width in Hz of the magnitude response of the gradient
        system for the three axes ``z, y, x``.
        If it is a scalar, assume the
        same delay for all the axes.
    delay : float | Sequence[float]
        Non-integer delay (in seconds) for the three axes
        ``z, y, x``. If it is a scalar, assume the
        same delay for all the axes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - The frequency axis (in Hz) as a NumPy array.
        - Complex GIRF of shape ``(3, N)``.

    Notes
    -----
    Typical values for gradient raster times ``dt`` are around 5-10 us
    (GE: 4us; Siemens: 10us; Philips: 6.4us)

    GIRF magnitudes Full Width Half Maximum are on the order of 5 kHz,
    while gradient delays are on the order of 1us.

    GIRF for the x and y axes are more similar to each other
    when compared to z axes.

    GIRF is typically measured (via thin slice method or dynamic field camera)
    with a time window T around 100ms (N = T / dt).

    Examples
    --------
    >>> from mrtwin import generate_girf

    Assuming same fwhm and delay for each axes, we have:

    >>> freqs, girf = generate_girf(4e-6, 25000, 5.0e3, 1.1e-6)

    """
    # Handle scalar case
    if np.isscalar(fwhm):
        fwhm = [fwhm] * 3
    fwhm = list(fwhm)
    if np.isscalar(delay):
        delay = [delay] * 3
    delay = list(delay)

    # Calculate frequency axis (Hz)
    freqs = np.fft.fftfreq(N, d=dt)

    # Generate GIRF magnitude
    girf_magn = np.stack([_gaussian_freq_domain(dt, N, f) for f in fwhm], axis=0)
    girf_magn = girf_magn.astype(np.float32)

    # Generate GIRF phase
    girf_phase = np.stack([_phase_shift_line(dt, N, d) for d in delay], axis=0)
    girf_phase = girf_phase.astype(np.float32)

    return freqs, girf_magn * np.exp(1j * girf_phase)


def _gaussian_freq_domain(dt: float, N: int, fwhm: float):
    # Calculate frequency axis (Hz)
    freqs = np.fft.fftfreq(N, d=dt)

    # Calculate standard deviation from FWHM
    sigma_f = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # Generate Gaussian in the frequency domain
    gaussian = np.exp(-0.5 * (freqs / sigma_f) ** 2)

    return gaussian


def _phase_shift_line(dt: float, N: int, delay: float):
    # Calculate frequency axis (Hz)
    freqs = np.fft.fftfreq(N, d=dt)

    # Generate the linear phase ramp: -2 * pi * f * delay
    phase_ramp = -2 * np.pi * freqs * delay

    return phase_ramp
