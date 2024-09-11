"""Test GIRF generation."""

import pytest
import numpy as np

from mrtwin import generate_girf


@pytest.mark.parametrize(
    "dt, N",
    [
        (10e-6, 25000),
        (4e-6, 50000),
    ],
)
@pytest.mark.parametrize("fwhm", [5.0e3, [5.0e3, 10.0e3, 15.0e3]])
@pytest.mark.parametrize("delay", [0.1e-6, [0.1e-6, 0.2e-6, 0.3e-6]])
def test_generate_girf_basic(dt, N, fwhm, delay):
    """
    Test the generate_girf function for basic input values with both scalar and sequence delays/FWHMs.
    """
    freq_axis, girf = generate_girf(dt, N, fwhm, delay)

    # Check if the frequency axis and GIRF have the expected shapes
    assert isinstance(freq_axis, np.ndarray), "Frequency axis should be a NumPy array."
    assert isinstance(girf, np.ndarray), "GIRF should be a NumPy array."

    # Validate shapes
    assert freq_axis.shape == (
        N,
    ), f"Expected frequency axis shape {(N,)}, but got {freq_axis.shape}."
    assert girf.shape == (3, N), f"Expected GIRF shape (3, {N}), but got {girf.shape}."


@pytest.mark.parametrize("fwhm", [(5.0e3, 10.0e3, 15.0e3), 7.0e3])
@pytest.mark.parametrize("delay", [(0.1e-6, 0.2e-6, 0.3e-6), 0.2e-6])
def test_generate_girf_axis_independence(fwhm, delay):
    """
    Test the GIRF to ensure that each axis responds independently when provided with different FWHM and delay values.
    """
    if not (isinstance(fwhm, tuple)) and not (isinstance(delay, tuple)):
        pytest.skip("either fwhm or delay must be a tuple")

    dt = 4e-6
    N = 25000

    freq_axis, girf = generate_girf(dt, N, fwhm, delay)

    # Ensure that the GIRF contains different values for different axes when FWHM or delay varies
    assert not np.allclose(
        girf[0], girf[1]
    ), "GIRF for axis 0 and 1 should not be identical when parameters differ."
    assert not np.allclose(
        girf[1], girf[2]
    ), "GIRF for axis 1 and 2 should not be identical when parameters differ."
    assert not np.allclose(
        girf[0], girf[2]
    ), "GIRF for axis 0 and 2 should not be identical when parameters differ."


def test_generate_girf_single_axis_behavior():
    """
    Test the function with the same FWHM and delay for all axes and ensure uniform GIRF.
    """
    dt = 4e-6
    N = 25000
    fwhm = 7.0e3
    delay = 0.2e-6

    _, girf = generate_girf(dt, N, fwhm, delay)

    # Ensure that all axes should behave identically when FWHM and delay are the same
    assert np.allclose(
        girf[0], girf[1]
    ), "GIRF for all axes should be identical when FWHM and delay are identical."
    assert np.allclose(
        girf[1], girf[2]
    ), "GIRF for all axes should be identical when FWHM and delay are identical."
