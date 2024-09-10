"""Test transmit field maps generation."""

import pytest
import numpy as np

from mrtwin import b1field


@pytest.mark.parametrize("shape", [(128, 128), (64, 64), (64, 64, 64)])
def test_b1field_basic_shape(shape):
    """
    Test the b1field function with basic input shapes.
    """
    # Call the b1field function
    b1map = b1field(shape)

    # Validate the output is a numpy ndarray
    assert isinstance(b1map, np.ndarray), "B1 map output should be a numpy ndarray."

    # Check that the output has the correct shape (for default nmodes=1)
    expected_shape = shape
    assert (
        b1map.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {b1map.shape}."


@pytest.mark.parametrize("nmodes", [2, 4])
def test_b1field_nmodes(nmodes):
    """
    Test the b1field function with different number of modes.
    """
    shape = (128, 128)

    # Call the b1field function with different nmodes
    b1map = b1field(shape, nmodes=nmodes)

    # Validate the output is a numpy ndarray
    assert isinstance(b1map, np.ndarray), "B1 map output should be a numpy ndarray."

    # Check that the output has the correct shape based on nmodes
    expected_shape = (nmodes,) + shape
    assert (
        b1map.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {b1map.shape}."


@pytest.mark.parametrize("b1range", [(0.5, 2.0), (0.8, 1.2), (1.0, 3.0)])
def test_b1field_b1range(b1range):
    """
    Test the b1field function with different b1range values.
    """
    shape = (128, 128)

    # Call the b1field function with different b1range
    b1map = b1field(shape, b1range=b1range)

    # Validate the output is a numpy ndarray
    assert isinstance(b1map, np.ndarray), "B1 map output should be a numpy ndarray."

    # Validate that the output shape is correct
    assert b1map.shape == shape, f"Expected shape {(1, *shape)}, but got {b1map.shape}."

    # Check that values are within the specified b1range
    assert np.all(b1map >= b1range[0]) and np.all(b1map <= b1range[1]), (
        f"B1 map values should be in range {b1range}, but got min {b1map.min()} "
        f"and max {b1map.max()}"
    )


@pytest.mark.parametrize("dphi", [0.0, 45.0, 90.0])
def test_b1field_dphi(dphi):
    """
    Test the b1field function with different coil rotation angles (dphi).
    """
    shape = (128, 128)

    # Call the b1field function with different dphi values
    b1map = b1field(shape, dphi=dphi)

    # Validate the output is a numpy ndarray
    assert isinstance(b1map, np.ndarray), "B1 map output should be a numpy ndarray."

    # Validate the output shape matches the input shape
    assert b1map.shape == shape, f"Expected shape {(1, *shape)}, but got {b1map.shape}."


def test_b1field_shift():
    """
    Test the b1field function with a coil center shift.
    """
    shape = (128, 128)

    # Call the b1field function with a shift
    shift = (-3, 5)
    b1map = b1field(shape, shift=shift)

    # Validate the output is a numpy ndarray
    assert isinstance(b1map, np.ndarray), "B1 map output should be a numpy ndarray."

    # Validate the output shape matches the input shape
    assert b1map.shape == shape, f"Expected shape {(1, *shape)}, but got {b1map.shape}."


def test_b1field_with_mask():
    """
    Test the b1field function with a mask.
    """
    shape = (128, 128)

    # Create a binary mask
    mask = np.ones(shape)

    # Call the b1field function with a mask
    b1map = b1field(shape, mask=mask)

    # Validate the output is a numpy ndarray
    assert isinstance(b1map, np.ndarray), "B1 map output should be a numpy ndarray."

    # Validate the shape of the output matches the input shape
    assert b1map.shape == shape, f"Expected shape {(1, *shape)}, but got {b1map.shape}."


@pytest.mark.parametrize("coil_width", [1.0, 1.5, 2.0])
def test_b1field_coil_width(coil_width):
    """
    Test the b1field function with different coil widths.
    """
    shape = (128, 128)

    # Call the b1field function with different coil widths
    b1map = b1field(shape, coil_width=coil_width)

    # Validate the output is a numpy ndarray
    assert isinstance(b1map, np.ndarray), "B1 map output should be a numpy ndarray."

    # Validate the output shape matches the input shape
    assert b1map.shape == shape, f"Expected shape {(1, *shape)}, but got {b1map.shape}."


@pytest.mark.parametrize("ncoils", [2, 4, 8])
def test_b1field_ncoils(ncoils):
    """
    Test the b1field function with different numbers of coils.
    """
    shape = (128, 128)

    # Call the b1field function with different ncoils
    b1map = b1field(shape, ncoils=ncoils)

    # Validate the output is a numpy ndarray
    assert isinstance(b1map, np.ndarray), "B1 map output should be a numpy ndarray."

    # Validate the output shape matches the input shape
    assert b1map.shape == shape, f"Expected shape {(1, *shape)}, but got {b1map.shape}."


def test_b1field_with_3d_input():
    """
    Test the b1field function with 3D input data.
    """
    shape = (64, 64, 64)

    # Call the b1field function with 3D input
    b1map = b1field(shape)

    # Validate the output is a numpy ndarray
    assert isinstance(b1map, np.ndarray), "B1 map output should be a numpy ndarray."

    # Validate the output shape matches the input shape
    assert b1map.shape == shape, f"Expected shape {(1, *shape)}, but got {b1map.shape}."
