"""Test static field map generation."""

import pytest
import numpy as np

from mrtwin import b0field


@pytest.mark.parametrize("shape", [(128, 128), (64, 64), (64, 64, 64)])
def test_b0field_basic_shape(shape):
    """
    Test the b0field function with basic input shapes for chi (susceptibility map).
    """
    # Create a random susceptibility map (chi) of the given shape
    chi = np.random.rand(*shape)

    # Call the b0field function
    b0map = b0field(chi)

    # Validate the output is a numpy ndarray
    assert isinstance(b0map, np.ndarray), "B0 map output should be a numpy ndarray."

    # Validate the output shape matches the input shape
    assert b0map.shape == shape, f"Expected shape {shape}, but got {b0map.shape}."


@pytest.mark.parametrize("b0range", [None, (-500, 500), (-1000, 1000)])
def test_b0field_b0range(b0range):
    """
    Test the b0field function with different b0range values.
    """
    shape = (128, 128)
    chi = np.random.rand(*shape)

    # Call the b0field function with varying b0range
    b0map = b0field(chi, b0range=b0range)

    # Validate the output is a numpy ndarray
    assert isinstance(b0map, np.ndarray), "B0 map output should be a numpy ndarray."

    # Validate the shape of the output matches the input shape
    assert b0map.shape == shape, f"Expected shape {shape}, but got {b0map.shape}."

    if b0range:
        # Check that values are within the specified range
        assert np.all(b0map >= b0range[0]) and np.all(b0map <= b0range[1]), (
            f"B0 map values should be in range {b0range}, but got min {b0map.min()} "
            f"and max {b0map.max()}"
        )


def test_b0field_default_b0range():
    """
    Test the b0field function with default b0range and a static field B0 of 1.5T.
    """
    shape = (128, 128)
    chi = np.random.rand(*shape)

    # Call the b0field function without specifying b0range (default)
    b0map = b0field(chi, B0=1.5)

    # Validate the output is a numpy ndarray
    assert isinstance(b0map, np.ndarray), "B0 map output should be a numpy ndarray."

    # Validate the output shape matches the input shape
    assert b0map.shape == shape, f"Expected shape {shape}, but got {b0map.shape}."


@pytest.mark.parametrize("mask", [None, np.ones((128, 128)), np.zeros((128, 128))])
def test_b0field_with_mask(mask):
    """
    Test the b0field function with and without a mask.
    """
    shape = (128, 128)
    chi = np.random.rand(*shape)

    # Call the b0field function with the mask
    b0map = b0field(chi, mask=mask)

    # Validate the output is a numpy ndarray
    assert isinstance(b0map, np.ndarray), "B0 map output should be a numpy ndarray."

    # Validate the shape of the output matches the input shape
    assert b0map.shape == shape, f"Expected shape {shape}, but got {b0map.shape}."


def test_b0field_with_3d_input():
    """
    Test the b0field function with 3D input data.
    """
    shape = (64, 64, 64)
    chi = np.random.rand(*shape)

    # Call the b0field function with 3D input
    b0map = b0field(chi)

    # Validate the output is a numpy ndarray
    assert isinstance(b0map, np.ndarray), "B0 map output should be a numpy ndarray."

    # Validate the output shape matches the input shape
    assert b0map.shape == shape, f"Expected shape {shape}, but got {b0map.shape}."


def test_b0field_output_units():
    """
    Test that the b0field output has units in Hz by checking magnitude with different B0 values.
    """
    shape = (128, 128)
    chi = np.random.rand(*shape)

    # Compute B0 maps with different static field strengths
    b0map_1T = b0field(chi, B0=1.0)
    b0map_3T = b0field(chi, B0=3.0)

    # Ensure the output is still a numpy ndarray
    assert isinstance(b0map_1T, np.ndarray), "B0 map output should be a numpy ndarray."
    assert isinstance(b0map_3T, np.ndarray), "B0 map output should be a numpy ndarray."

    # Ensure the shapes are correct
    assert b0map_1T.shape == shape, f"Expected shape {shape}, but got {b0map_1T.shape}."
    assert b0map_3T.shape == shape, f"Expected shape {shape}, but got {b0map_3T.shape}."

    # Check that B0 scaling is applied (3T should have larger field values than 1T)
    assert np.max(np.abs(b0map_3T)) > np.max(
        np.abs(b0map_1T)
    ), "B0 field with 3T should have larger field values than with 1T."
