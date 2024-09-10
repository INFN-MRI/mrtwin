"""Test sensitivity maps generation."""

import pytest
import numpy as np

from mrtwin import sensmap


@pytest.mark.parametrize("shape", [(8, 128, 128), (8, 64, 64), (8, 64, 64, 64)])
def test_sensmap_basic_shape(shape):
    """
    Test the sensmap function with basic shape parameters.
    """
    smap = sensmap(shape)

    # Validate the output is not None
    assert smap is not None, "Sensmap output should not be None."

    # Validate the output type is numpy.ndarray
    assert isinstance(smap, np.ndarray), "Sensmap output should be a numpy ndarray."

    # Validate the shape of the output matches the input shape
    assert smap.shape == shape, f"Expected shape {shape}, but got {smap.shape}."

    # Check that the output contains complex numbers
    assert np.iscomplexobj(smap), "Sensmap output should be a complex numpy array."


@pytest.mark.parametrize("coil_width", [1.0, 2.0, 3.5])
def test_sensmap_coil_width(coil_width):
    """
    Test the sensmap function with different coil_width values.
    """
    shape = (8, 128, 128)
    smap = sensmap(shape, coil_width=coil_width)

    # Validate the output is a numpy ndarray
    assert isinstance(smap, np.ndarray), "Sensmap output should be a numpy ndarray."

    # Validate the shape of the output matches the input shape
    assert smap.shape == shape, f"Expected shape {shape}, but got {smap.shape}."


@pytest.mark.parametrize("shift", [None, (0, 0), (5, -5), (10, 10, 10)])
def test_sensmap_shift(shift):
    """
    Test the sensmap function with different shift values.
    """
    shape = (8, 128, 128)
    if len(shift or ()) == 3:
        shape = (8, 64, 64, 64)
    smap = sensmap(shape, shift=shift)

    # Validate the output is a numpy ndarray
    assert isinstance(smap, np.ndarray), "Sensmap output should be a numpy ndarray."

    # Validate the shape of the output matches the input shape
    assert smap.shape == shape, f"Expected shape {shape}, but got {smap.shape}."


@pytest.mark.parametrize("dphi", [0.0, 15.0, 90.0])
def test_sensmap_dphi(dphi):
    """
    Test the sensmap function with different dphi values (coil rotation angle).
    """
    shape = (8, 128, 128)
    smap = sensmap(shape, dphi=dphi)

    # Validate the output is a numpy ndarray
    assert isinstance(smap, np.ndarray), "Sensmap output should be a numpy ndarray."

    # Validate the shape of the output matches the input shape
    assert smap.shape == shape, f"Expected shape {shape}, but got {smap.shape}."


@pytest.mark.parametrize("nrings", [None, 2, 4])
def test_sensmap_nrings(nrings):
    """
    Test the sensmap function with different nrings values.
    """
    shape = (8, 128, 128)
    smap = sensmap(shape, nrings=nrings)

    # Validate the output is a numpy ndarray
    assert isinstance(smap, np.ndarray), "Sensmap output should be a numpy ndarray."

    # Validate the shape of the output matches the input shape
    assert smap.shape == shape, f"Expected shape {shape}, but got {smap.shape}."


def test_sensmap_default():
    """
    Test the sensmap function with default parameters.
    """
    shape = (8, 128, 128)
    smap = sensmap(shape)

    # Validate the output is a numpy ndarray
    assert isinstance(smap, np.ndarray), "Sensmap output should be a numpy ndarray."

    # Validate the shape of the output matches the input shape
    assert smap.shape == shape, f"Expected shape {shape}, but got {smap.shape}."

    # Validate the output contains complex numbers
    assert np.iscomplexobj(smap), "Sensmap output should be a complex numpy array."
