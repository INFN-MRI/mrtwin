"""Test OSF database phantom generation."""

import os
import tempfile


import pytest


import numpy as np
import numpy.testing as npt


from mrtwin import osf_phantom


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("subject", [1, 2])
@pytest.mark.parametrize("shape", [None, 128, (128, 128), (128, 128, 128)])
@pytest.mark.parametrize("output_res", [None, 1.0, (1.0, 1.0, 1.0)])
def test_osf_phantom(ndim, subject, shape, output_res):
    """
    Test the osf_phantom function with various parameter combinations.
    """
    if ndim == 2 and shape == (128, 128, 128):
        pytest.skip("2D phantom not compatible with len(shape)==3")
    if ndim == 3 and shape == (128, 128):
        pytest.skip("3D phantom not compatible with len(shape)==2")
    if ndim == 2 and output_res == (1.0, 1.0, 1.0):
        pytest.skip("2D phantom not compatible with len(output_res)==3")
    if ndim == 3 and output_res == (1.0, 1.0):
        pytest.skip("3D phantom not compatible with len(output_res)==2")

    phantom = osf_phantom(
        ndim=ndim,
        subject=subject,
        shape=shape,
        output_res=output_res,
        verify=False,
    )

    # Validate the output is not None
    assert phantom is not None, "Phantom output should not be None."

    # Validate the shape of the output if a shape is provided
    if shape is not None:
        expected_shape = tuple([shape] * ndim) if isinstance(shape, int) else shape
        actual_shape = phantom.T1.shape
        npt.assert_allclose(actual_shape, expected_shape)


def test_osf_phantom_default():
    """
    Test the osf_phantom function with default parameters.
    """
    ndim = 2
    subject = 1

    phantom = osf_phantom(ndim=ndim, subject=subject, verify=False)

    # Validate the output is not None
    assert phantom is not None, "Phantom output should not be None."

    # Validate the default shape for 2D phantom (since no shape was provided)
    assert phantom.T1.shape == (256, 256)
