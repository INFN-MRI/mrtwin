"""Test Shepp-Logan phantom generation."""

import os
import tempfile


import pytest


import numpy.testing as npt


from mrtwin import shepplogan_phantom


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("shape", [128, (128, 128), (128, 128, 128)])
@pytest.mark.parametrize("model", ["single-pool", "mw-model", "mt-model", "mwmt-model"])
@pytest.mark.parametrize("segtype", ["crisp", False])
def test_shepplogan_phantom(ndim, shape, model, segtype):
    """
    Test the shepplogan_phantom function with various parameter combinations.
    """
    if ndim == 2 and shape == (128, 128, 128):
        pytest.skip("2D phantom not compatible with len(shape)==3")
    if ndim == 3 and shape == (128, 128):
        pytest.skip("3D phantom not compatible with len(shape)==2")

    phantom = shepplogan_phantom(ndim=ndim, shape=shape, model=model, segtype=segtype)

    # Validate the output is not None
    assert phantom is not None, "Phantom output should not be None."

    # Validate the shape of the output if a shape is provided
    if shape is not None:
        expected_shape = tuple([shape] * ndim) if isinstance(shape, int) else shape
        actual_shape = phantom.shape[-ndim:] if segtype else phantom.T1.shape[-ndim:]
        npt.assert_allclose(actual_shape, expected_shape)
