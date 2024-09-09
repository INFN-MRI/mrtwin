"""Test Brainweb phantom generation."""

import os
import tempfile


import pytest


import numpy.testing as npt


from mrtwin import brainweb_phantom


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("subject", [4, 44])
@pytest.mark.parametrize("shape", [None, 128, (128, 128), (128, 128, 128)])
@pytest.mark.parametrize("model", ["single-pool", "mw-model", "mt-model", "mwmt-model"])
@pytest.mark.parametrize("segtype", ["crisp", "fuzzy", False])
@pytest.mark.parametrize("output_res", [None, 1.0, (1.0, 1.0), (1.0, 1.0, 1.0)])
def test_brainweb_phantom(ndim, subject, shape, model, segtype, output_res):
    """
    Test the brainweb_phantom function with various parameter combinations.
    """
    if ndim == 2 and shape == (128, 128, 128):
        pytest.skip("2D phantom not compatible with len(shape)==3")
    if ndim == 3 and shape == (128, 128):
        pytest.skip("3D phantom not compatible with len(shape)==2")
    if ndim == 2 and output_res == (1.0, 1.0, 1.0):
        pytest.skip("2D phantom not compatible with len(output_res)==3")
    if ndim == 3 and output_res == (1.0, 1.0):
        pytest.skip("3D phantom not compatible with len(output_res)==2")

    phantom = brainweb_phantom(
        ndim=ndim,
        subject=subject,
        shape=shape,
        model=model,
        segtype=segtype,
        output_res=output_res,
        verify=False,
    )

    # Validate the output is not None
    assert phantom is not None, "Phantom output should not be None."

    # Validate the shape of the output if a shape is provided
    if shape is not None:
        expected_shape = tuple([shape] * ndim) if isinstance(shape, int) else shape
        actual_shape = phantom.shape[-ndim:] if segtype else phantom.T1.shape[-ndim:]
        npt.assert_allclose(actual_shape, expected_shape)


def test_brainweb_phantom_default():
    """
    Test the brainweb_phantom function with default parameters.
    """
    ndim = 2
    subject = 4

    phantom = brainweb_phantom(ndim=ndim, subject=subject, verify=False)

    # Validate the output is not None
    assert phantom is not None, "Phantom output should not be None."

    # Validate the default shape for 2D phantom (since no shape was provided)
    assert phantom.shape[-2:] == (200, 200)
