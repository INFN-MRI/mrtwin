"""Test rigid motion pattern generation."""

import pytest
import numpy as np

from mrtwin import rigid_motion


@pytest.mark.parametrize("ndims", [2, 3])
@pytest.mark.parametrize("nframes", [10, 100])
@pytest.mark.parametrize("degree", ["subtle", "moderate", "severe"])
def test_rigid_motion_basic(ndims, nframes, degree):
    """
    Test the rigid motion generator function for basic cases with predefined degrees of motion.
    """
    # Call the rigid motion function
    params = rigid_motion(ndims, nframes, degree=degree)

    if ndims == 2:
        angles = tuple([params[0]])
        translations = params[1:]
    elif ndims == 3:
        angles = params[:3]
        translations = params[3:]

    # Validate the shape of the returned arrays
    for angle in angles:
        assert angle.shape == (
            nframes,
        ), f"Expected angle shape to be {(nframes,)}, but got {angle.shape}."
    for translation in translations:
        assert translation.shape == (
            nframes,
        ), f"Expected translation shape to be {(nframes,)}, but got {translation.shape}."


@pytest.mark.parametrize(
    "degree, max_rot, max_trans",
    [
        ("subtle", 5.0, 2.0),
        ("moderate", 10.0, 8.0),
        ("severe", 16.0, 16.0),
    ],
)
def test_rigid_motion_degree_values(degree, max_rot, max_trans):
    """
    Test the rigid motion generator with predefined severity levels to ensure the values fall within expected ranges.
    """
    ndims = 3
    nframes = 50

    params = rigid_motion(ndims, nframes, degree=degree)
    angles = params[:3]
    translations = params[3:]

    # Check that rotation values are within the expected range
    for angle in angles:
        assert np.all(
            np.abs(angle) <= max_rot
        ), f"Expected max rotation {max_rot} degrees, but got a value exceeding that."

    # Check that translation values are within the expected range
    for translation in translations:
        assert np.all(
            np.abs(translation) <= max_trans
        ), f"Expected max translation {max_trans} mm, but got a value exceeding that."


@pytest.mark.parametrize("degree", [(7.0, 3.5), (15.0, 6.0)])
def test_rigid_motion_custom_degree(degree):
    """
    Test the rigid motion generator with custom degrees for both rotation and translation.
    """
    ndims = 3
    nframes = 50

    params = rigid_motion(ndims, nframes, degree=degree)
    angles = params[:3]
    translations = params[3:]

    # Validate the max rotation and translation
    max_rot, max_trans = degree

    for angle in angles:
        assert np.all(
            np.abs(angle) <= max_rot
        ), f"Expected max rotation {max_rot} degrees, but got a value exceeding that."
    for translation in translations:
        assert np.all(
            np.abs(translation) <= max_trans
        ), f"Expected max translation {max_trans} mm, but got a value exceeding that."


def test_rigid_motion_seed():
    """
    Test that the random seed produces reproducible results.
    """
    ndims = 3
    nframes = 50
    degree = "moderate"
    seed = 42

    # Generate motion with the seed
    params1 = rigid_motion(ndims, nframes, degree=degree, seed=seed)
    angles1 = params1[:3]
    translations1 = params1[3:]

    params2 = rigid_motion(ndims, nframes, degree=degree, seed=seed)
    angles2 = params1[:3]
    translations2 = params2[3:]

    # Ensure that results are identical when the seed is set
    for a1, a2 in zip(angles1, angles2):
        np.testing.assert_array_equal(
            a1, a2, err_msg="Angles do not match for the same seed."
        )
    for t1, t2 in zip(translations1, translations2):
        np.testing.assert_array_equal(
            t1, t2, err_msg="Translations do not match for the same seed."
        )


@pytest.mark.parametrize("ndims", [2, 3])
@pytest.mark.parametrize("nframes", [10, 100])
def test_rigid_motion_dimensions(ndims, nframes):
    """
    Test that the number of dimensions (2D/3D) affects the number of outputs correctly.
    """
    degree = "moderate"

    params = rigid_motion(ndims, nframes, degree=degree)
    angles = params[:3]
    translations = params[3:]

    if ndims == 2:
        angles = tuple([params[0]])
        translations = params[1:]
    elif ndims == 3:
        angles = params[:3]
        translations = params[3:]

    # Validate that the correct number of angles and translations are generated based on ndims
    if ndims == 2:
        assert len(angles) == 1, f"Expected 1 angle, but got {len(angles)}."
    elif ndims == 3:
        assert len(angles) == ndims, f"Expected {ndims} angles, but got {len(angles)}."
    assert (
        len(translations) == ndims
    ), f"Expected {ndims} translations, but got {len(translations)}."
