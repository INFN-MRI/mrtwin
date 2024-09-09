"""OSF download tools."""

__all__ = ["get_osf_maps"]

import os

from pathlib import Path
from typing import Sequence

import numpy as np
import nibabel as nib

from osfclient import OSF

from .._utils import ssl_verification, CacheDirType

from .. import _prescription

# PREDATOR dataset project ID
DATASET_ID = "qkbca"

# +fmt: off
SUB_ID = [1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 22, 23, 25, 27, 28]
# +fmt: on


# Directory where data will be stored
def get_osf_dir(osf_dir: CacheDirType = None) -> Path:
    """Get the OSF directory.

    Adapted from brainweb_dl._brainweb.get_brainweb_dir

    Parameters
    ----------
    osf_dir : CacheDirType
       osf_directory to download the data.

    Returns
    -------
    os.PathLike
        Path to osf_dir

    Notes
    -----
    The osf directory is set in the following order:
    - The osf_dir passed as argument.
    - The environment variable OSF_DIR.
    - The default osf_directory ~/.cache/osf.
    """
    if osf_dir is not None:
        osf_dir = Path(osf_dir)
    elif "OSF_DIR" in os.environ:
        osf_dir = Path(os.environ["OSF_DIR"])
    else:
        osf_dir = Path.home() / ".cache" / "osf"
    os.makedirs(Path(osf_dir), exist_ok=True)
    return Path(osf_dir)


# Function to download and o rganize data per subject
def get_osf_maps(
    ndim: int,
    subject: int,
    shape: int | Sequence[int] | None = None,
    output_res: float | Sequence[float] | None = None,
    osf_dir: CacheDirType = None,
    force: bool = False,
    verify: bool = True,
):
    """
    Get quantitative maps adaped from Open Science CBS Neuroimaging Repository.

    Base resolution if 0.4 mm isotropic. Each subject dataset include
    M0, T1, T2, T2* and Chi values (skull-stripped). Values are
    extrapolated to 3T.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions. If ndim == 2, use a single slice
        (central axial slice).
    subject : int
        Subject id to download.
    shape: int | Sequence[int] | None, optional
        Shape of the output data, the data will be interpolated to the given shape.
        If int, assume isotropic matrix. The default is None (original shape).
    output_res: float | Sequence[float] | None, optional
        Resolution of the output data, the data will be rescale to the given resolution.
        If scalar, assume isotropic resolution. The default is None (keep original 0.4mm iso).
    osf_dir : CacheDirType, optional
        osf_directory to download the data.
        The default is None (~/.cache/osf).
    force : bool, optional
        Force download even if the file already exists.
        The default is False.
    verify : bool, optional
        Enable SSL verification.
        DO NOT DISABLE (i.e., verify=False)IN PRODUCTION.
        The default is True.

    Returns
    -------
    np.ndarray.
        3D/4D stacked array of shape (ncontrasts, *shape).
        The order of contrasts is:

            1. Equilibrium magnetization (M0)
            2. T1 in [ms]
            3. T2 in [ms]
            4. T2* in [ms]
            5. Chi in [ppb]

    """
    assert ndim == 2 or ndim == 3, ValueError(
        f"Number of spatial dimensions (={ndim}) must be either 2 or 3."
    )
    assert subject in SUB_ID, ValueError(
        f"subject (={subject}) must be one of {SUB_ID}"
    )

    # Default params
    if shape is not None and np.isscalar(shape):
        shape = shape * np.ones(ndim, dtype=int)
    if shape is not None:
        assert len(shape) == ndim, ValueError(
            "If shape is not None, it must be either a scalar or a ndim-length sequence."
        )
    if output_res is not None and np.isscalar(output_res):
        output_res = output_res * np.ones(ndim, dtype=int)
    if output_res is not None:
        assert len(output_res) == ndim, ValueError(
            "If output_res is not None, it must be either or a scalar a ndim-length sequence."
        )

    # Original resolution (0.4 mm iso)
    orig_res = 0.4 * np.ones(ndim)

    # Get base directory
    base_dir = get_osf_dir(osf_dir)

    # Create a subject-specific directory
    sub_dir = os.path.join(base_dir, f"sub{subject:02d}")
    os.makedirs(sub_dir, exist_ok=True)

    # Actual download
    if verify is False:
        with ssl_verification(verify=False):
            data = _actual_download(sub_dir, subject, force)
    else:
        data = _actual_download(sub_dir, subject, force)

    # Stack maps
    data = np.stack(
        [data["PD"], data["qT1"], data["qT2"], data["qT2STAR"], data["QSM"]], axis=0
    )

    # Select single slice
    if ndim == 2:
        center = int(data.shape[-3] // 2)
        data = data[:, center, :, :]

    # Make sure it is contiguous
    data = np.ascontiguousarray(data)

    if shape is None and output_res is None:
        data = np.nan_to_num(data, posinf=0.0, neginf=0.0)
        return data.astype(np.float32)
    elif output_res is None:
        output_res = orig_res  # 0.4 mm iso

    # Set prescription
    data = _prescription.set_prescription(
        data, orig_res, data.shape[-ndim:], output_res, shape
    )

    # Clean-up
    data = np.nan_to_num(data, posinf=0.0, neginf=0.0)
    return data.astype(np.float32)


def _actual_download(sub_dir, subject, force):
    # Initialize OSF client
    osf = OSF()

    # Replace 'your_osf_project_id' with your actual OSF project ID
    project = osf.project(DATASET_ID)

    # Replace 'your_storage_name' with the actual storage name (e.g., 'osfstorage')
    storage = project.storage("osfstorage")

    for folder in storage.folders:
        # Check if file belongs to the subject (adjust according to your naming convention)
        if f"sub{subject:02d}" == folder.name:

            # initialize
            keys = []
            values = []

            for file in folder.files:
                file_path = os.path.join(sub_dir, file.name)

                # Erase file if force
                if os.path.exists(file_path) and force:
                    os.remove(file_path)

                # Skip( download if the file already exists
                if not os.path.exists(file_path):
                    with open(file_path, "wb") as f:
                        file.write_to(f)

                # Load data
                keys.append(_get_contrast(file.name))
                values.append(nib.load(file_path).get_fdata().T)

    return dict(zip(keys, values))


def _get_contrast(file_name):
    return file_name.split("_")[3]
