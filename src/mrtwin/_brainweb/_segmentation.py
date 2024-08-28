"""
Wrapper around brainweb-dl to extract tissue segmentation.

N.B. I did not directly used get_mri to allow for
SSL verification disabling (not advised, use with care).
"""

__all__ = ["get_brainweb"]

import gzip
import io
import logging
import os
import requests
import warnings

import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import brainweb_dl

from contextlib import contextmanager
from pathlib import Path

from numpy.typing import DTypeLike
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm.auto import tqdm

from brainweb_dl._brainweb import (
    BASE_URL,
    SUB_ID,
    STD_RES_SHAPE,
    BrainWebDirType,
    load_array,
)

from .. import _prescription

@contextmanager
def ssl_verification(verify=True):  # noqa
    # Default behaviour (do not disable)
    if verify:
        yield

    # Store the original `requests.Session.send` method
    original_send = requests.Session.send

    # Define a new `send` method that disables SSL verification
    def send_with_ssl_disabled(self, *args, **kwargs):
        kwargs["verify"] = False
        return original_send(self, *args, **kwargs)

    # Replace the original `send` method with the new one
    requests.Session.send = send_with_ssl_disabled

    try:
        # Execute the code within the context
        yield
    finally:
        # Restore the original `send` method
        requests.Session.send = original_send


def _request_get_brainweb(
    download_command: str,
    path: os.PathLike,
    force: bool = False,
    dtype: DTypeLike = np.float32,
    shape: tuple = STD_RES_SHAPE,
) -> np.ndarray:
    """Request to download brainweb dataset.

    Parameters
    ----------
    download_command : str
        Formatted request code to download a volume from brainweb.
    path : os.PathLike
        Path to save the downloaded file.
    force : bool, optional
        Force download even if the file already exists.
        The default is False.
    dtype : DTypeLike, optional
        Data type of the downloaded file.
        The default is np.float32.
    shape : tuple, optional
        Shape of the downloaded file.
        The default is (182, 217, 181)

    Returns
    -------
    np.ndarray
        Downloaded file.

    Raises
    ------
    Exception
        If the download fails.
    """
    path = Path(path)
    # don't download if it cached.
    if path.exists() and not force:
        return load_array(path)
    d = requests.get(
        _get_url(BASE_URL, download_command),
        stream=True,
        headers={"Accept-Encoding": "identity", "Content-Encoding": "gzip"},
    )

    # download
    with io.BytesIO() as buffer, tqdm(
        total=float(d.headers.get("Content-length", 0)),
        desc=f"Downloading {download_command}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=False,
        position=2,
    ) as pbar:
        for chunk in d.iter_content(chunk_size=1024):
            buffer.write(chunk)
            pbar.update(len(chunk))
        data = np.frombuffer(gzip.decompress(buffer.getvalue()), dtype=dtype)
    if data.size != np.prod(shape):
        raise ValueError(f"Mismatch between data size and shape {data.size} != {shape}")
    data = abs(data).reshape(shape)
    return data


def _get_url(BASE_URL, download_command):
    URL = (
        BASE_URL
        + "?"
        + "&".join(
            [
                f"{k}={v}"
                for k, v in {
                    "do_download_alias": download_command,
                    "format_value": "raw_short",
                    "zip_value": "gnuzip",
                    "who_name": "",
                    "who_institution": "",
                    "who_email": "",
                    "download_for_real": "%5BStart+download%21%5D",
                }.items()
                if v
            ]
        )
    )
    return URL


# Monkey patch
brainweb_dl._brainweb._request_get_brainweb = _request_get_brainweb

# Actual functions
logger = logging.getLogger("brainweb_dl")


def get_brainweb(
    ndim: int,
    subject: int,
    shape: int | tuple[int, int] | tuple[int, int, int] | None = None,
    output_res: float | tuple[float, float] | tuple[float, float, float] | None = None,
    brainweb_dir: BrainWebDirType = None,
    force: bool = False,
    verify: bool = True,
):
    """
    Get fuzzy BrainWeb tissue segmentation.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions. If ndim == 2, use a single slice
        (central axial slice).
    subject : int
        Subject id to download.
    shape: int | tuple[int, int] | tuple[int, int, int] | None, optional
        Shape of the output data, the data will be interpolated to the given shape.
        If int, assume isotropic matrix. The default is None (original shape).
    output_res: float | tuple[float, float] | tuple[float, float, float] | None, optional
        Resolution of the output data, the data will be rescale to the given resolution.
        If scalar, assume isotropic resolution. The default is None (1mm iso).
    brainweb_dir : BrainWebDirType, optional
        Brainweb_directory to download the data.
        The default is None (~/.cache/brainweb).
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
        Brainweb segmentation.

    """
    assert ndim == 2 or ndim == 3, ValueError(f"Number of spatial dimensions (={ndim}) must be either 2 or 3.")
    assert subject in SUB_ID, ValueError(
        f"subject (={subject}) must be one of {SUB_ID}"
    )
    logger.debug(f"Get MRI data for subject {subject:02d}")
    
    # default params
    if shape is not None and np.isscalar(shape):
        shape = shape * np.ones(ndim, dtype=int)
    if shape is not None:
        assert len(shape) == ndim, ValueError("If shape is not None, it must be either a scalar or a ndim-length sequence.")
    if output_res is not None and np.isscalar(output_res):
        output_res = output_res * np.ones(ndim, dtype=int)
    if output_res is not None:
        assert len(output_res) == ndim, ValueError("If output_res is not None, it must be either or a scalar a ndim-length sequence.")
          
    # original resolution (0.5 mm iso)
    orig_res = 0.5 * np.ones(ndim)
        
    # get data
    with ssl_verification(verify=verify):
        data = brainweb_dl.get_mri(
            subject,
            "fuzzy",
            brainweb_dir=brainweb_dir,
            force=force,
        )

    # put tissue classes as leading axis
    data = data.transpose(-1, 0, 1, 2)
    data = np.flip(data, axis=-2)
    
    # select single slice
    if ndim == 2:
        center = int(data.shape[-3] // 2)
        data = data[:, center, :, :]
    
    # make sure it is contiguous
    data = np.ascontiguousarray(data)
    
    if shape is None and output_res is None:
        # normalize probability
        data = data / data.sum(axis=0)
        data = np.nan_to_num(data, posinf=0.0, neginf=0.0)
        return data
    elif output_res is None:
        output_res = 2 * orig_res # 1 mm iso
        
    # set prescription
    data = _prescription.set_prescription(data, orig_res, data.shape[-ndim:], output_res)
    
    # normalize probability
    data = data / data.sum(axis=0)
    data = np.nan_to_num(data, posinf=0.0, neginf=0.0)
    
    return data.astype(np.float32)
