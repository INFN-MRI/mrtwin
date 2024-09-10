"""Base Open Science CBS phantom builder class."""

__all__ = ["OSFPhantom"]

import os
import numpy as np

from typing import Sequence

from ..build import PhantomMixin
from .._utils import CacheDirType, get_mrtwin_dir

from ._maps import get_osf_maps

BIG_RES_SHAPE = (454, 544, 454)
BIG_RES_MM = (0.5, 0.5, 0.5)


class OSFPhantom(PhantomMixin):
    """Base OSF phantom builder."""

    def __init__(
        self,
        ndim: int,
        subject: int,
        shape: int | Sequence[int] | None = None,
        output_res: float | Sequence[float] | None = None,
        cache: bool = True,
        cache_dir: CacheDirType = None,
        osf_dir: CacheDirType = None,
        force: bool = False,
        verify: bool = True,
    ):
        # keep dim
        self._ndim = ndim

        # default fov, resolution
        shape, output_res = self._default_prescription(ndim, shape, output_res)

        # get filename
        _fname = self.get_filename(ndim, subject, shape, output_res)

        # try to load parameter maps
        self.maps, file_path = self.get_maps(
            _fname,
            ndim,
            subject,
            shape,
            output_res,
            cache,
            cache_dir,
            osf_dir,
            force,
            verify,
        )

        # cache the result
        if cache:
            self.cache(file_path, self.maps)

    def _default_prescription(
        self,
        ndim: int,
        output_shape: int | Sequence[int] | None,
        output_res: float | Sequence[float] | None,
    ):
        # original shape and resolution
        orig_shape = np.asarray(BIG_RES_SHAPE)[-ndim:]
        orig_res = np.asarray(BIG_RES_MM)[-ndim:]
        orig_fov = orig_shape * orig_res

        # default shape
        if output_shape is None:
            output_shape = ndim * [256]
        if output_shape is not None and np.isscalar(output_shape):
            output_shape = ndim * [output_shape]
        if output_shape is not None:
            assert len(output_shape) == ndim, ValueError(
                "If shape is not None, it must be either a scalar or a ndim-length sequence."
            )
        output_shape = np.asarray(output_shape)

        # default resolution
        if output_res is not None and np.isscalar(output_res):
            output_res = ndim * [output_res]
        if output_res is not None:
            assert len(output_res) == ndim, ValueError(
                "If output_res is not None, it must be either or a scalar a ndim-length sequence."
            )
        if output_res is None:
            output_res = [max(orig_fov) / max(output_shape)] * ndim
        output_res = np.asarray(output_res)

        return output_shape, output_res

    def __repr__(self):  # noqa
        ptype = "Dense"
        msg = f"{ptype} OSF phantom with following properties:\n"
        msg += f"Number of spatial dimensions: {self._ndim}\n"
        msg += f"Tissue properties: {self._properties.keys()}\n"
        _shape = list(self._properties.values())[0].shape[-self._ndim :]
        msg += f"Matrix size: {_shape}\n"

        return msg

    def get_filename(
        self,
        ndim: int,
        subject: int,
        shape: int | Sequence[int],
        resolution: float | Sequence[float],
    ):
        """
        Generate filename starting from FOV and matrix shape.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions. If ndim == 2, use a single slice
            (central axial slice).
        shape: int | Sequence[int]
            Shape of the output data, the data will be interpolated to the given shape.
            If int, assume isotropic matrix.
        resolution: float | Sequence[float]
            Resolution of the output data, the data will be rescale to the given resolution.
            If scalar, assume isotropic resolution.

        Returns
        -------
        str
            Filename for caching.

        """
        _fov = np.ceil(shape * resolution).astype(int).tolist()

        fov_str = [str(ri) for ri in _fov]
        shape_str = [str(di) for di in shape.tolist()]

        fov_str = "x".join(tuple(fov_str))
        shape_str = "x".join(tuple(shape_str))

        return f"{self.__class__.__name__.lower()}{subject:02d}_{fov_str}fov_{shape_str}mtx.npy"

    def get_maps(
        self,
        fname: str,
        ndim: int,
        subject: int,
        shape: int | Sequence[int],
        output_res: float | Sequence[float],
        cache: bool,
        cache_dir: CacheDirType,
        osf_dir: CacheDirType,
        force: bool,
        verify: bool,
    ):
        """
        Get OSF parameter maps.

        Parameters
        ----------
        fname : str
            Filename for caching.
        ndim : int
            Number of spatial dimensions. If ndim == 2, use a single slice
            (central axial slice).
        subject : int
            Subject id to download.
        shape: int | Sequence[int]
            Shape of the output data, the data will be interpolated to the given shape.
            If int, assume isotropic matrix.
        output_res: float | Sequence[float]
            Resolution of the output data, the data will be rescale to the given resolution.
            If scalar, assume isotropic resolution.
        cache : bool
            If True, cache the result.
        cache_dir : CacheDirType
            Directory for maps caching.
        osf_dir : CacheDirType
            osf_directory to download the data.
        force : bool
            Force download even if the file already exists.
        verify : bool
            Enable SSL verification.
            DO NOT DISABLE (i.e., verify=False)IN PRODUCTION.

        Returns
        -------
        np.ndarray.
            OSF parameter maps.
        file_path : str
            Path on disk to generated maps for caching.

        """
        # get base directory
        cache_dir = get_mrtwin_dir(cache_dir)

        # get file path
        file_path = os.path.join(cache_dir, fname)

        # try to load
        if os.path.exists(file_path) and not (force):
            return np.load(file_path), file_path
        else:
            maps = get_osf_maps(
                ndim, subject, shape, output_res, osf_dir, force, verify
            )

        return maps, file_path
