"""Base Shepp-Logan phantom builder class."""

__all__ = ["SheppLoganPhantom"]

import os
import numpy as np

from typing import Sequence

from ..build import PhantomMixin
from .._utils import CacheDirType, get_mrtwin_dir

from ._segmentation import get_shepp_logan


class SheppLoganPhantom(PhantomMixin):
    """Base Shepp-Logan phantom builder."""

    def __init__(
        self,
        ndim: int,
        shape: int | Sequence[int] | None = None,
        cache: bool = True,
        cache_dir: CacheDirType = None,
    ):
        # keep dim
        self._ndim = ndim

        # default shape
        if np.isscalar(shape):
            shape = [shape] * ndim
        shape = np.asarray(shape)

        # get filename
        _fname = self.get_filename(ndim, shape)

        # try to load segmentation
        self.segmentation, file_path = self.get_segmentation(
            _fname,
            ndim,
            shape,
            cache,
            cache_dir,
        )

        # cache the result
        if cache:
            self.cache(file_path, self.segmentation)

    def __repr__(self):  # noqa
        if self.segmentation is None:
            ptype = "Dense"
        elif len(self.segmentation.shape) == self._ndim:
            ptype = "Crisp"
        msg = f"{ptype} Shepp-Logan phantom with following properties:\n"
        msg += f"Number of spatial dimensions: {self._ndim}\n"
        msg += f"Tissue properties: {self._properties.keys()}\n"
        if self.segmentation is not None:
            _shape = self.shape[-self._ndim :]
        else:
            _shape = list(self._properties.values())[0].shape[-self._ndim :]
        msg += f"Matrix size: {_shape}\n"
        if self.segmentation is not None and len(self.segmentation.shape) != self._ndim:
            msg += f"Number of tissue classes: {self.segmentation.shape[0]}\n"
        return msg

    def get_filename(
        self,
        ndim: int,
        shape: int | Sequence[int],
    ):
        """
        Generate filename starting from matrix shape.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions. If ndim == 2, use a single slice
            (central axial slice).
        shape: int | Sequence[int]
            Shape of the output data, the data will be interpolated to the given shape.
            If int, assume isotropic matrix.

        Returns
        -------
        str
            Filename for caching.

        """
        shape_str = [str(di) for di in shape.tolist()]
        shape_str = "x".join(tuple(shape_str))

        return f"{self.__class__.__name__.lower()}_{shape_str}mtx.npy"

    def get_segmentation(
        self,
        fname: str,
        ndim: int,
        shape: int | Sequence[int],
        cache: bool,
        cache_dir: CacheDirType,
    ):
        """
        Get crisp Shepp-Logan tissue segmentation.

        Parameters
        ----------
        fname : str
            Filename for caching.
        ndim : int
            Number of spatial dimensions. If ndim == 2, use a single slice
            (central axial slice).
        shape: int | Sequence[int]
            Shape of the output data, the data will be interpolated to the given shape.
            If int, assume isotropic matrix.
        cache : bool
            If True, cache the result.
        cache_dir : CacheDirType
            Directory for segmentation caching.

        Returns
        -------
        np.ndarray.
            Shepp-Logan segmentation.
        file_path : str
            Path on disk to generated segmentation for caching.

        """
        # get base directory
        cache_dir = get_mrtwin_dir(cache_dir)

        # get file path
        file_path = os.path.join(cache_dir, fname)

        # try to load
        if os.path.exists(file_path):
            return np.load(file_path), file_path
        else:
            segmentation = get_shepp_logan(ndim, shape)

        return segmentation, file_path

    def __array__(self):  # noqa
        # This method tells NumPy how to convert the object to an array
        return self.segmentation
