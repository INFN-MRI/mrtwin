"""Base BrainWeb phantom builder class."""

__all__ = ["BrainwebPhantom"]

import os
import numpy as np

from ..build import PhantomMixin
from .._utils import CacheDirType, get_mrtwin_dir

from ._segmentation import get_brainweb_segmentation

class BrainwebPhantom(PhantomMixin):
    """Base BrainWeb phantom builder."""
    
    def __init__(
            self,
            ndim: int,
            subject: int,
            shape: int | tuple[int, int] | tuple[int, int, int] = (200, 200, 200),
            output_res: float | tuple[float, float] | tuple[float, float, float] = (1., 1., 1.),
            cache: bool = True,
            cache_dir: CacheDirType = None,
            brainweb_dir: CacheDirType = None,
            force: bool = False,
            verify: bool = True,
            ):
        
        # get filename
        _fname = self.get_filename(ndim, subject, shape, output_res)
        
        # try to load segmentation
        self.segmentation, file_path = self.get_segmentation(
            _fname,
            ndim,
            subject,
            shape,
            output_res,
            cache,
            cache_dir,
            brainweb_dir,
            force,
            verify,
            )
        
        # cache the result
        if cache:
            self.cache(file_path, self.segmentation)
            
    
    def get_filename(
            self,
            ndim: int,
            subject: int, 
            shape: int | tuple[int, int] | tuple[int, int, int],
            resolution: float | tuple[float, float] | tuple[float, float, float],
            ):
        """
        Generate filename starting from FOV and matrix shape.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions. If ndim == 2, use a single slice
            (central axial slice).
        shape: int | tuple[int, int] | tuple[int, int, int]
            Shape of the output data, the data will be interpolated to the given shape.
            If int, assume isotropic matrix.
        resolution: float | tuple[float, float] | tuple[float, float, float]
            Resolution of the output data, the data will be rescale to the given resolution.
            If scalar, assume isotropic resolution.

        Returns
        -------
        str
            Filename for caching.

        """
        assert ndim == 2 or ndim == 3, ValueError(
            f"Number of spatial dimensions (={ndim}) must be either 2 or 3."
        )

        # default params
        if shape is not None and np.isscalar(shape):
            shape = shape * np.ones(ndim, dtype=int)
        if shape is not None:
            assert len(shape) == ndim, ValueError(
                "If shape is not None, it must be either a scalar or a ndim-length sequence."
            )
        if resolution is not None and np.isscalar(resolution):
            resolution = resolution * np.ones(ndim, dtype=int)
        if resolution is not None:
            assert len(resolution) == ndim, ValueError(
                "If resolution is not None, it must be either or a scalar a ndim-length sequence."
            )
        _fov = np.ceil(np.asarray(shape[-ndim:]) * np.asarray(resolution[-ndim:])).astype(int).tolist()
        return f"{self.__class__.__name__.lower()}{subject:02d}_{_fov}fov_{shape}mtx.npy"
    
    def get_segmentation(
            self,
            fname: str,
            ndim: int,
            subject: int,
            shape: int | tuple[int, int] | tuple[int, int, int],
            output_res: float | tuple[float, float] | tuple[float, float, float],
            cache: bool,
            cache_dir: CacheDirType,
            brainweb_dir: CacheDirType,
            force: bool,
            verify: bool,
            ):
        """
        Get fuzzy BrainWeb tissue segmentation.

        Parameters
        ----------
        fname : str
            Filename for caching.
        ndim : int
            Number of spatial dimensions. If ndim == 2, use a single slice
            (central axial slice).
        subject : int
            Subject id to download.
        shape: int | tuple[int, int] | tuple[int, int, int]
            Shape of the output data, the data will be interpolated to the given shape.
            If int, assume isotropic matrix.
        output_res: float | tuple[float, float] | tuple[float, float, float]
            Resolution of the output data, the data will be rescale to the given resolution.
            If scalar, assume isotropic resolution.
        cache : bool
            If True, cache the result.
        brainweb_dir : CacheDirType
            Directory for segmentation caching.
        brainweb_dir : CacheDirType
            Brainweb_directory to download the data.
        force : bool
            Force download even if the file already exists.
        verify : bool
            Enable SSL verification.
            DO NOT DISABLE (i.e., verify=False)IN PRODUCTION.

        Returns
        -------
        np.ndarray.
            Brainweb segmentation.
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
            segmentation = get_brainweb_segmentation(
                ndim, 
                subject,
                shape,
                output_res,
                brainweb_dir,
                force,
                verify
                )
                        
        return segmentation, file_path



    
    