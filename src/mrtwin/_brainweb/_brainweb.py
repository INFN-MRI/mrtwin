"""Fuzzy BrainWeb phantom builder class."""

__all__ = ["FuzzyBrainwebPhantom"]

from ..build import FuzzyPhantomMixin
from .._utils import CacheDirType

from ._base import BrainwebPhantom

class FuzzyBrainwebPhantom(BrainwebPhantom, FuzzyPhantomMixin):
    """Fuzzy BrainWeb phantom builder."""
    
    def __init__(
            self,
            ndim: int,
            subject: int,
            shape: int | tuple[int, int] | tuple[int, int, int] = (200, 200, 200),
            output_res: float | tuple[float, float] | tuple[float, float, float] = (1., 1., 1.),
            B0: float = 1.5,
            cache: bool = True,
            cache_dir: CacheDirType = None,
            brainweb_dir: CacheDirType = None,
            force: bool = False,
            verify: bool = True,
            ):
        
        # initialize segmentation
        super().__init__(
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
        
        # initialize model
        


    
    