"""Phantom mixins."""

__all__ = ["PhantomMixin", "CrispPhantomMixin", "FuzzyPhantomMixin"]

import os

from copy import deepcopy

import numpy as np

class PhantomMixin:
    """Base phantom mixin."""
    
    def cache(self, file_path: str, array: np.ndarray):
        """
        Cache an array for fast retrieval.

        Parameters
        ----------
        file_path : str
            Path on disk to cached array.
        array : np.ndarray
            Array to be cached.

        """
        if os.path.exists(file_path) is False:
            np.save(file_path, array)
    
    
class CrispPhantomMixin(PhantomMixin):
    """Crisp phantom mixin."""
    
    def __getitem__(self, idx): # noqa
        if self.segmentation is not None:
            return self.segmentation.__getitem__(idx)
        else:
            return None
    
    @property
    def shape(self): # noqa
        if self.segmentation is not None:
            return self.segmentation.shape
        else:
            return None

    def as_numeric(self, copy: bool = True): # noqa
        """Convert crisp phantom into numeric phantom."""
        if copy:
            out = deepcopy(self)
        else:
            out = self
        for param in out._properties.keys():
            param_map = np.zeros(out.segmentation.shape, dtype=np.float32)
            for idx in range(len(out._properties[param])):
                param_map += param[idx] * (out.segmentation == out._label[idx])
            out._properties[param] = param_map
            
        return  out


class FuzzyPhantomMixin(PhantomMixin):
    """Fuzzy phantom mixin."""
    
    def __getitem__(self, idx): # noqa
        if self.segmentation is not None:
            return self.segmentation.__getitem__(idx)
        else:
            return None
    
    @property
    def shape(self): # noqa
        if self.segmentation is not None:
            return self.segmentation.shape
        else:
            return None

    def as_crisp(self, copy: bool = True):
        """Convert fuzzy phantom into crisp phantom."""
        if copy:
            out = deepcopy(self)
        else:
            out = self
        out.segmentation = _fuzzy_to_crisp(out.segmentation)
        return out
        
    def as_numeric(self, copy: bool = True):
        """Convert fuzzy phantom into numeric phantom."""
        if self.segmentation.ndim != self._ndim:
            out = self.as_crisp(copy)
        else:
            if copy:
                out = deepcopy(self)
            else:
                out = self
        
        # build tissue maps
        for param in out._properties.keys():
            param_map = np.zeros(out.segmentation.shape, dtype=np.float32)
            for idx in range(len(out._properties[param])):
                param_map += out._properties[param][idx] * (out.segmentation == out._label[idx])
            out._properties[param] = param_map
            
        # erase segmentation
        out.segmentation = None
            
        return  out
        

def _fuzzy_to_crisp(fuzzy_segmentation: np.ndarray) -> np.ndarray:
    """
    Convert fuzzy segmentation into crisp segmentation.
    
    Conversion is performed assigning each voxel to the class with
    highest probability.

    Parameters
    ----------
    fuzzy_segmentation : np.ndarray
        Input fuzzy segmentation of shape (nclasses, *shape).

    Returns
    -------
    np.ndarray
        Output crisp segmentation of shape (*shape).

    """
    crisp_segmentation = np.argmax(fuzzy_segmentation, axis=0)
    return crisp_segmentation.astype(int)