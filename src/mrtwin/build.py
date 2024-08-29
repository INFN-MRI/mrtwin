"""Phantom mixins."""

__all__ = ["PhantomMixin", "CrispPhantomMixin", "FuzzyPhantomMixin"]

import os
import numpy as np

class PhantomMixin:
    """Base phantom mixin."""
    
    def cache(self, file_path: str, array: np.ndarray):
        """
        Cache an array for fast retrieval.

        Parameters
        ----------
        file_path : str
            DESCRIPTION.
        array : np.ndarray
            DESCRIPTION.

        """
        if os.path.exists(file_path) is False:
            np.save(file_path, array)
    
class CrispPhantomMixin(PhantomMixin):
    """Crisp phantom mixin."""

    @classmethod
    def from_numeric(cls):
        pass

    def as_numeric(self):
        pass


class FuzzyPhantomMixin(PhantomMixin):
    """Fuzzy phantom mixin."""

    def as_crisp(self):
        pass

    def as_numeric(self):
        pass

