"""Single-pool BrainWeb phantom builder class."""

__all__ = ["FuzzyBrainwebPhantom", "CrispBrainwebPhantom", "NumericBrainwebPhantom"]

import os
from typing import Sequence

import numpy as np

from .. import _classes

from .._build import FuzzyPhantomMixin, CrispPhantomMixin, _fuzzy_to_crisp
from .._utils import CacheDirType

from ._base import BrainwebPhantom


class FuzzyBrainwebPhantom(BrainwebPhantom, FuzzyPhantomMixin):
    """Fuzzy BrainWeb phantom builder."""

    def __init__(
        self,
        ndim: int,
        subject: int,
        shape: int | Sequence[int] = None,
        output_res: float | Sequence[float] = None,
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
        self.get_model(B0)

    def get_model(self, B0: float):
        """Initialize model.

        Parameters
        ----------
        B0, float
            Static field strength in [T].

        """
        model_dict = _classes.tissue_map("single-pool")
        self._label = []
        self._properties = {"M0": [], "T1": [], "T2": [], "T2s": [], "Chi": []}
        for tissue in model_dict:
            self._label.append(tissue["Label"])
            self._properties["M0"].append(tissue["M0"])
            self._properties["T1"].append(_classes.get_t1(tissue, B0, 1.5))
            self._properties["T2"].append(tissue["T2"])
            self._properties["T2s"].append(_classes.get_t2star(tissue, B0, 1.5))
            self._properties["Chi"].append(tissue["Chi"])

        # cast to array
        self._label = np.asarray(self._label, dtype=int)
        self._properties["M0"] = np.asarray(self._properties["M0"], dtype=np.float32)
        self._properties["T1"] = np.asarray(self._properties["T1"], dtype=np.float32)
        self._properties["T2"] = np.asarray(self._properties["T2"], dtype=np.float32)
        self._properties["T2s"] = np.asarray(self._properties["T2s"], dtype=np.float32)
        self._properties["Chi"] = np.asarray(self._properties["Chi"], dtype=np.float32)

    @property
    def M0(self):  # noqa
        return self._properties["M0"]

    @property
    def T1(self):  # noqa
        return self._properties["T1"]

    @property
    def T2(self):  # noqa
        return self._properties["T2"]

    @property
    def T2s(self):  # noqa
        return self._properties["T2s"]

    @property
    def Chi(self):  # noqa
        return self._properties["Chi"]

    @property
    def properties(self):  # noqa
        return self._properties


class CrispBrainwebPhantom(FuzzyBrainwebPhantom, CrispPhantomMixin):
    """Crisp BrainWeb phantom builder."""

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
            array = _fuzzy_to_crisp(array)
            np.save(file_path, array)
            self.as_crisp(copy=False)


class NumericBrainwebPhantom(CrispBrainwebPhantom):
    """Numeric BrainWeb phantom builder."""

    def __init__(
        self,
        ndim: int,
        subject: int,
        shape: int | Sequence[int] = None,
        output_res: float | Sequence[float] = None,
        B0: float = 1.5,
        cache: bool = True,
        cache_dir: CacheDirType = None,
        brainweb_dir: CacheDirType = None,
        force: bool = False,
        verify: bool = True,
    ):

        super().__init__(
            ndim,
            subject,
            shape,
            output_res,
            B0,
            cache,
            cache_dir,
            brainweb_dir,
            force,
            verify,
        )

        self.as_numeric(copy=False)
