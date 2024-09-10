"""Single-pool Shepp-Logan phantom builder class."""

__all__ = ["CrispSheppLoganPhantom", "NumericSheppLoganPhantom"]

from typing import Sequence

import numpy as np

from .. import _classes

from .._build import CrispPhantomMixin
from .._utils import CacheDirType

from ._base import SheppLoganPhantom


class CrispSheppLoganPhantom(SheppLoganPhantom, CrispPhantomMixin):
    """Fuzzy Shepp-Logan phantom builder."""

    def __init__(
        self,
        ndim: int,
        shape: int | Sequence[int] = None,
        B0: float = 1.5,
        cache: bool = True,
        cache_dir: CacheDirType = None,
    ):

        # initialize segmentation
        super().__init__(
            ndim,
            shape,
            cache,
            cache_dir,
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


class NumericSheppLoganPhantom(CrispSheppLoganPhantom):
    """Numeric Shepp-Logan phantom builder."""

    def __init__(
        self,
        ndim: int,
        shape: int | Sequence[int] = None,
        B0: float = 1.5,
        cache: bool = True,
        cache_dir: CacheDirType = None,
    ):

        super().__init__(
            ndim,
            shape,
            B0,
            cache,
            cache_dir,
        )

        self.as_numeric(copy=False)
