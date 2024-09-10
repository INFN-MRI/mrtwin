"""Single-pool Open Science CBS phantom builder class."""

__all__ = ["NumericOSFPhantom"]

import os
from typing import Sequence

import numpy as np

from .. import _classes

from .._build import PhantomMixin
from .._utils import CacheDirType

from ._base import OSFPhantom

fudge_factor = 1.5  # MRF has lower value wrt SE


class NumericOSFPhantom(OSFPhantom, PhantomMixin):
    """Numeric OSF phantom builder."""

    def __init__(
        self,
        ndim: int,
        subject: int,
        shape: int | Sequence[int] = None,
        output_res: float | Sequence[float] = None,
        B0: float = 3.0,
        cache: bool = True,
        cache_dir: CacheDirType = None,
        osf_dir: CacheDirType = None,
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
            osf_dir,
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
        self._properties = {}
        self.properties["M0"] = self.maps[0]
        self.properties["T1"] = _classes.extrapolate_t1(self.maps[1], 3.0, B0)
        self._properties["T2"] = fudge_factor * self.maps[2]
        self._properties["T2s"] = _classes.extrapolate_t2star(
            self.maps[3],
            self._properties["T2"],
            3.0,
            B0,
        )
        self.properties["Chi"] = self.maps[4]

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
