"""Two-pool (I/E W + MW) Shepp-Logan phantom builder class."""

__all__ = ["CrispMWSheppLoganPhantom", "NumericMWSheppLoganPhantom"]

import numpy as np

from .. import _classes

from ._shepplogan import CrispSheppLoganPhantom, NumericSheppLoganPhantom


class CrispMWSheppLoganPhantom(CrispSheppLoganPhantom):
    """Crisp MW Shepp-Logan phantom builder."""

    def get_model(self, B0: float):
        """Initialize model.

        Parameters
        ----------
        B0, float
            Static field strength in [T].

        """
        model_dict = _classes.tissue_map("mw-model")
        self._label = []
        self._properties = {
            "MWF": [],
            "T1w": [],
            "T1m": [],
            "T2w": [],
            "T2m": [],
            "k": [],
            "chemshift": [],
        }
        for tissue in model_dict:
            self._label.append(tissue["Label"])
            self._properties["MWF"].append(tissue["MWF"])
            self._properties["T1w"].append(tissue["T1w"])
            self._properties["T1m"].append(tissue["T1m"])
            self._properties["T2w"].append(tissue["T2w"])
            self._properties["T2m"].append(tissue["T2m"])
            self._properties["k"].append(tissue["k"])
            self._properties["chemshift"].append(tissue["chemshift"])

        # cast to array
        self._label = np.asarray(self._label, dtype=int)
        self._properties["MWF"] = np.asarray(self._properties["MWF"], dtype=np.float32)
        self._properties["T1w"] = np.asarray(self._properties["T1w"], dtype=np.float32)
        self._properties["T1m"] = np.asarray(self._properties["T1m"], dtype=np.float32)
        self._properties["T2w"] = np.asarray(self._properties["T2w"], dtype=np.float32)
        self._properties["T2m"] = np.asarray(self._properties["T2m"], dtype=np.float32)
        self._properties["k"] = np.asarray(self._properties["k"], dtype=np.float32)
        self._properties["chemshift"] = np.asarray(
            self._properties["chemshift"], dtype=np.float32
        )

    @property
    def MWF(self):  # noqa
        return self._properties["MWF"]

    @property
    def T1(self):  # noqa
        return np.stack((self._properties["T1w"], self._properties["T1m"]))

    @property
    def T2(self):  # noqa
        return np.stack((self._properties["T2w"], self._properties["T2m"]))

    @property
    def k(self):  # noqa
        return self._properties["k"]

    @property
    def chemshift(self):  # noqa
        return self._properties["chemshift"]

    @property
    def properties(self):  # noqa
        _properties = {
            "weight": self.MWF,
            "T1": self.T1,
            "T2": self.T2,
            "k": self.k,
            "chemshift": self.chemshift,
        }
        return _properties


class NumericMWSheppLoganPhantom(
    CrispMWSheppLoganPhantom, NumericSheppLoganPhantom
):  # noqa
    """Numeric MW Shepp-Logan phantom builder."""

    pass
