"""Two-pool (Water + MT) BrainWeb phantom builder class."""

__all__ = ["FuzzyMTBrainwebPhantom", "CrispMTBrainwebPhantom", "NumericMTBrainwebPhantom"]

import numpy as np

from .. import _classes

from ._brainweb import FuzzyBrainwebPhantom, CrispBrainwebPhantom, NumericBrainwebPhantom

class FuzzyMTBrainwebPhantom(FuzzyBrainwebPhantom):
    """Fuzzy MT BrainWeb phantom builder."""
        
    def get_model(self, B0: float):
        """Initialize model.
        
        Parameters
        ----------
        B0, float
            Static field strength in [T].
        
        """        
        model_dict = _classes.tissue_map("mt-model")
        self._label = []
        self._properties = {"MVF": [], "T1w": [], "T2w": [], "k": []}
        for tissue in model_dict:
            self._label.append(tissue["Label"])
            self._properties["MVF"].append(tissue["MVF"])
            self._properties["T1w"].append(tissue["T1w"])
            self._properties["T2w"].append(tissue["T2w"])
            self._properties["k"].append(tissue["k"])
        
        # cast to array
        self._label = np.asarray(self._label, dtype=int)
        self._properties["MVF"] = np.asarray(self._properties["MVF"], dtype=np.float32)
        self._properties["T1w"] = np.asarray(self._properties["T1w"], dtype=np.float32)
        self._properties["T2w"] = np.asarray(self._properties["T2w"], dtype=np.float32)
        self._properties["k"] = np.asarray(self._properties["k"], dtype=np.float32)
        
    @property
    def MVF(self): # noqa
        return self._properties["MVF"]
    
    @property
    def T1(self): # noqa
        return self._properties["T1"]
    
    @property
    def T2(self): # noqa
        return self._properties["T2"]
        
    @property
    def k(self): # noqa
        return self._properties["k"]
    
    @property
    def properties(self): # noqa
        _properties = {
            "weight": self.MVF,
            "T1": self.T1,
            "T2": self.T2,
            "k": self.k
            }
        return _properties


class CrispMTBrainwebPhantom(FuzzyMTBrainwebPhantom, CrispBrainwebPhantom): # noqa
    """Crisp MT BrainWeb phantom builder."""
    pass 
   
            
class NumericMTBrainwebPhantom(FuzzyMTBrainwebPhantom, NumericBrainwebPhantom): # noqa
    """Numeric MT BrainWeb phantom builder."""
    pass
    