"""Main MR-Twin API."""

__all__ = []

from ._brainweb import brainweb_phantom
from ._osf import osf_phantom
from ._shepplogan import shepplogan_phantom

from ._fieldmap import b0field
from ._fieldmap import b1field
from ._fieldmap import sensmap

# Phantoms
__all__.append("brainweb_phantom")
__all__.append("osf_phantom")
__all__.append("shepplogan_phantom")

# Fields
__all__.append("b0field")
__all__.append("b1field")
__all__.append("sensmap")
