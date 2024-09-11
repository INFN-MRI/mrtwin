"""Main MR-Twin API."""

__all__ = []

from ._brainweb import brainweb_phantom  # noqa
from ._osf import osf_phantom  # noqa
from ._shepplogan import shepplogan_phantom  # noqa

from ._fieldmap import b0field  # noqa
from ._fieldmap import b1field  # noqa
from ._fieldmap import sensmap  # noqa

from ._misc import rigid_motion  # noqa
from ._misc import generate_girf  # noqa

# Phantoms
__all__.append("brainweb_phantom")
__all__.append("osf_phantom")
__all__.append("shepplogan_phantom")

# Fields
__all__.append("b0field")
__all__.append("b1field")
__all__.append("sensmap")

# Miscellaneous
__all__.append("rigid_motion")
__all__.append("generate_girf")
