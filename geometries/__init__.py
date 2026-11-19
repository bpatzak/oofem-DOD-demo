"""Specimen geometries. Importing a module registers its geometry."""
from .base import GEOMETRIES, Geometry, register  # noqa: F401
from . import beam3pb  # noqa: F401,E402
from . import beam4pb  # noqa: F401,E402
