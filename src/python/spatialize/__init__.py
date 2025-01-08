class SpatializeError(Exception):
    pass

from ._version import __version__
from ._util import GridSearchResult, EstimationResult, SingletonType

__all__ = ["gs", "gs.idw", "gs.esi", "gs.esi.aggfunction"]
