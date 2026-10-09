"""2D binary shape matching modulo Sim(2) / Aff(2).

Pipeline (see README.md):
    canonicalize (closed form, removes 5 of 6 affine dof)
      -> residual O(2) search by polar FFT, keeping ALL peaks
      -> optional local refinement over the full group
      -> rescore by re-rasterized IoU.

Coordinate convention
---------------------
Everything is done in array-index coordinates: a point is ``(i, j)`` =
``(row, col)``.  Polygon vertex arrays are ``(N, 2)`` with column 0 = row.
This is a consistent relabelling of the usual (x, y) plane (it is a
reflection), so every group-theoretic statement below is unaffected; it just
avoids a transpose on every rasterization.
"""

from .config import Config, DEFAULT
from .raster import regular_polygon, rasterize, signed_distance
from .canonical import moments, sqrtm_sym, normalizer, canonicalize, CanonicalFrame
from .polar import polar_resample, rotation_correlation, find_peaks, Peak
from .match import Dictionary, match, MatchResult, iou
from .api import ShapeMatcher, ShapeMatch, match_shape

__all__ = [
    "Config", "DEFAULT",
    "regular_polygon", "rasterize", "signed_distance",
    "moments", "sqrtm_sym", "normalizer", "canonicalize", "CanonicalFrame",
    "polar_resample", "rotation_correlation", "find_peaks", "Peak",
    "Dictionary", "match", "MatchResult", "iou",
    "ShapeMatcher", "ShapeMatch", "match_shape",
]

__version__ = "0.1.0"
