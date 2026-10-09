"""Polygon generation, supersampled rasterization, signed distance transform."""

import numpy as np
from scipy import ndimage
from skimage.draw import polygon2mask

CANVAS = 256
RADIUS = 90.0


def regular_polygon(n, R=RADIUS, phase=0.0, center=(CANVAS / 2, CANVAS / 2)):
    """Vertices of a regular n-gon, circumradius R, first vertex at angle `phase`."""
    t = np.arange(n) * 2 * np.pi / n + phase
    c = np.asarray(center, float)
    return np.stack([c[0] + R * np.cos(t), c[1] + R * np.sin(t)], axis=1)


def rasterize(verts, N=CANVAS, ss=4):
    """Rasterize a simple polygon to an N x N bool mask.

    Supersample by `ss`, area-average, threshold at 0.5.  Naive scan conversion
    has an orientation-dependent area bias that would leak a spurious rotation
    signal into the correlation stage; area-averaging kills it to
    O(1/ss^2).
    """
    m = polygon2mask((N * ss, N * ss), np.asarray(verts, float) * ss)
    return m.reshape(N, ss, N, ss).mean(axis=(1, 3)) >= 0.5


def signed_distance(mask, spacing=1.0, clamp=None):
    """Signed distance field: negative inside, positive outside.

    Must be computed in whatever frame it will be compared in -- warping a
    distance field is not the distance field of the warped shape under an
    anisotropic map.
    """
    mask = np.asarray(mask, bool)
    d = (ndimage.distance_transform_edt(~mask) * spacing
         - ndimage.distance_transform_edt(mask) * spacing)
    return np.clip(d, -clamp, clamp) if clamp is not None else d
