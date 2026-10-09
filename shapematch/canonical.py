"""Moment canonicalization: the closed-form part of the alignment.

Affine.  With centroid `mu` and second central moment `Sigma`, the map
`x -> Sigma^{-1/2}(x - mu)` sends the region to one with identity covariance.
If `y = Ax + b` then `Sigma_y = A Sigma_x A^T`, so
`Sigma_y^{-1/2} A Sigma_x^{1/2}` is orthogonal:

    the canonical frames of two affinely related shapes differ by exactly
    an element of O(2).

Five of six affine dof are therefore removed exactly, in closed form.  What is
left is one angle plus a reflection, handled in `polar.py`.

Similarity.  Identical machinery with `Sigma^{1/2}` replaced by the isotropic
`sqrt(trace(Sigma)/2) * I`; the residual group is again O(2).  That is the
whole of the Sim/Aff switch at this stage.

Caveat: this is an *invariant* construction, not a minimizer of
the matching objective.  It is an exact initializer, nothing more.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import DEFAULT


@dataclass
class CanonicalFrame:
    mask: np.ndarray   # bool, nc x nc, the shape resampled in canonical coordinates
    mu: np.ndarray     # (2,) centroid in image coordinates
    S: np.ndarray      # (2,2) canonical -> image linear part:  x_img = mu + S @ u
    area: int          # pixel count of the source mask


def moments(mask):
    """Centroid and second central moment of the set of on-pixels."""
    p = np.stack(np.nonzero(np.asarray(mask, bool)), axis=1).astype(float)
    if len(p) < 3:
        raise ValueError("mask is empty or degenerate")
    mu = p.mean(0)
    C = np.cov((p - mu).T, bias=True)
    return mu, C, len(p)


def sqrtm_sym(C):
    """Symmetric square root and inverse square root of an SPD 2x2."""
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-12)
    return (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T


def normalizer(C, group="affine"):
    """The matrix `S` with `x_img = mu + S u`.

    affine     -> S = Sigma^{1/2}   (removes scale, anisotropy and shear)
    similarity -> S = s I,  s = sqrt(trace(Sigma)/2)   (removes isotropic scale only)
    """
    if group == "affine":
        S, _ = sqrtm_sym(C)
        return S
    if group in ("similarity", "sim"):
        return np.eye(2) * np.sqrt(np.trace(C) / 2.0)
    raise ValueError(f"unknown group {group!r}")


def canonicalize(mask, group="affine", cfg=DEFAULT):
    """Resample `mask` onto the canonical grid [-half, half]^2 at nc x nc."""
    mu, C, area = moments(mask)
    S = normalizer(C, group)
    u = _grid(cfg)
    xy = mu[:, None] + S @ u
    out = ndimage.map_coordinates(np.asarray(mask, np.float32), xy,
                                  order=1, mode="constant")
    return CanonicalFrame(out.reshape(cfg.nc, cfg.nc) >= 0.5, mu, S, area)


def _grid(cfg):
    """(2, nc*nc) stack of canonical-frame coordinates, row-major."""
    g = np.linspace(-cfg.half, cfg.half, cfg.nc)
    U, V = np.meshgrid(g, g, indexing="ij")
    return np.stack([U.ravel(), V.ravel()], axis=0)


def sample_canonical(field, W, t, cfg=DEFAULT, order=1, mode="nearest"):
    """Sample `field` (nc x nc, canonical frame) at `W u + t` for every grid point u."""
    xy = W @ _grid(cfg) + np.asarray(t, float)[:, None]
    idx = np.stack([(xy[0] + cfg.half) / (2 * cfg.half) * (cfg.nc - 1),
                    (xy[1] + cfg.half) / (2 * cfg.half) * (cfg.nc - 1)], axis=0)
    return ndimage.map_coordinates(field, idx, order=order,
                                   mode=mode).reshape(cfg.nc, cfg.nc)


def warp_mask(cmask, W, t, cfg=DEFAULT):
    """Re-rasterize a canonical-frame mask under `u -> W u + t`."""
    return sample_canonical(np.asarray(cmask, np.float32), W, t, cfg,
                            order=1, mode="constant") >= 0.5
