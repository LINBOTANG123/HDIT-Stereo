"""Residual O(2) search: polar resampling, circular FFT correlation, peak keeping.

After canonicalization the only freedom left is a rotation and a reflection.
A rotation is a cyclic shift in theta, and

    ||f - g_theta||^2 = ||f||^2 + ||g||^2 - 2 <f, g_theta>,

so all the theta dependence sits in the correlation and one FFT gives every
shift at once.  Maximizing correlation is exactly minimizing L2.

The stage returns a *set* of candidates, not an argmax.  A regular n-gon has
n near-equal correlation peaks (dihedral group D_n); once it is distorted those
peaks stop being equivalent and lead to genuinely different basins.  Measured
in the probe: the winning alignment came from a peak that was not the global
maximum in 21 of 36 queries, and in 40-57% across the full sweep (RESULTS.md
section 6).  Peak keeping is the feature that
makes the method work, not an ablation.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import DEFAULT


@dataclass
class Peak:
    index: int        # theta bin of the discrete maximum
    offset: float     # sub-bin correction in (-0.5, 0.5) from parabolic interpolation
    value: float      # correlation value
    rank: int         # 0 = global maximum

    def angle(self, ntheta):
        return 2 * np.pi * (self.index + self.offset) / ntheta


def polar_grid(cfg=DEFAULT):
    """Sampling coordinates and ring weights for the polar resample."""
    r = np.linspace(cfg.rmin, cfg.half, cfg.nr)
    t = np.arange(cfg.ntheta) * 2 * np.pi / cfg.ntheta
    R, T = np.meshgrid(r, t, indexing="ij")
    x, y = R * np.cos(T), R * np.sin(T)
    idx = np.stack([(x + cfg.half) / (2 * cfg.half) * (cfg.nc - 1),
                    (y + cfg.half) / (2 * cfg.half) * (cfg.nc - 1)],
                   axis=0).reshape(2, -1)
    # the polar Jacobian is r dr dtheta, so an unweighted sum over-counts the
    # centre; r**0.5 makes the sum of squares approximate the Cartesian L2 norm.
    return idx, (r ** cfg.weight_exp)[:, None]


def polar_resample(f, cfg=DEFAULT, cache=None):
    """Resample a canonical-frame field onto the (nr, ntheta) polar grid."""
    idx, w = polar_grid(cfg) if cache is None else cache
    v = ndimage.map_coordinates(f, idx, order=1, mode="nearest")
    return v.reshape(cfg.nr, cfg.ntheta) * w


def rotation_correlation(pq, ps):
    """Circular cross-correlation summed over rings.

    Returns `c` with `c[k] = sum_{r,theta} pq[r, theta+k] ps[r, theta]`, so bin
    `k` corresponds to sampling the query at `R(2 pi k / ntheta) u`.
    """
    ntheta = pq.shape[1]
    F = np.fft.rfft(pq, axis=1) * np.conj(np.fft.rfft(ps, axis=1))
    return np.fft.irfft(F, n=ntheta, axis=1).sum(0)


def rotation_correlation_from_spectra(Fq, Fs, ntheta):
    """Same, with the per-ring rffts already computed (dictionary side is cached)."""
    return np.fft.irfft(Fq * np.conj(Fs), n=ntheta, axis=1).sum(0)


def find_peaks(c, cfg=DEFAULT):
    """Circular NMS on the correlation curve, then parabolic sub-bin refinement.

    Keeps every local maximum within `rho * (max - min)` of the global max,
    subject to a minimum angular separation, capped at `pmax`.
    """
    n = len(c)
    lo, hi = float(c.min()), float(c.max())
    thresh = hi - cfg.rho * (hi - lo)

    prev, nxt = np.roll(c, 1), np.roll(c, -1)
    cand = np.nonzero((c > prev) & (c >= nxt) & (c >= thresh))[0]
    if len(cand) == 0:                       # degenerate / perfectly flat curve
        cand = np.array([int(np.argmax(c))])

    kept = []
    for i in cand[np.argsort(-c[cand])]:
        i = int(i)
        if all(min(abs(i - j.index), n - abs(i - j.index)) >= cfg.minsep for j in kept):
            kept.append(Peak(i, _parabolic(c, i, n), float(c[i]), len(kept)))
        if len(kept) >= cfg.pmax:
            break
    return kept


def _parabolic(c, i, n):
    """Sub-bin offset of the vertex of the parabola through c[i-1], c[i], c[i+1]."""
    a, b, d = c[(i - 1) % n], c[i], c[(i + 1) % n]
    den = a - 2 * b + d
    if den == 0:
        return 0.0
    return float(np.clip(0.5 * (a - d) / den, -0.5, 0.5))


def rot(theta):
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, -st], [st, ct]])
