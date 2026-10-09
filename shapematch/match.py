"""Top-level matcher: canonicalize -> polar FFT over all peaks -> rescore by IoU."""

from dataclasses import dataclass, field

import numpy as np

from .config import DEFAULT
from .canonical import canonicalize, warp_mask
from .polar import polar_grid, polar_resample, rotation_correlation_from_spectra, find_peaks, rot
from .raster import signed_distance
from .refine import refine as _refine

MIRROR = np.diag([1.0, -1.0])   # index-axis-1 flip on the symmetric canonical grid


def iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return float((a & b).sum()) / max(int((a | b).sum()), 1)


def asym_dist(a, b):
    """Symmetric-difference area normalized by the *template* area.

    Asymmetric, but a truer "how much of the template is explained" than 1-IoU.
    Reported alongside, never used for ranking.
    """
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return float((a ^ b).sum()) / max(int(b.sum()), 1)


@dataclass
class MatchResult:
    name: object
    iou: float
    dist: float               # 1 - IoU, the ranking score
    asym: float
    W: np.ndarray             # canonical dict -> canonical query linear part (parity folded in)
    t: np.ndarray
    parity: int               # +1 or -1 (reflection)
    peak_rank: int            # 0 if the winner came from the global correlation max
    n_peaks: int              # candidates evaluated across both parities
    A_img: np.ndarray = field(default=None)   # image-space dict -> query linear part
    b_img: np.ndarray = field(default=None)
    aligned: np.ndarray = field(default=None)  # query re-rasterized on the item's canonical grid


class _Prepared:
    """Canonical frame + clamped SDT + cached polar spectrum for one mask."""

    def __init__(self, mask, group, cfg, mirror=False):
        self.frame = canonicalize(mask, group, cfg)
        cmask = self.frame.mask
        h = 2 * cfg.half / (cfg.nc - 1)
        sdt = signed_distance(cmask, spacing=h, clamp=cfg.tau)
        if mirror:
            cmask, sdt = cmask[:, ::-1], sdt[:, ::-1]
        self.cmask, self.sdt = cmask, sdt
        self.polar = polar_resample(sdt, cfg, cache=polar_grid(cfg))
        self.spec = np.fft.rfft(self.polar, axis=1)


class Dictionary:
    """Precomputed dictionary side, for one (group, cfg) pair."""

    def __init__(self, masks, group="affine", cfg=DEFAULT):
        self.group, self.cfg = group, cfg
        self.names = list(masks.keys())
        self.masks = dict(masks)
        self.items = {k: _Prepared(v, group, cfg) for k, v in masks.items()}


def match(query_mask, dictionary, group=None, cfg=None, top=None):
    """Rank dictionary items against `query_mask`.  Returns a list sorted by 1-IoU."""
    cfg = cfg or dictionary.cfg
    group = group or dictionary.group

    # the query is prepared once and reused for every dictionary item
    parities = (+1, -1) if cfg.reflection else (+1,)
    q = {p: _Prepared(query_mask, group, cfg, mirror=(p < 0)) for p in parities}

    out = []
    for name, s in dictionary.items.items():
        cands = []
        n_peaks = 0
        for parity in parities:
            qp = q[parity]
            c = rotation_correlation_from_spectra(qp.spec, s.spec, cfg.ntheta)
            peaks = find_peaks(c, cfg)
            n_peaks += len(peaks)
            for pk in peaks:
                R0 = rot(pk.angle(cfg.ntheta))
                if cfg.refine and cfg.refine_scope == "all":
                    W, t, _ = _refine(qp.sdt, s.sdt, R0, group, cfg, cfg.half)
                else:
                    W, t = R0, np.zeros(2)
                w = warp_mask(qp.cmask, W, t, cfg)
                cands.append([iou(w, s.cmask), w, W, t, parity, pk.rank, R0])

        best = max(cands, key=lambda z: z[0])
        if cfg.refine and cfg.refine_scope == "best":
            qp = q[best[4]]
            W, t, _ = _refine(qp.sdt, s.sdt, best[6], group, cfg, cfg.half)
            w = warp_mask(qp.cmask, W, t, cfg)
            v = iou(w, s.cmask)
            if v > best[0]:
                best = [v, w, W, t, best[4], best[5], best[6]]

        v, w_best, W, t, parity, rank, _ = best
        av = asym_dist(w_best, s.cmask)
        P = MIRROR if parity < 0 else np.eye(2)
        Weff, teff = P @ W, P @ t
        Ss_inv = np.linalg.inv(s.frame.S)
        A_img = q[+1].frame.S @ Weff @ Ss_inv
        b_img = q[+1].frame.mu - A_img @ s.frame.mu + q[+1].frame.S @ teff
        out.append(MatchResult(name, v, 1.0 - v, av, Weff, teff, parity,
                               rank, n_peaks, A_img, b_img, w_best))

    out.sort(key=lambda r: r.dist)
    return out[:top] if top else out


def distance_matrix(masks, group="affine", cfg=DEFAULT):
    """Pairwise `min over G of 1 - IoU`, as percentages.  Step 0 regression test."""
    d = Dictionary(masks, group, cfg)
    names = d.names
    M = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        res = {r.name: r.dist for r in match(masks[a], d)}
        for j, b in enumerate(names):
            M[i, j] = 100 * res[b]
    return names, M
