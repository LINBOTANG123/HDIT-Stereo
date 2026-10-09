"""Local refinement over the full group.

Demoted to optional.  The probe's eps=0.06 ablation showed canonicalization +
rotation search alone matches the refined pipeline to within the trial-to-trial
spread: whitening is already O(eps)-accurate, so the residual affine error
costs only *second* order in IoU.  Kept behind `Config.refine` for the regimes
where it might pay (large eps, large cond(A), elongated dictionary items).

Known bias, stated up front: `f_Q(W x + t)` is a reparametrized distance field,
not the distance field of the warped query, so under strongly anisotropic `W`
the surrogate is biased.  Mitigated by the small clamp `tau` and, decisively,
by never ranking on the surrogate -- ranking always uses re-rasterized IoU.
"""

import numpy as np
from scipy import ndimage, optimize

from .polar import rot


def _grid(n, half):
    g = np.linspace(-half, half, n)
    U, V = np.meshgrid(g, g, indexing="ij")
    return np.stack([U.ravel(), V.ravel()], axis=0)


def _sample(field, uv, W, t, half):
    n = field.shape[0]
    xy = W @ uv + np.asarray(t, float)[:, None]
    idx = np.stack([(xy[0] + half) / (2 * half) * (n - 1),
                    (xy[1] + half) / (2 * half) * (n - 1)], axis=0)
    return ndimage.map_coordinates(field, idx, order=1, mode="nearest")


def _unpack(p, R0, group):
    if group == "affine":
        return (np.eye(2) + p[:4].reshape(2, 2)) @ R0, p[4:6]
    return (1.0 + p[0]) * rot(p[1]) @ R0, p[2:4]


def refine(fq, fs, R0, group="affine", cfg=None, half=3.5):
    """Multi-resolution L-BFGS-B on the clamped-SDT L2 surrogate.

    `fq`, `fs` are canonical-frame signed distance fields; `R0` is the
    initialization from one kept correlation peak.  Returns (W, t, residual).
    """
    from .config import DEFAULT
    cfg = cfg or DEFAULT
    nc = fq.shape[0]
    levels = sorted({min(L, nc) for L in cfg.refine_levels} | {nc})

    p = np.zeros(6 if group == "affine" else 4)
    fun = np.inf
    for L in levels:
        z = L / nc
        fqL = fq if L == nc else ndimage.zoom(fq, z, order=1)
        fsL = (fs if L == nc else ndimage.zoom(fs, z, order=1)).ravel()
        uv = _grid(fqL.shape[0], half)

        def obj(q, fqL=fqL, fsL=fsL, uv=uv):
            W, t = _unpack(q, R0, group)
            return float(((_sample(fqL, uv, W, t, half) - fsL) ** 2).mean())

        r = optimize.minimize(obj, p, method="L-BFGS-B",
                              options=dict(maxiter=cfg.refine_maxiter, eps=1e-3))
        p, fun = r.x, float(r.fun)

    W, t = _unpack(p, R0, group)
    return W, t, fun
