"""Evaluation metrics that need group structure."""

import numpy as np


def rotation_part(M):
    """Orthogonal factor of the polar decomposition M = Q P."""
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def angular_error(A_rec, A_true, n):
    """Recovered-rotation error, reduced modulo the target's symmetry.

    A regular n-gon has symmetry group D_n, so the absolute rotation is
    unrecoverable by construction: only the residue modulo 2*pi/n carries
    information.  If the residual is a reflection,
    the relevant period is pi/n -- the n-gon's mirror axes sit at k*pi/n.

    Returns a signed angle in radians, in [-period/2, period/2].
    """
    Q = rotation_part(np.linalg.solve(np.asarray(A_true, float), np.asarray(A_rec, float)))
    if np.linalg.det(Q) > 0:
        th, period = np.arctan2(Q[1, 0], Q[0, 0]), 2 * np.pi / n
    else:
        th, period = 0.5 * np.arctan2(Q[1, 0], Q[0, 0]), np.pi / n
    return float(th - period * np.round(th / period))


def margin(dists):
    """(2nd best - best) / best, over a dict or list of distances.

    More diagnostic than accuracy: it degrades continuously and shows the
    failure approaching before any label flips.
    """
    d = np.sort(np.asarray(list(dists.values()) if isinstance(dists, dict) else dists, float))
    if len(d) < 2 or d[0] <= 0:
        return np.inf
    return float((d[1] - d[0]) / d[0])
