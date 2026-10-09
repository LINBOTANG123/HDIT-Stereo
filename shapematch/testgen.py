"""Query generation: non-uniform distortion, then a random group element.

The two distortion types are not equally informative:

  vertex -- jitter each vertex.  A vertex-jittered *triangle is still a
            triangle*, hence still affine-equivalent to the regular triangle,
            so n=3 is trivially correct in affine mode.  Reported separately
            for exactly that reason.
  bow    -- subdivide each edge and displace the midpoints, giving curved
            edges.  Non-affine for every n; this is the honest test.

Both use displacements of magnitude exactly `eps * R` in a random direction so
that `eps` means the same thing on both axes of the sweep.
"""

from dataclasses import dataclass

import numpy as np

from .raster import regular_polygon, rasterize, RADIUS, CANVAS

QUERY_CANVAS = 200      # deliberately != dictionary canvas, to avoid resolution-matched artifacts
QUERY_FIT = 95.0        # max half-extent after fitting into the query canvas


def _unit(rng, k=1):
    d = rng.normal(size=(k, 2))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def bow(verts, eps, rng, R=RADIUS):
    """Subdivide each edge, displace the midpoint by eps*R.  Non-affine for every n."""
    v = np.asarray(verts, float)
    m = 0.5 * (v + np.roll(v, -1, axis=0))
    m = m + eps * R * _unit(rng, len(v))
    out = np.empty((2 * len(v), 2))
    out[0::2], out[1::2] = v, m
    return out


def vertex_jitter(verts, eps, rng, R=RADIUS):
    """Displace each vertex by eps*R in a random direction."""
    v = np.asarray(verts, float)
    return v + eps * R * _unit(rng, len(v))


DISTORTIONS = {"bow": bow, "vertex": vertex_jitter}


def random_transform(rng, group="affine", maxcond=3.0, reflect_p=0.5):
    """A random element of the generating group.

    similarity -> rotation x isotropic scale (x reflection)
    affine     -> the above composed with an anisotropy of condition number
                  drawn in [1, maxcond], at a random axis.
    """
    th = rng.uniform(0, 2 * np.pi)
    s = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
    c, sn = np.cos(th), np.sin(th)
    A = s * np.array([[c, -sn], [sn, c]])
    if group == "affine":
        k = np.sqrt(rng.uniform(1.0, maxcond))
        ph = rng.uniform(0, np.pi)
        cp, sp = np.cos(ph), np.sin(ph)
        Rp = np.array([[cp, -sp], [sp, cp]])
        A = A @ (Rp @ np.diag([k, 1 / k]) @ Rp.T)
    if rng.random() < reflect_p:
        A = A @ np.diag([1.0, -1.0])
    return A


@dataclass
class Query:
    mask: np.ndarray
    n: int
    eps: float
    kind: str
    group: str
    A: np.ndarray     # ground-truth linear part, distorted-polygon coords -> query canvas
    b: np.ndarray
    verts: np.ndarray  # final vertices in query-canvas coordinates


def make_query(n, eps, rng, group="affine", kind="bow", maxcond=3.0,
               canvas=QUERY_CANVAS, reflect_p=0.5):
    v0 = regular_polygon(n)
    if eps > 0:
        v0 = DISTORTIONS[kind](v0, eps, rng)
    elif kind == "bow":
        v0 = bow(v0, 0.0, rng)      # keep the vertex count identical at eps=0

    A = random_transform(rng, group, maxcond, reflect_p)
    c = np.array([CANVAS / 2, CANVAS / 2])
    v1 = (A @ (v0 - c).T).T + c

    m1 = v1.mean(0)
    sc = min(1.0, QUERY_FIT / np.abs(v1 - m1).max())
    off = np.array([canvas / 2, canvas / 2], float)
    v2 = (v1 - m1) * sc + off

    B = sc * A
    b = v2[0] - B @ v0[0]
    return Query(rasterize(v2, N=canvas), n, eps, kind, group, B, b, v2)
