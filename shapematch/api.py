"""Deployment API: one object, one call, arrays out.

    from shapematch import ShapeMatcher

    matcher = ShapeMatcher(dictionary)          # precompute the dictionary side once
    out = matcher.match(query_mask)             # then one call per query

    out.canonical_dictionary   # (K, G, G) the dictionary in the canonical frame
    out.scores                 # (K,)      1 - IoU, lower is better
    out.aligned_query          # (K, G, G) the query aligned to each item, same grid
    out.best_aligned           # (G, G)    the winner

Everything is ordered like the dictionary and never re-sorted; use `ranking` or
`best` to index into it.

Differences from the defaults used in the paper evaluation, both deliberate:

  reflection  defaults to **False** here.  Searching the reflected component of
              O(2) lets a shape match its own mirror image.  For an *achiral*
              dictionary that search is redundant -- aligning a mirrored query
              to an achiral template is the same problem as aligning the
              un-mirrored query -- and for a *chiral* dictionary it is wrong,
              because it collapses two distinct classes onto each other.
              Measured on the regular-polygon dictionary: identical accuracy,
              self-distance 6.28% vs 6.31%, and 30% faster.  Turn it on only if
              your classes really are mirror-equivalent.

  refine      defaults to **True** here, scoped to the winning correlation peak
              rather than all of them.  Refinement is a local polish of an
              already-chosen basin, so re-solving from every peak buys almost
              nothing: measured 5.69% vs 5.64% self-distance for 6.8x the cost.
              Refinement improves the *alignment*, not the ranking -- over 120
              paired queries it changed the predicted label once, and that once
              it was wrong.  Set refine=False for a ~20x speedup if you only
              need the scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

from .config import Config, DEFAULT
from .canonical import canonicalize
from .match import Dictionary, match
from .raster import signed_distance

__all__ = ["ShapeMatcher", "ShapeMatch", "match_shape"]


def as_mask(a: Any, threshold: float = 0.5, name: str = "mask") -> np.ndarray:
    """Coerce an array-like to a 2D boolean mask.

    bool stays as-is, integers are compared against zero, floats are thresholded.
    """
    arr = np.asarray(a)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    if arr.dtype == bool:
        m = arr
    elif np.issubdtype(arr.dtype, np.integer):
        m = arr != 0
    else:
        m = arr > threshold
    if m.sum() < 3:
        raise ValueError(f"{name} has {int(m.sum())} set pixels; need at least 3 "
                         "to define a centroid and second moment")
    return m


def _rotation_and_scale(A: np.ndarray) -> tuple[float, float, float]:
    """(rotation in degrees, uniform scale, condition number) of a 2x2 map."""
    U, sv, Vt = np.linalg.svd(A)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:                 # fold the flip into the reflection flag
        Q = Q @ np.diag([1.0, -1.0])
    return (float(np.degrees(np.arctan2(Q[1, 0], Q[0, 0]))),
            float(np.sqrt(abs(np.linalg.det(A)))),
            float(sv[0] / max(sv[-1], 1e-12)))


@dataclass(frozen=True)
class ShapeMatch:
    """Result of matching one query against a dictionary.

    All per-item arrays have leading dimension K and are in dictionary order.
    """

    names: tuple                      # dictionary keys, in order
    canonical_dictionary: np.ndarray  # (K, G, G) bool  -- the dictionary in canonical frame
    scores: np.ndarray                # (K,) float      -- 1 - IoU, lower is better
    aligned_query: np.ndarray         # (K, G, G) bool  -- query aligned to each item
    ranking: np.ndarray               # (K,) int        -- argsort(scores)
    best: int                         # ranking[0]

    scores_symdiff: np.ndarray        # (K,) float -- symdiff / template area, never ranked on
    canonical_query: np.ndarray       # (G, G) bool -- query in its own frame, pre-rotation

    transform_A: np.ndarray           # (K, 2, 2) dictionary-image -> query-image linear part
    transform_b: np.ndarray           # (K, 2)    the translation of the same map
    reflected: np.ndarray             # (K,) bool -- det(A) < 0

    rotation_deg: np.ndarray          # (K,) rotation of the recovered map, degrees
    scale: np.ndarray                 # (K,) sqrt|det A|
    linear_cond: np.ndarray           # (K,) cond(A); 1.0 in similarity mode
    peak_rank: np.ndarray             # (K,) 0 if the winner was the global correlation max
    n_peaks: np.ndarray               # (K,) candidates evaluated

    dictionary_sdt: np.ndarray | None  # (K, G, G) float32, clamped, canonical frame
    aligned_sdt: np.ndarray | None     # (K, G, G) float32, recomputed from aligned_query

    group: str
    grid: int
    extent: float
    reflection: bool
    refined: bool

    @property
    def best_name(self) -> Hashable:
        return self.names[self.best]

    @property
    def best_score(self) -> float:
        return float(self.scores[self.best])

    @property
    def best_aligned(self) -> np.ndarray:
        return self.aligned_query[self.best]

    @property
    def best_template(self) -> np.ndarray:
        return self.canonical_dictionary[self.best]

    @property
    def margin(self) -> float:
        """(second best - best) / best.  inf for a single-item dictionary."""
        if len(self.scores) < 2:
            return float("inf")
        s = np.sort(self.scores)
        return float("inf") if s[0] <= 0 else float((s[1] - s[0]) / s[0])

    def __repr__(self) -> str:
        return (f"ShapeMatch(best={self.best_name!r}, score={self.best_score:.4f}, "
                f"margin={self.margin:.2f}, K={len(self.names)}, grid={self.grid}, "
                f"group={self.group!r}, refined={self.refined})")


class ShapeMatcher:
    """Match binary 2D shapes modulo Aff(2) (default) or Sim(2).

    The dictionary side -- canonical frames, signed distance fields, polar
    spectra -- is computed once at construction, so matching N queries costs N
    query-side preparations rather than N * (K + 1).

    Parameters
    ----------
    dictionary : mapping {name: mask} or sequence of masks
        Binary 2D arrays, which may differ from each other and from the query in
        size.  Floats are thresholded at 0.5, integers compared against zero.
    similarity : bool, default False
        False searches Aff(2) (6 dof).  True restricts to Sim(2) (rotation,
        isotropic scale, translation) by normalizing scale only.
    reflection : bool, default False
        Whether to also search the reflected component of O(2).  See module
        docstring -- redundant for an achiral dictionary, wrong for a chiral one.
    refine : bool, default True
        Local refinement over the full group after the FFT search.  Improves the
        alignment by ~0.3 pp of 1 - IoU; costs roughly 20x.  Can be overridden
        per call.
    grid : int, default 128
        The canonical frame is (grid, grid).  Both the returned dictionary
        rasterizations and the aligned query live on it.
    extent : float, default 3.5
        The canonical frame spans [-extent, extent]^2.  After whitening the
        covariance is the identity, so a regular polygon has circumradius ~2;
        3.5 leaves margin for the distance field.
    config : Config, optional
        Full control over the peak search and refinement.  A supplied Config is
        authoritative: the defaults listed above apply only when it is omitted,
        and the keyword arguments override it only where actually passed.  So
        `ShapeMatcher(d, config=DEFAULT)` reproduces the evaluation settings
        exactly, while `ShapeMatcher(d, config=DEFAULT, refine=True)` changes
        only refinement.
    """

    def __init__(self, dictionary: Mapping[Hashable, Any] | Sequence[Any], *,
                 similarity: bool = False, reflection: bool | None = None,
                 refine: bool | None = None, grid: int | None = None,
                 extent: float | None = None, threshold: float = 0.5,
                 config: Config | None = None) -> None:
        if isinstance(dictionary, Mapping):
            items = list(dictionary.items())
        else:
            items = list(enumerate(dictionary))
        if not items:
            raise ValueError("dictionary is empty")

        masks = {k: as_mask(v, threshold, f"dictionary[{k!r}]") for k, v in items}

        self.group = "similarity" if similarity else "affine"
        self.threshold = threshold
        # A supplied Config is authoritative: the keyword arguments override it only
        # where they were actually passed.  Without this, `config=DEFAULT` would be
        # silently overwritten by the deployment defaults below, and would not
        # reproduce the evaluation.
        over = {}
        if reflection is not None:
            over["reflection"] = reflection
        if refine is not None:
            over["refine"] = refine
        if grid is not None:
            over["nc"] = grid
        if extent is not None:
            over["half"] = extent
        if config is not None:
            base = config
        else:
            # deployment defaults, applied only when no Config was supplied
            base = DEFAULT.evolve(refine_scope="best")
            over.setdefault("reflection", False)
            over.setdefault("refine", True)
            over.setdefault("nc", 128)
            over.setdefault("half", 3.5)
        self.cfg = base.evolve(**over)
        self._dict = Dictionary(masks, self.group, self.cfg)
        self.names: tuple = tuple(masks)

        self._cdict = np.stack([self._dict.items[k].cmask for k in self.names])
        self._csdt = np.stack([self._dict.items[k].sdt for k in self.names]).astype(np.float32)

    # -- dictionary side, available without a query -------------------------

    @property
    def canonical_dictionary(self) -> np.ndarray:
        """(K, G, G) bool -- the dictionary rasterized in the canonical frame."""
        return self._cdict

    @property
    def canonical_dictionary_sdt(self) -> np.ndarray:
        """(K, G, G) float32 -- clamped signed distance fields of the above."""
        return self._csdt

    @property
    def grid(self) -> int:
        return self.cfg.nc

    @property
    def extent(self) -> float:
        return self.cfg.half

    def __len__(self) -> int:
        return len(self.names)

    def __repr__(self) -> str:
        return (f"ShapeMatcher(K={len(self.names)}, group={self.group!r}, "
                f"grid={self.grid}, reflection={self.cfg.reflection}, "
                f"refine={self.cfg.refine})")

    # -- the call -----------------------------------------------------------

    def match(self, query: Any, *, refine: bool | None = None,
              with_fields: bool = True) -> ShapeMatch:
        """Match one query against the whole dictionary.

        Parameters
        ----------
        query : 2D array-like
            The binary map to match.  Need not be the same size as any
            dictionary item.
        refine : bool, optional
            Override the matcher's refinement setting for this call.
        with_fields : bool, default True
            Also return the canonical-frame signed distance fields.  Set False
            to save K * grid^2 floats per call.

        Returns
        -------
        ShapeMatch
        """
        qmask = as_mask(query, self.threshold, "query")
        cfg = self.cfg if refine is None else self.cfg.evolve(refine=refine)

        results = {r.name: r for r in match(qmask, self._dict, cfg=cfg)}
        rs = [results[k] for k in self.names]

        scores = np.array([r.dist for r in rs], float)
        aligned = np.stack([r.aligned for r in rs])
        A = np.stack([r.A_img for r in rs])
        rot_scale_cond = np.array([_rotation_and_scale(a) for a in A])

        asdt = None
        if with_fields:
            h = 2 * cfg.half / (cfg.nc - 1)
            asdt = np.stack([signed_distance(a, h, cfg.tau) for a in aligned]
                            ).astype(np.float32)

        return ShapeMatch(
            names=self.names,
            canonical_dictionary=self._cdict,
            scores=scores,
            aligned_query=aligned,
            ranking=np.argsort(scores, kind="stable"),
            best=int(np.argmin(scores)),
            scores_symdiff=np.array([r.asym for r in rs], float),
            canonical_query=canonicalize(qmask, self.group, cfg).mask,
            transform_A=A,
            transform_b=np.stack([r.b_img for r in rs]),
            reflected=np.array([np.linalg.det(a) < 0 for a in A]),
            rotation_deg=rot_scale_cond[:, 0],
            scale=rot_scale_cond[:, 1],
            linear_cond=rot_scale_cond[:, 2],
            peak_rank=np.array([r.peak_rank for r in rs], int),
            n_peaks=np.array([r.n_peaks for r in rs], int),
            dictionary_sdt=self._csdt if with_fields else None,
            aligned_sdt=asdt,
            group=self.group, grid=cfg.nc, extent=cfg.half,
            reflection=cfg.reflection, refined=bool(cfg.refine),
        )

    __call__ = match


def match_shape(query: Any, dictionary: Mapping[Hashable, Any] | Sequence[Any],
                **kwargs: Any) -> ShapeMatch:
    """One-shot convenience wrapper.

    Builds a ShapeMatcher and matches a single query.  If you have more than one
    query, build the ShapeMatcher once and reuse it -- the dictionary side is
    the expensive part.  Accepts every ShapeMatcher keyword, plus `with_fields`,
    which is forwarded to `ShapeMatcher.match`.
    """
    call = {k: kwargs.pop(k) for k in ("with_fields",) if k in kwargs}
    matcher = ShapeMatcher(dictionary, **kwargs)
    return matcher.match(query, **call)
