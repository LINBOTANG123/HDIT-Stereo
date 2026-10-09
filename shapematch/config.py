"""Tunables, in one place so the sensitivity sweep can vary them."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    # canonical frame
    nc: int = 128          # canonical grid is nc x nc
    half: float = 3.5      # spanning [-half, half]^2; a whitened n-gon has circumradius ~2
    tau: float = 0.75      # signed-distance clamp, canonical units

    # polar resampling / rotation search
    ntheta: int = 256      # power of 2; rotation resolution is 2*pi/ntheta before interpolation
    nr: int = 64
    rmin: float = 0.15     # skip the singular centre ring
    weight_exp: float = 0.5  # ring weight r**weight_exp; 0.5 makes the polar sum ~ Cartesian L2

    # O(2) has two components.  Searching the reflected one lets a shape match its
    # own mirror image, which is right for an achiral dictionary and WRONG for a
    # chiral one.  True here keeps the eval reproducible; the deployment API in
    # api.py defaults it to False.  See README.
    reflection: bool = True

    # peak keeping -- load-bearing; see README, "Residual O(2) search"
    rho: float = 0.05      # keep local maxima within rho * (range) of the global max
    pmax: int = 8          # per reflection parity, so <= 16 candidates
    minsep: int = 8        # minimum angular separation between kept peaks, in theta bins

    # refinement -- off by default, it is not load-bearing
    refine: bool = False
    # "all": re-solve from every kept peak (what the eval measured, ~10x the cost).
    # "best": re-solve only from the peak that already won -- a local polish of an
    # already-chosen basin, which is what refinement actually is.
    refine_scope: str = "all"
    refine_levels: tuple = (48, 96, 192)
    refine_maxiter: int = 50

    def evolve(self, **kw):
        return replace(self, **kw)


DEFAULT = Config()
