"""Shape-recovery evaluation: apply shape-match-2d to per-object masks.

WHAT THIS MEASURES
    report_metrics.py's `cmd_shape` already asks "is the predicted object
    boundary in the right PLACE" (Chamfer/Hausdorff between pred and GT
    contours, same pixel frame, whole scene at once).

    This script asks a different, complementary question, using
    shape-match-2d (vendored in this repo as ./shapematch/):
    "even factoring out translation/rotation/scale errors, did the model
    recover the right SHAPE?" -- e.g. did a predicted square come out with
    square corners, or rounded/melted ones; did a triangle keep 3 straight
    edges. shapematch aligns the predicted mask to the GT mask over Sim(2)
    (or Aff(2)) first, *then* scores 1-IoU, so pose/scale error contributes
    nothing to the score -- only residual shape disagreement does. See
    the upstream shape-match-2d README for the method.

    It is run per OBJECT INSTANCE, not per scene, because shapematch expects
    one binary blob.

GT: the amodal mask directly, no SEG
    GT = gt_dir/amodal_mask_obj{1,2}/<stem>_mask.npy, used as-is -- the
    object's full shape, including whatever's occluded. No exact visible-
    region ground truth is used here.

    That mask is saved on the CYCLOPEAN grid, but disp_left and every
    prediction's disp.npy are on the LEFT-VIEW grid -- a different
    coordinate system, shifted per object by its own +-d0/2 in x. Using the
    raw cyclopean mask as the crop/anchor against left-view disparity was
    confirmed to corrupt extraction badly (a verified-frontal, i.e.
    provably-constant-disparity object measured std=24.5 px within its own
    footprint when anchored this way; correctly shifted, std=4.9, ordinary
    edge noise). anchor_mask_left (via to_left_view) applies that shift
    using the object's own known disparity from disp_obj{1,2} before the
    mask is used for anything -- see its docstring for the derivation.

    Pred is extracted from the model's own disparity (pred_dir/<stem>/
    <stem>_<seed>_disp.npy) via mask_from_disp: crop to a padded box around
    the (left-view-shifted) amodal mask's bounding box, local Otsu threshold
    (this pipeline only ever scores the foremost/nearer object, so "larger
    disparity = keep" is the correct, not merely default, polarity), a
    value-based floor using the object's own known true disparity (d0, from
    disp_obj{1,2}) to reject occluded-object leakage that slips past Otsu,
    constrain to a dilated version of the amodal mask so a touching/
    overlapping neighbor object doesn't get pulled into the same connected
    component, fill holes, keep the connected component overlapping the
    amodal mask most.

    Known, accepted limitation: disparity carries no information about
    occluded pixels, so a heavily-occluded object's disp-derived mask is
    necessarily just its visible sliver -- scored here against the FULL
    amodal mask anyway. That will show up as a bad score for occlusion
    severity, not (only) shape fidelity, on any object with significant
    overlap. No UDF/amodal-completion channel is used to work around this.

USAGE
    python shape_recovery_eval.py \\
        --gt-dir generated_boundary_area_textureback_128_test_left \\
        --pred-dir "lcc_dino=stereo_diffusion/results_boundary_area_textureback_128_left/stereo_diffusion_lcc_dino_dispudf_128_left_backtexture_01000000" \\
        --out-csv shape_recovery.csv

    Each --pred-dir is one row-group ("model") in the output; label defaults
    to the directory's basename (override with name=path).

TODO (not implemented, flagged rather than faked)
    - 3+-object scenes: would need per-object anchors recovered from SEG via
      connected components (obj3 anchor masks do exist on disk in some GT
      dirs, e.g. amodal_mask_obj3, but the loop below only tries obj 1/2).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

try:
    from skimage.filters import threshold_otsu
except ImportError:
    threshold_otsu = None

from shapematch import match_shape  # vendored package at ./shapematch

PAD = 14           # px padding around the anchor bbox for the local crop
MIN_FG_PX = 8       # below this, treat extraction as failed (nan row)


# --------------------------------------------------------------------------- #
#  mask extraction
# --------------------------------------------------------------------------- #

def _bbox(mask: np.ndarray, pad: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    h, w = mask.shape
    return (max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, h),
            max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, w))


def _crop(a: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return a[y0:y1, x0:x1]


def _pick_component_matching_anchor(fg: np.ndarray, anchor_crop: np.ndarray) -> np.ndarray:
    """Among connected components of `fg`, keep the one overlapping
    `anchor_crop` most -- ties a locally-segmented blob back to the specific
    object instance without assuming it's centred or largest."""
    lbl, n = ndimage.label(fg)
    if n <= 1:
        return fg
    overlaps = ndimage.sum(anchor_crop, lbl, index=range(1, n + 1))
    best = 1 + int(np.argmax(overlaps))
    return lbl == best


def mask_from_disp(disp: np.ndarray, bbox: tuple[int, int, int, int],
                    anchor_crop: np.ndarray, anchor_dilate: int = 3,
                    d0_floor: float | None = None) -> np.ndarray:
    """Disparity -> filled instance mask via local Otsu, ALWAYS taking the
    nearer/larger-disparity side -- this pipeline only ever scores the
    foremost object (never the occluded one), so "nearer = this object" is
    the correct assumption here, not a bug to guard against.

    Two failure modes, both confirmed on real data, both addressed here:

    1. Otsu + connected-components alone does not separate two objects that
       touch/overlap in the crop (two_object_11: obj1/obj2 amodal masks
       physically overlap by 802px; Otsu pulled a chunk of obj2 into obj1's
       "extracted" mask as one blob, 2573px vs a true anchor of 1538px).
       Fix: constrain the Otsu foreground to this object's own dilated
       anchor before filling/picking.

    2. A blind global Otsu split can still leak a few pixels of the occluded
       (farther, lower-disparity) object into the foreground when they sit
       within the dilated anchor and the crop-wide threshold happens to sit
       below them. Confirmed twice, and a fixed fractional margin (the first
       attempt: reject anything > 40% of this object's own d0 below d0) was
       NOT robust -- it caught two_object_0/obj2 in the backtexture dataset
       (d0=57 vs the occluder's ~11, huge gap) but missed two_object_0/obj2
       in the frontal100 dataset (d0=14.8 vs the occluder's 10.5, only a 29%
       relative gap, inside the 40% margin -- confirmed the 158 leaked
       pixels' disparity, 10.2-10.7, sat right at the occluder's own value).
       Fix: `d0_floor`, precomputed by the caller as the MIDPOINT between
       this object's own known true disparity and the occluded object's (both
       readable from disp_obj{1,2}) -- exact regardless of how close the two
       depths are, unlike a fixed percentage of either one alone.
    """
    crop = _crop(disp, bbox)
    if threshold_otsu is None or crop.max() - crop.min() < 1e-3:
        return np.zeros_like(crop, dtype=bool)
    fg = crop > threshold_otsu(crop)
    if d0_floor is not None:
        fg &= crop > d0_floor
    if anchor_dilate:
        fg &= ndimage.binary_dilation(anchor_crop, iterations=anchor_dilate)
    fg = ndimage.binary_fill_holes(fg)
    return _pick_component_matching_anchor(fg, anchor_crop)


# --------------------------------------------------------------------------- #
#  scoring
# --------------------------------------------------------------------------- #

def shape_score(gt_mask: np.ndarray, pred_mask: np.ndarray, *,
                 similarity: bool, refine: bool) -> tuple[dict, object | None]:
    """Returns (row_dict, ShapeMatch-or-None). The ShapeMatch is kept around
    (not just the scalars) so save_viz() can plot shapematch's own canonical-
    frame alignment -- the exact thing `score` was computed from -- rather
    than reconstructing it separately."""
    if int(gt_mask.sum()) < 3 or int(pred_mask.sum()) < MIN_FG_PX:
        return dict(score=float("nan"), rotation_deg=float("nan"), scale=float("nan"),
                    gt_area=int(gt_mask.sum()), pred_area=int(pred_mask.sum())), None
    out = match_shape(pred_mask, {"gt": gt_mask}, similarity=similarity,
                       refine=refine, reflection=False, with_fields=False)
    row = dict(score=out.best_score, rotation_deg=float(out.rotation_deg[0]),
               scale=float(out.scale[0]),
               gt_area=int(gt_mask.sum()), pred_area=int(pred_mask.sum()))
    return row, out


def save_viz(path: Path, gt_mask_: np.ndarray, pred_mask: np.ndarray,
             out: object | None, title: str) -> None:
    """One PNG per (stem, obj): the amodal GT mask, the disp-derived pred
    mask, their raw overlay (native crop pixel frame -- shows localization
    error), and shapematch's own canonical-frame aligned overlay (what
    `score` was actually computed from)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def overlay(a, b):
        # a=GT (blue), b=pred (orange), both (dark) -- same convention as
        # shape-match-2d/examples/quickstart.py
        ov = np.ones(a.shape + (3,))
        ov[a & ~b] = [.62, .76, .88]
        ov[b & ~a] = [.94, .62, .50]
        ov[a & b] = [.20, .26, .33]
        return ov

    panels = [(gt_mask_, "GT (amodal)"), (pred_mask, "pred (disp, extracted)"),
              (overlay(gt_mask_, pred_mask), "raw overlay")]
    if out is not None:
        panels.append((overlay(out.canonical_dictionary[0], out.best_aligned), "aligned overlay"))

    fig, ax = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.0))
    for a, (img, lab) in zip(ax, panels):
        a.imshow(img, cmap="gray_r" if img.dtype == bool else None, interpolation="nearest")
        a.set_title(lab, fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  data discovery
# --------------------------------------------------------------------------- #

def discover_stems(gt_dir: Path) -> list[str]:
    d = gt_dir / "amodal_mask_obj1"
    stems = [p.name[:-len("_mask.npy")] for p in d.glob("two_object_*_mask.npy")]
    return sorted(stems, key=lambda s: int(s.rsplit("_", 1)[1]))


def anchor_mask(gt_dir: Path, stem: str, obj: int) -> np.ndarray | None:
    p = gt_dir / f"amodal_mask_obj{obj}" / f"{stem}_mask.npy"
    if not p.exists():
        return None
    return np.load(p).astype(bool)


def select_foremost(gt_dir: Path, stem: str) -> int | None:
    """The front object: by shape_stereo_slant.py's own generation
    convention (its gen_two_objects comment literally says "obj1 = back
    object, obj2 = front object"), the front one is whichever has the
    LARGER known disparity (nearer = larger disparity) -- exact by
    construction, not something to infer.

    An earlier version of this inferred "foremost" from SEG-visible-fraction
    (>90% visible) instead. That heuristic disagreed with the true front
    object on 4/100 backtexture-fronto instances (presumably SEG-labeling
    edge cases at the boundary), each scored badly (mean 1-IoU 0.40 vs 0.02
    for the correctly-selected front objects) precisely because they weren't
    actually the front object. Comparing d0 directly removes that failure
    mode instead of just tolerating it."""
    d1 = disp_obj_value(gt_dir, stem, 1)
    d2 = disp_obj_value(gt_dir, stem, 2)
    if d1 is None or d2 is None:
        return None
    return 2 if d2 >= d1 else 1


def disp_obj_value(gt_dir: Path, stem: str, obj: int) -> float | None:
    """This object's own (constant, CYCLOPEAN-grid) disparity, read from
    disp_obj{obj}/<stem>_disp.npy (= its own amodal mask * a single scalar
    d0, per shape_stereo_slant.py's gen_two_objects). None if missing."""
    p = gt_dir / f"disp_obj{obj}" / f"{stem}_disp.npy"
    if not p.exists():
        return None
    d = _load_2d(p)
    nz = d[d > 0]
    return float(nz.mean()) if nz.size else 0.0


def disp_floor(gt_dir: Path, stem: str, obj: int, d0: float | None) -> float | None:
    """Midpoint between this object's own known true disparity (`d0`) and
    the OTHER object's, for use as mask_from_disp's d0_floor -- exact
    regardless of how close the two objects' depths are (see
    mask_from_disp's docstring for why a fixed fractional margin isn't
    robust to that). None if either object's d0 isn't available (e.g. a
    one-object scene, or a missing disp_obj file); mask_from_disp already
    treats d0_floor=None as "no floor filter"."""
    if d0 is None:
        return None
    other = 2 if obj == 1 else 1
    d0_other = disp_obj_value(gt_dir, stem, other)
    if d0_other is None:
        return None
    return (d0 + d0_other) / 2.0


def to_left_view(mask: np.ndarray, d0: float) -> np.ndarray:
    """CYCLOPEAN-grid mask -> LEFT-VIEW-grid mask via a +d0/2 horizontal
    shift. amodal_mask_obj{1,2}/SEG are saved on the cyclopean grid (source:
    the generator's own mask_obj1_cyc/mask_obj2_cyc variable names and
    SEG_DIR being written from segmap==seg_c); disp_left is a DIFFERENT
    coordinate system, shifted per-object by its own +-d0/2 in x (see
    sample_slanted_homographies: Hl = makeH(+0.5*d0)). Using the raw
    cyclopean mask to crop/anchor disp_left without this correction was
    confirmed to be the dominant cause of what looked like "severe slant
    breaking Otsu": on two_object_16 (a verified-frontal, std=0 by
    construction object), the uncorrected anchor measured disp std=24.5
    within the object's own footprint; the shift correction below brings
    that to std=4.9 (residual = ordinary rasterization/edge noise). This
    shift is EXACT when the object's slant magnitude is 0 (a pure
    translation, as verified directly against
    disparity_from_composed_homographies_on_left); for genuinely slanted
    objects it's an approximation (the true Hl differs from Hc by more than
    translation), but still a large improvement over no correction at all
    (two_object_4, genuinely slanted: std 26.2 -> 12.4)."""
    shifted = ndimage.shift(mask.astype(np.float32), shift=(0, d0 / 2.0),
                             order=1, mode="constant", cval=0.0)
    return shifted > 0.5


def anchor_mask_left(gt_dir: Path, stem: str, obj: int) -> np.ndarray | None:
    """anchor_mask(), shifted into left-view coordinates (see to_left_view).
    Falls back to the raw cyclopean mask (with a one-time warning upstream)
    if disp_obj{obj} isn't available to read d0 from."""
    m = anchor_mask(gt_dir, stem, obj)
    if m is None:
        return None
    d0 = disp_obj_value(gt_dir, stem, obj)
    if d0 is None:
        return m
    return to_left_view(m, d0)


def _load_2d(p: Path) -> np.ndarray:
    a = np.load(p)
    return a[..., 0] if a.ndim == 3 else a


def pred_disp_file(pred_dir: Path, stem: str, seed: str) -> Path | None:
    """Our diffusion model writes per-scene subdirs with a seed/aggregate
    suffix (<stem>/<stem>_median_disp.npy); deterministic baselines (e.g.
    Selective-IGEV) write one flat file per scene straight in pred_dir, no
    subdir, no seed suffix (<stem>_disp.npy) -- try both layouts."""
    candidates = [
        pred_dir / stem / f"{stem}_{seed}_disp.npy",
        pred_dir / stem / f"{stem}_disp.npy",
        pred_dir / f"{stem}_disp.npy",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _resolve_pred_base(pred_dir: Path, stems: list[str]) -> Path:
    """Some results layouts put an extra split-name level between the model
    dir and the per-scene dirs, e.g. <pred_dir>/regular/two_object_0/... (seen
    in results_boundary_area_textureback_128_left) instead of
    <pred_dir>/two_object_0/... (seen in results_ablation_regular100). Probe
    a handful of stems directly under pred_dir first; if none are there, look
    one level down and descend into it iff exactly one subdirectory contains
    any of them -- ambiguous cases (e.g. multiple splits) are surfaced, not
    silently guessed.
    """
    probe = stems[: min(20, len(stems))]
    if any((pred_dir / s).is_dir() for s in probe):
        return pred_dir
    hits: dict[Path, int] = {}
    for sub in pred_dir.iterdir():
        if not sub.is_dir():
            continue
        n = sum(1 for s in probe if (sub / s).is_dir())
        if n:
            hits[sub] = n
    if len(hits) == 1:
        (only,) = hits
        print(f"  [{pred_dir.name}] auto-descending into split subdir '{only.name}/'")
        return only
    if len(hits) > 1:
        names = ", ".join(h.name for h in hits)
        raise SystemExit(f"[{pred_dir}] ambiguous: multiple split subdirs contain scene dirs "
                          f"({names}) -- point --pred-dir at one specific split, e.g. "
                          f"{pred_dir / next(iter(hits)).name}")
    return pred_dir  # no match anywhere; fall through so the caller reports 0 scorable pairs


# --------------------------------------------------------------------------- #
#  driver
# --------------------------------------------------------------------------- #

def run_model(label: str, gt_dir: Path, pred_dir: Path, stems: list[str], *,
              seed: str, pad: int, anchor_dilate: int, similarity: bool, refine: bool,
              limit: int | None, viz_dir: Path | None, viz_limit: int | None) -> list[dict]:
    rows: list[dict] = []
    n_viz = 0
    if limit:
        stems = stems[:limit]
    pred_dir = _resolve_pred_base(pred_dir, stems)
    skipped_no_foremost = 0
    for stem in stems:
        dp = pred_disp_file(pred_dir, stem, seed)
        if dp is None:
            continue
        pred_disp = _load_2d(dp)

        # ONLY the foremost (near-fully-visible) object -- never the occluded
        # one. Disparity carries no information about occluded pixels, so
        # "extracting" the occluded object from disparity alone can only ever
        # grab the wrong (occluder's) territory or its own thin visible
        # sliver, neither of which is what a disp-derived shape-recovery
        # score should be evaluating. See select_foremost's docstring for why
        # this has to be SEG-based rather than amodal-mask overlap.
        obj = select_foremost(gt_dir, stem)
        if obj is None:
            skipped_no_foremost += 1
            continue

        # left-view-shifted, not the raw cyclopean amodal mask -- disp_left
        # (and every prediction's disp.npy) is on the left-view grid, which
        # differs from cyclopean by a per-object +-d0/2 horizontal shift.
        # See anchor_mask_left / to_left_view docstrings.
        anchor = anchor_mask_left(gt_dir, stem, obj)
        if anchor is None:
            continue
        bbox = _bbox(anchor, pad)
        if bbox is None:
            continue
        gt_amodal = _crop(anchor, bbox)
        d0 = disp_obj_value(gt_dir, stem, obj)
        floor = disp_floor(gt_dir, stem, obj, d0)

        pred_mask = mask_from_disp(pred_disp, bbox, gt_amodal, anchor_dilate=anchor_dilate,
                                    d0_floor=floor)
        r, out = shape_score(gt_amodal, pred_mask, similarity=similarity, refine=refine)
        rows.append({"model": label, "stem": stem, "obj": obj, **r})

        if viz_dir is not None and (viz_limit is None or n_viz < viz_limit):
            save_viz(viz_dir / label / f"{stem}_obj{obj}.png", gt_amodal, pred_mask, out,
                     f"{label}  {stem} obj{obj}  score={r['score']:.3f}")
            n_viz += 1
    if skipped_no_foremost:
        print(f"  [{label}] {skipped_no_foremost} scene(s) had no disp_obj{{1,2}} to compare "
              f"-- skipped (could not determine the front object)")
    return rows


def summarize(rows: list[dict]) -> None:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r["score"] == r["score"]:  # not nan
            groups[(r["model"], r["obj"])].append(r["score"])
    hdr = f"  {'model':<28}{'obj':>4}{'n':>6}{'mean 1-IoU':>12}{'median':>10}{'std':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for (model, obj), vals in sorted(groups.items()):
        a = np.array(vals)
        print(f"  {model:<28}{obj:>4}{len(a):>6}{a.mean():>12.4f}"
              f"{np.median(a):>10.4f}{a.std():>9.4f}")
    print()
    print("score = 1-IoU after optimal Sim(2) alignment (0=identical shape, 1=disjoint). "
          "GT is the full amodal mask; pred is disp-derived (necessarily visible-only). "
          "A heavily-occluded object's score here reflects occlusion severity as well as "
          "shape fidelity -- check gt_area vs pred_area, or the viz, before reading a bad "
          "score as purely a shape-recovery failure.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", required=True, type=Path)
    ap.add_argument("--pred-dir", action="append", required=True,
                     help="path, or label=path to override the model label")
    ap.add_argument("--seed", default="median")
    ap.add_argument("--pad", type=int, default=PAD)
    ap.add_argument("--anchor-dilate", type=int, default=3,
                     help="px to dilate each object's amodal anchor mask before "
                          "constraining the Otsu foreground to it -- prevents a "
                          "touching/overlapping neighbor object from being pulled "
                          "into the same connected component; 0 disables")
    ap.add_argument("--affine", action="store_true",
                     help="use Aff(2) instead of the default Sim(2) -- allows shear/"
                          "anisotropic scale to also be divided out, which can mask "
                          "genuine shape errors (e.g. ellipse-for-circle); off by default")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap stems per model, for a quick check")
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--viz-dir", type=Path, default=None,
                     help="if set, save a PNG per scored (stem, obj) under "
                          "<viz-dir>/<model>/ -- GT/pred masks, raw overlay, and "
                          "shapematch's own canonical-frame aligned overlay")
    ap.add_argument("--viz-limit", type=int, default=None,
                     help="cap PNGs saved per model (default: all scored rows)")
    args = ap.parse_args()

    stems = discover_stems(args.gt_dir)
    if not stems:
        print(f"no two_object_*_mask.npy found under {args.gt_dir}/amodal_mask_obj1")
        return
    print(f"{len(stems)} two_object test scenes found under {args.gt_dir}\n")

    all_rows: list[dict] = []
    for spec in args.pred_dir:
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            label, path = Path(spec).name, spec
        rows = run_model(label, args.gt_dir, Path(path), stems,
                          seed=args.seed, pad=args.pad, anchor_dilate=args.anchor_dilate,
                          similarity=not args.affine, refine=not args.no_refine,
                          limit=args.limit, viz_dir=args.viz_dir, viz_limit=args.viz_limit)
        if not rows:
            print(f"[skip] {label}: no scorable (stem, obj) pairs found under {path}")
            continue
        all_rows.extend(rows)

    if not all_rows:
        return
    summarize(all_rows)

    if args.out_csv:
        cols = ["model", "stem", "obj", "score", "rotation_deg", "scale", "gt_area", "pred_area"]
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k, "") for k in cols})
        print(f"per-object shape-recovery scores -> {args.out_csv}")


if __name__ == "__main__":
    main()
