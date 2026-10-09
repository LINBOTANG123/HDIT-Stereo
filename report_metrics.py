#!/usr/bin/env python3
"""
Unified stereo metrics reporter.

Replaces (and merges) two older scripts:
  * stereo_diffusion/aggregate_metrics.py  — per-seed mean/median/best-K aggregation
  * baselines/summarize_metrics.py          — per-metric Mean/Std/Min/Max spread

It works on BOTH csv formats automatically:
  * our method (infer.py):  columns  stem, seed, udf_*, disp_*
  * sota baselines:         columns  stem, epe, bad2, mse   (no seed column)

Subcommands
-----------
csv        Aggregate one or more metrics.csv files (our method or baselines).
boundary   Compute a boundary-sharpness metric directly from predicted vs GT
           disparity .npy folders (no csv needed).
shape      Compute a shape/contour-fidelity metric (Chamfer + Hausdorff, in
           pixels) between the boundary contour implied by a prediction and
           the GT boundary contour -- each located independently via the
           identical fgbg rule, so neither side gets a location shortcut.
           Unlike boundary_epe, this is decoupled from disparity magnitude:
           a fixed pixel shift costs the same Chamfer distance regardless of
           how large the underlying fg/bg disparity jump is.

Examples
--------
  # aggregate / compare any mix of our + baseline csvs
  python report_metrics.py csv stereo_diffusion/results_ablation_regular100/*/ \\
                               baselines/FoundationStereo/3obj_test_results_fs \\
                               --best-k 1 3 --out-csv agg.csv

  # boundary sharpness: predicted disp folder vs GT disp folder
  python report_metrics.py boundary \\
        --pred-dir baselines/FoundationStereo/3obj_test_results_fs \\
        --gt-dir   generated_test_100/disp \\
        --out-csv  fs_boundary.csv

  # our multi-seed run, score the median map only
  python report_metrics.py boundary \\
        --pred-dir stereo_diffusion/results_3_object \\
        --gt-dir   generated_test_100/disp --use-median

  # shape/contour fidelity: predicted disp folder vs GT disp folder
  python report_metrics.py shape \\
        --pred-dir baselines/FoundationStereo/3obj_test_results_fs \\
        --gt-dir   generated_test_100/disp \\
        --out-csv  fs_shape.csv
"""
import argparse
import csv
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None

try:
    from scipy import stats as _scipy_stats
except ImportError:
    _scipy_stats = None

warnings.filterwarnings("ignore")  # silence all-NaN slice warnings


# ============================================================================ #
#  CSV AGGREGATION  (merges aggregate_metrics.py + summarize_metrics.py)
# ============================================================================ #

DEFAULT_SHOW = ["disp_epe", "disp_rmse", "disp_bad0.5", "disp_bad1",
                "disp_bad2", "disp_bad3", "disp_bad4", "udf_epe", "udf_bad2",
                "boundary_epe",
                # baseline-style unprefixed columns
                "epe", "rmse", "bad2", "mse"]


def resolve(path):
    """Return (csv_path, model_label). A dir resolves to its metrics.csv."""
    if os.path.isdir(path):
        return os.path.join(path, "metrics.csv"), os.path.basename(path.rstrip("/"))
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return path, parent or os.path.splitext(os.path.basename(path))[0]


def resolve_ensemble(path):
    """Path of the metrics_ensemble.csv (map-median scores) paired with `path`."""
    if os.path.isdir(path):
        return os.path.join(path, "metrics_ensemble.csv")
    d = os.path.dirname(os.path.abspath(path))
    if os.path.basename(path) == "metrics_ensemble.csv":
        return path
    return os.path.join(d, "metrics_ensemble.csv")


def resolve_boundary(path):
    """Path of the boundary_metrics.csv (per-seed boundary EPE etc.) paired with `path`."""
    if os.path.isdir(path):
        return os.path.join(path, "boundary_metrics.csv")
    d = os.path.dirname(os.path.abspath(path))
    if os.path.basename(path) == "boundary_metrics.csv":
        return path
    return os.path.join(d, "boundary_metrics.csv")


def load_csv_samples(path):
    """Group rows by stem. Works whether or not a 'seed' column exists.

    Also returns a stem -> [seed, ...] map (aligned with the per-stem sample
    list) so callers can join another per-seed CSV (e.g. boundary metrics) by
    (stem, seed) instead of assuming row order matches.
    """
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}, [], {}
    cols = [c for c in rows[0] if c not in ("stem", "seed", "model")]
    samples = defaultdict(list)
    seeds = defaultdict(list)
    for r in rows:
        d = {}
        for c in cols:
            try:
                d[c] = float(r.get(c))
            except (TypeError, ValueError):
                d[c] = float("nan")
        samples[r["stem"]].append(d)
        seeds[r["stem"]].append(r.get("seed"))
    return samples, cols, seeds


def _nanmean(vals):
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def _nanmedian(vals):
    vals = [v for v in vals if v == v]
    return float(np.median(vals)) if vals else float("nan")


def aggregate(samples, cols, mode, k, rank):
    """Collapse each sample over its seeds, then average over samples."""
    per_sample, n_seeds = [], []
    for seeds in samples.values():
        n_seeds.append(len(seeds))
        if mode == "median":
            m = {c: _nanmedian([s[c] for s in seeds]) for c in cols}
        elif mode == "bestk":
            valid = [s for s in seeds if s.get(rank, float("nan")) == s.get(rank, float("nan"))]
            ordered = sorted(valid, key=lambda s: s[rank]) if valid else list(seeds)
            chosen = ordered[:max(1, min(k, len(ordered)))]
            m = {c: _nanmean([s[c] for s in chosen]) for c in cols}
        else:  # mean
            m = {c: _nanmean([s[c] for s in seeds]) for c in cols}
        per_sample.append(m)
    agg = {c: _nanmean([s[c] for s in per_sample]) for c in cols}
    return agg, per_sample, len(per_sample), int(round(np.mean(n_seeds))) if n_seeds else 0


def fmt(col, v):
    if v != v:
        return f"{'-':>8}"
    return f"{v:8.2f}" if "bad" in col else f"{v:8.4f}"


def cmd_csv(args):
    out_rows = []
    print()
    for path in args.paths:
        csv_path, label = resolve(path)
        if not os.path.exists(csv_path):
            print(f"[skip] {label}: no CSV at {csv_path}")
            continue
        samples, cols, seeds_map = load_csv_samples(csv_path)
        if not samples:
            print(f"[skip] {label}: empty CSV")
            continue

        if args.boundary:
            bpath = resolve_boundary(path)
            if os.path.exists(bpath):
                bsamples, _, bseeds = load_csv_samples(bpath)
                blookup = {}
                for stem, dicts in bsamples.items():
                    for d, sd in zip(dicts, bseeds.get(stem, [])):
                        blookup[(stem, sd)] = d.get("boundary_epe", float("nan"))
                for stem, seed_list in seeds_map.items():
                    for i, sd in enumerate(seed_list):
                        samples[stem][i]["boundary_epe"] = blookup.get((stem, sd), float("nan"))
                if "boundary_epe" not in cols:
                    cols.append("boundary_epe")
            else:
                print(f"  [boundary] no boundary_metrics.csv beside {label}")

        rank = args.rank_metric if args.rank_metric in cols else \
            next((c for c in cols if "epe" in c), cols[0])
        show = args.show or [c for c in DEFAULT_SHOW if c in cols] or cols

        computed, mean_per_sample, nsamp, nseed = [], None, 0, 0
        # per-seed aggregations (mean = report accuracy, median = robustness stat)
        for mode in ("mean", "median"):
            agg, per_sample, nsamp, nseed = aggregate(samples, cols, mode, 1, rank)
            if mode == "mean":
                mean_per_sample = per_sample
            computed.append((mode, agg))

        # map-median ensemble: the per-pixel median of the seed prediction maps,
        # scored in metrics_ensemble.csv (disp_epe_med). NOT the same as the
        # metric-space 'median' row above — this is the headline single estimate.
        if args.ensemble:
            ens_path = resolve_ensemble(path)
            ens_samples, ens_cols, _ = (load_csv_samples(ens_path)
                                        if os.path.exists(ens_path) else ({}, [], {}))
            if ens_samples:
                agg, _, _, _ = aggregate(ens_samples, ens_cols, "mean", 1, rank)
                computed.append(("map-median (ens)", agg))
            else:
                print(f"  [ensemble] no metrics_ensemble.csv beside {label}")

        # oracle best-K-of-N (uses GT to pick seeds → upper bound only)
        for k in sorted(set(args.best_k)):
            agg, _, nsamp, nseed = aggregate(samples, cols, "bestk", k, rank)
            computed.append((f"best{min(k, nseed)}of{nseed} (oracle)", agg))

        hdr = f"  {'aggregation':>18}  " + " ".join(f"{c.replace('disp_',''):>8}" for c in show)
        print("=" * len(hdr))
        print(f" {label}   ({nsamp} samples x {nseed} seed(s))")
        print("=" * len(hdr))
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for disp, row in computed:
            print(f"  {disp:>18}  " + " ".join(fmt(c, row.get(c, float('nan'))) for c in show))
            out_rows.append({"model": label, "aggregation": disp,
                             "n_samples": nsamp, "n_seeds": nseed,
                             **{c: row.get(c, float('nan')) for c in cols}})

        # spread table (from summarize_metrics.py): Mean/Std/Min/Max over samples,
        # computed on the per-sample seed-mean values.
        print(f"\n  spread over {nsamp} samples (seed-mean per sample):")
        print(f"    {'metric':<16}{'Mean':>11}{'Std':>11}{'Min':>11}{'Max':>11}")
        print(f"    {'-'*60}")
        for c in show:
            vals = [s[c] for s in mean_per_sample if s[c] == s[c]]
            if not vals:
                continue
            a = np.array(vals)
            print(f"    {c:<16}{a.mean():>11.4f}{a.std():>11.4f}{a.min():>11.4f}{a.max():>11.4f}")
        print()

    print("NOTE: best-K-of-N is an ORACLE upper bound (seeds picked using GT). "
          "Report 'mean' as accuracy; show best-K only as a labeled upper bound.\n")

    if args.out_csv and out_rows:
        metric_cols = []
        for r in out_rows:
            for c in r:
                if c not in ("model", "aggregation", "n_samples", "n_seeds") and c not in metric_cols:
                    metric_cols.append(c)
        fields = ["model", "aggregation", "n_samples", "n_seeds"] + metric_cols
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in out_rows:
                w.writerow({k: r.get(k, "") for k in fields})
        print(f"Long-form results saved → {args.out_csv}")


# ============================================================================ #
#  BOUNDARY SHARPNESS  (pred disp folder vs GT disp folder)
# ============================================================================ #
#
# Idea: object boundaries are depth discontinuities.  An oversmoothed prediction
# spreads that jump over several pixels, lowering the *peak* gradient at the edge;
# a sharp prediction keeps it concentrated.  So we
#   1. locate GT boundary pixels via Depth Pro's fgbg_depth contours
#      (neighbour disparity ratio > fgbg_t), collapsed to a per-pixel mask,
#   2. measure the local-max gradient of pred / GT in a small window around each
#      edge pixel (window = robust to a 1-2 px edge misalignment), and
#   3. report sharp_ratio = mean(pred local-max) / mean(GT local-max) over edges.
# sharp_ratio ≈ 1 → matches GT sharpness; <1 → oversmoothed; >1 → ringing/noise.
# We also report EPE split into the boundary band vs the interior.

def _grey_dilate(a, radius):
    """Max filter over a (2r+1)x(2r+1) square. Pure-numpy (no scipy)."""
    if radius <= 0:
        return a
    H, W = a.shape
    pad = np.pad(a, radius, mode="edge")
    out = np.full_like(a, -np.inf)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out = np.maximum(out, pad[radius + dy:radius + dy + H,
                                       radius + dx:radius + dx + W])
    return out


def _binary_dilate(mask, radius):
    if radius <= 0:
        return mask
    return _grey_dilate(mask.astype(np.float32), radius) > 0.5


def _gradmag(d):
    gy, gx = np.gradient(d.astype(np.float64))
    return np.hypot(gx, gy)


def _safe_fill(d):
    """Replace non-finite with the finite median (neutral for gradients)."""
    m = np.isfinite(d)
    if m.all():
        return d.astype(np.float64), m
    out = d.astype(np.float64).copy()
    out[~m] = float(np.median(d[m])) if m.any() else 0.0
    return out, m


# ---------------------------------------------------------------------------- #
#  Boundary contour extraction — Depth Pro's fgbg_depth (verbatim).
#  apple/ml-depth-pro: src/depth_pro/eval/boundary_metrics.py (Bochkovskii 2024).
#  An occluding contour exists between neighbours when the inverse-depth ratio
#  exceeds t. Disparity *is* inverse depth (scale cancels in the ratio), so we
#  apply it to disparity directly. We REUSE this contour set to locate the GT
#  boundary for our sharpness metric. (The Depth Pro F1/P/R metric itself is
#  commented out at the bottom of this block.)
# ---------------------------------------------------------------------------- #

def fgbg_depth(d, t, min_abs_diff=0.5):
    """Left/top/right/bottom fg-bg relations (Depth Pro's ratio test, plus an
    absolute-difference floor -- see _fgbg_edge_mask docstring for why the
    floor is necessary here).
    """
    right_is_big_enough = (d[..., :, 1:] / d[..., :, :-1]) > t
    left_is_big_enough = (d[..., :, :-1] / d[..., :, 1:]) > t
    bottom_is_big_enough = (d[..., 1:, :] / d[..., :-1, :]) > t
    top_is_big_enough = (d[..., :-1, :] / d[..., 1:, :]) > t
    if min_abs_diff > 0:
        dx_big = np.abs(d[..., :, 1:] - d[..., :, :-1]) > min_abs_diff
        dy_big = np.abs(d[..., 1:, :] - d[..., :-1, :]) > min_abs_diff
        right_is_big_enough &= dx_big
        left_is_big_enough &= dx_big
        bottom_is_big_enough &= dy_big
        top_is_big_enough &= dy_big
    return (left_is_big_enough, top_is_big_enough,
            right_is_big_enough, bottom_is_big_enough)


def _fgbg_edge_mask(d, t, eps=1e-6, min_abs_diff=0.5):
    """Collapse Depth Pro's 4 directional fgbg contours into one per-pixel
    boundary mask: a pixel is an edge if it borders an occluding step
    (neighbour ratio > t) in any direction. Both pixels of each contributing
    pair are marked, so the discontinuity is flagged on both sides.

    min_abs_diff guards against a real numerical failure mode: this dataset's
    background disparity is exactly 0 in GT, but predictions carry tiny
    floating-point noise there (e.g. +/-0.0005). A RATIO between two such
    near-zero, differently-signed values can trivially exceed t even though
    the true difference is sub-pixel and meaningless -- verified directly: an
    unguarded ratio test flags almost the entire background of a textured
    scene as "boundary". Requiring the absolute difference to also clear
    min_abs_diff (default 0.5px) suppresses this while a genuine multi-pixel
    object-boundary jump clears it trivially. Set min_abs_diff=0 to recover
    the original ratio-only rule.
    """
    d = np.clip(d.astype(np.float64), eps, None)
    left, top, right, bottom = fgbg_depth(d, t, min_abs_diff)
    H, W = d.shape
    edge = np.zeros((H, W), dtype=bool)
    h = left | right          # H×(W-1): horizontal step between col j and j+1
    edge[:, :-1] |= h
    edge[:, 1:] |= h
    v = top | bottom          # (H-1)×W: vertical step between row i and i+1
    edge[:-1, :] |= v
    edge[1:, :] |= v
    return edge


# ---- Depth Pro F1/P/R metric — COMMENTED OUT (kept for reference) ----------- #
# def boundary_f1(pr, gt, t, return_p=False, return_r=False):
#     """Boundary F1 / precision / recall at one threshold (verbatim Depth Pro)."""
#     ap, bp, cp, dp = fgbg_depth(pr, t)
#     ag, bg, cg, dg = fgbg_depth(gt, t)
#     r = 0.25 * (
#         np.count_nonzero(ap & ag) / max(np.count_nonzero(ag), 1)
#         + np.count_nonzero(bp & bg) / max(np.count_nonzero(bg), 1)
#         + np.count_nonzero(cp & cg) / max(np.count_nonzero(cg), 1)
#         + np.count_nonzero(dp & dg) / max(np.count_nonzero(dg), 1)
#     )
#     p = 0.25 * (
#         np.count_nonzero(ap & ag) / max(np.count_nonzero(ap), 1)
#         + np.count_nonzero(bp & bg) / max(np.count_nonzero(bp), 1)
#         + np.count_nonzero(cp & cg) / max(np.count_nonzero(cp), 1)
#         + np.count_nonzero(dp & dg) / max(np.count_nonzero(dp), 1)
#     )
#     if r + p == 0:
#         return 0.0
#     if return_p:
#         return p
#     if return_r:
#         return r
#     return 2 * (r * p) / (r + p)
#
#
# def get_thresholds_and_weights(t_min, t_max, N):
#     """Linspace thresholds, weighted toward stricter t (verbatim Depth Pro)."""
#     thresholds = np.linspace(t_min, t_max, N)
#     weights = thresholds / thresholds.sum()
#     return thresholds, weights
#
#
# def si_boundary_prf(pred_disp, gt_disp, t_min=1.05, t_max=1.25, N=10, eps=1e-6):
#     """Depth Pro SI boundary (F1, precision, recall) on disparity maps."""
#     pr = np.clip(pred_disp.astype(np.float64), eps, None)
#     gt = np.clip(gt_disp.astype(np.float64), eps, None)
#     thresholds, weights = get_thresholds_and_weights(t_min, t_max, N)
#     f1 = np.array([boundary_f1(pr, gt, t) for t in thresholds])
#     p = np.array([boundary_f1(pr, gt, t, return_p=True) for t in thresholds])
#     r = np.array([boundary_f1(pr, gt, t, return_r=True) for t in thresholds])
#     return (float(np.sum(f1 * weights)),
#             float(np.sum(p * weights)),
#             float(np.sum(r * weights)))


def boundary_metrics(pred, gt, fgbg_t, win, min_abs_diff=0.5):
    """Metrics for one pred/GT disparity pair. None only if nothing is finite.

    Boundary location = Depth Pro fgbg_depth contours on the GT (neighbour
    disparity ratio > fgbg_t, AND absolute difference > min_abs_diff -- see
    _fgbg_edge_mask), collapsed to a per-pixel edge mask. Sharpness = our
    gradient-steepness ratio measured at those edge pixels.
    """
    gt_f, gt_m = _safe_fill(gt)
    pred_f, pred_m = _safe_fill(pred)
    finite = gt_m & pred_m
    if not finite.any():
        return None

    aerr = np.abs(pred_f - gt_f)

    # ---- boundary from Depth Pro's fgbg contour set (GT side) ----
    edge = _fgbg_edge_mask(gt_f, fgbg_t, min_abs_diff=min_abs_diff) & finite

    # ---- our gradient-steepness sharpness, measured at those edges ----
    gt_grad = _gradmag(gt_f)
    pred_grad = _gradmag(pred_f)
    if edge.any():
        band = _binary_dilate(edge, win) & finite     # transition zone
        pred_lmax = _grey_dilate(pred_grad, win)      # local-max gradient
        gt_lmax = _grey_dilate(gt_grad, win)          # (robust to misalignment)
        sharp_pred = float(np.mean(pred_lmax[edge]))
        sharp_gt = float(np.mean(gt_lmax[edge]))
        sharp_ratio = sharp_pred / sharp_gt if sharp_gt > 0 else float("nan")
        boundary_epe = float(np.mean(aerr[band])) if band.any() else float("nan")
    else:
        band = np.zeros_like(finite)
        sharp_pred = sharp_gt = sharp_ratio = boundary_epe = float("nan")
    interior = finite & ~band
    interior_epe = float(np.mean(aerr[interior])) if interior.any() else float("nan")

    return {
        "sharp_ratio":  sharp_ratio,
        "sharp_pred":   sharp_pred,
        "sharp_gt":     sharp_gt,
        "boundary_epe": boundary_epe,
        "interior_epe": interior_epe,
        "edge_frac":    float(edge.mean() * 100.0),
    }


_VARIANT_RE = re.compile(r"_(seed\d+)$")


def _base_and_variant(fname):
    """('multi_3obj_0_seed1_disp.npy') -> ('multi_3obj_0', 'seed1')."""
    s = fname[:-4] if fname.endswith(".npy") else fname
    if s.endswith("_disp_pred"):    # baseline convention (e.g. Selective-IGEV)
        s = s[:-len("_disp_pred")]
    elif s.endswith("_disp"):
        s = s[:-5]
    m = _VARIANT_RE.search(s)
    if m:
        return s[:m.start()], m.group(1)
    if s.endswith("_median"):
        return s[:-7], "median"
    return s, "plain"


def _find_gt(gt_dir, base):
    for cand in (f"{base}_disp.npy", f"{base}.npy"):
        p = gt_dir / cand
        if p.exists():
            return p
    return None


def _load_disp(path):
    a = np.load(path).astype(np.float32)
    while a.ndim > 2:
        a = a[..., 0] if a.shape[-1] == 1 else a[0]
    return a


# ============================================================================ #
#  RECOMPUTE METRICS FROM .NPY FILES
# ============================================================================ #

RECOMPUTE_SHOW = ["disp_epe", "disp_rmse", "disp_bad0.5", "disp_bad1",
                  "disp_bad2", "disp_bad3", "disp_bad4"]


def _compute_disp_metrics(pred, gt):
    """Compute all disparity metrics for one pred/GT array pair."""
    gt_f = gt.astype(np.float64)
    pred_f = pred.astype(np.float64)
    valid = np.isfinite(gt_f) & np.isfinite(pred_f)
    if not valid.any():
        return None
    g, p = gt_f[valid], pred_f[valid]
    ae = np.abs(p - g)
    n = len(ae)
    return {
        "disp_epe":    float(ae.mean()),
        "disp_rmse":   float(np.sqrt((ae ** 2).mean())),
        "disp_bad0.5": float((ae > 0.5).sum() * 100.0 / n),
        "disp_bad1":   float((ae > 1.0).sum() * 100.0 / n),
        "disp_bad2":   float((ae > 2.0).sum() * 100.0 / n),
        "disp_bad3":   float((ae > 3.0).sum() * 100.0 / n),
        "disp_bad4":   float((ae > 4.0).sum() * 100.0 / n),
    }


def cmd_recompute(args):
    """Compute all disparity metrics directly from .npy prediction files."""
    gt_dir = Path(args.gt_dir)
    print()
    for path in args.paths:
        pred_dir = Path(path)
        label = pred_dir.name

        files = sorted(pred_dir.rglob("*_disp.npy"))
        parsed = [(f, *_base_and_variant(f.name)) for f in files]

        if args.use_median:
            kept = [p for p in parsed if p[2] == "median"]
            if not kept:
                print(f"[skip] {label}: --use-median set but no *_median_disp.npy found")
                continue
        else:
            kept = [p for p in parsed if p[2] != "median"]

        groups = defaultdict(list)   # base stem -> list of pred files (per seed)
        for f, base, _ in kept:
            groups[base].append(f)

        samples = defaultdict(list)  # base stem -> list of metric dicts (per seed)
        missing_gt = shape_skip = 0
        for base in sorted(groups):
            gtp = _find_gt(gt_dir, base)
            if gtp is None:
                missing_gt += 1
                continue
            gt = _load_disp(gtp)
            for f in groups[base]:
                pred = _load_disp(f)
                if pred.shape != gt.shape:
                    shape_skip += 1
                    continue
                m = _compute_disp_metrics(pred, gt)
                if m is not None:
                    samples[base].append(m)

        if missing_gt:
            print(f"  [{label}] no GT match for {missing_gt} stem(s)")
        if shape_skip:
            print(f"  [{label}] shape mismatch on {shape_skip} file(s) — skipped")
        if not samples:
            print(f"[skip] {label}: no scorable pairs\n")
            continue

        cols = RECOMPUTE_SHOW
        rank = "disp_epe"
        nsamp = len(samples)
        nseed = int(round(np.mean([len(v) for v in samples.values()])))

        computed, mean_per_sample = [], None
        for mode in ("mean", "median"):
            agg, per_sample, _, _ = aggregate(samples, cols, mode, 1, rank)
            if mode == "mean":
                mean_per_sample = per_sample
            computed.append((mode, agg))
        for k in sorted(set(args.best_k)):
            agg, _, _, _ = aggregate(samples, cols, "bestk", k, rank)
            computed.append((f"best{min(k, nseed)}of{nseed} (oracle)", agg))

        hdr = (f"  {'aggregation':>18}  "
               + " ".join(f"{c.replace('disp_', ''):>8}" for c in cols))
        print("=" * len(hdr))
        print(f" {label}   ({nsamp} samples x {nseed} seed(s))")
        print("=" * len(hdr))
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for disp, row in computed:
            print(f"  {disp:>18}  "
                  + " ".join(fmt(c, row.get(c, float("nan"))) for c in cols))

        print(f"\n  spread over {nsamp} samples (seed-mean per sample):")
        print(f"    {'metric':<16}{'Mean':>11}{'Std':>11}{'Min':>11}{'Max':>11}")
        print(f"    {'-' * 60}")
        for c in cols:
            vals = [s[c] for s in mean_per_sample if s[c] == s[c]]
            if not vals:
                continue
            a = np.array(vals)
            print(f"    {c:<16}{a.mean():>11.4f}{a.std():>11.4f}"
                  f"{a.min():>11.4f}{a.max():>11.4f}")
        print()

    print("NOTE: best-K-of-N is an ORACLE upper bound (seeds picked using GT). "
          "Report 'mean' as accuracy; show best-K only as a labeled upper bound.\n")


def _boundary_one_dir(pred_dir, gt_dir, args):
    """Run the boundary metric over one pred dir; return (label, per_seed_rows).

    One row per (stem, seed) pred file — NOT averaged across seeds — so the
    result can be joined against metrics.csv by (stem, seed) and fed through
    the same mean/median/best-K aggregation as the disparity metrics.
    """
    pred_dir = Path(pred_dir)
    label = pred_dir.name
    files = sorted(pred_dir.rglob(args.pred_glob))
    parsed = [(f, *_base_and_variant(f.name)) for f in files]

    if args.use_median:
        kept = [p for p in parsed if p[2] == "median"]
        if not kept:
            print(f"[skip] {label}: --use-median set but no *_median_disp.npy found")
            return label, []
    else:
        # drop median maps (derived from seeds) to avoid double-counting
        kept = [p for p in parsed if p[2] != "median"]

    groups = defaultdict(list)            # base stem -> [(pred file, variant), ...]
    for f, base, variant in kept:
        groups[base].append((f, variant))

    rows = []
    missing_gt, shape_skip = 0, 0
    for base in sorted(groups):
        gtp = _find_gt(gt_dir, base)
        if gtp is None:
            missing_gt += 1
            continue
        gt = _load_disp(gtp)
        for f, variant in groups[base]:
            pred = _load_disp(f)
            if pred.shape != gt.shape:
                shape_skip += 1
                continue
            m = boundary_metrics(pred, gt, args.fgbg_t, args.win, args.min_abs_diff)
            if m is None:
                continue
            seed = variant[len("seed"):] if variant.startswith("seed") else variant
            rows.append({"stem": base, "seed": seed, **m})

    if missing_gt:
        print(f"  [{label}] no GT match for {missing_gt} stem(s)")
    if shape_skip:
        print(f"  [{label}] shape mismatch on {shape_skip} pred file(s) — skipped")
    return label, rows


BOUNDARY_SHOW = ["sharp_ratio", "boundary_epe", "interior_epe",
                 "sharp_pred", "sharp_gt", "edge_frac"]


def cmd_boundary(args):
    gt_dir = Path(args.gt_dir)
    all_out = []
    for pred_dir in args.pred_dir:
        label, rows = _boundary_one_dir(pred_dir, gt_dir, args)
        if not rows:
            print(f"[skip] {label}: no scorable pairs\n")
            continue

        n_stems = len({r["stem"] for r in rows})
        hdr = f"  {'metric':<16}{'Mean':>11}{'Std':>11}{'Min':>11}{'Max':>11}"
        print("=" * len(hdr))
        print(f" {label}   ({n_stems} samples x {len(rows) // max(n_stems, 1)} seed(s), "
              f"fgbg_t={args.fgbg_t} win={args.win}"
              f"{'  [median]' if args.use_median else ''})")
        print("=" * len(hdr))
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for c in BOUNDARY_SHOW:
            vals = [r[c] for r in rows if r.get(c) == r.get(c)]
            if not vals:
                continue
            a = np.array(vals)
            print(f"  {c:<16}{a.mean():>11.4f}{a.std():>11.4f}{a.min():>11.4f}{a.max():>11.4f}")
        print()
        for r in rows:
            all_out.append({"model": label, **r})

    print("Boundary located via Depth Pro fgbg_depth contours (neighbour ratio > "
          "fgbg_t) on GT.\nsharp_ratio ≈ 1 matches GT edge sharpness; <1 = "
          "oversmoothed (blurry boundary); >1 = overshoot/ringing.  Lower "
          "boundary_epe at sharp_ratio≈1 is the win.\n")

    if args.out_csv and all_out:
        cols = ["model", "stem", "seed"] + BOUNDARY_SHOW
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in all_out:
                w.writerow({k: r.get(k, "") for k in cols})
        print(f"Per-sample boundary metrics saved → {args.out_csv}")


# ============================================================================ #
#  SHAPE / CONTOUR FIDELITY  (Chamfer + Hausdorff between pred/GT contours)
# ============================================================================ #
#
# boundary_metrics() above answers "at the GT's boundary location, how wrong
# is the predicted disparity VALUE" -- it never asks where the prediction
# itself thinks the boundary is. This section answers that instead: locate a
# boundary contour independently in GT and in the prediction (same fgbg rule,
# applied blindly to whichever map it's given -- SOTA and our own UDF-headed
# model are treated identically, extracted from disparity only, so our model
# gets no shortcut from its extra UDF channel here), then measure the
# geometric (pixel-position) distance between the two contours.
#
# This is deliberately decoupled from disparity magnitude: a 3px localization
# error costs the same Chamfer distance whether the underlying fg/bg jump is
# 5px or 50px, unlike boundary_epe, whose magnitude is confounded by that
# jump size (this is why a handful of large-disparity-gap scenes can dominate
# boundary_epe's mean -- see the disparity-gap diagnosis). Chamfer distance is
# the "typical" shape deviation; Hausdorff is the worst-case local deviation
# (e.g. one badly-placed vertex even if the rest of the edge is fine).
#
# Chamfer/Hausdorff alone only test POSITION: a predicted contour that stays
# close to the true one scores well even if it quietly rounds off a sharp
# corner (a tight-radius arc hugging the true vertex is "close" in position
# but not the same LOCAL SHAPE). curvature_error below tests that directly:
# fit a local circle through each point's nearest same-contour neighbours
# (curvature = 1/radius -- tight bend = high curvature, straight/gentle arc =
# low), then compare curvature at each Chamfer-matched pair of points. This
# isolates local geometric character (sharp vs. rounded, straight vs. curved)
# independent of how well-positioned the contour is.

def _nearest_neighbor_indices(src, dst, chunk=2000):
    """For each point in src, (min distance, index into dst) of its nearest
    neighbour. Chunked brute-force; fine at these images' edge-pixel counts."""
    n = len(src)
    dmin = np.empty(n, dtype=np.float64)
    imin = np.empty(n, dtype=np.int64)
    for i in range(0, n, chunk):
        d = np.sqrt(((src[i:i + chunk, None, :] - dst[None, :, :]) ** 2).sum(-1))
        j = d.argmin(axis=1)
        imin[i:i + chunk] = j
        dmin[i:i + chunk] = d[np.arange(len(j)), j]
    return dmin, imin


def _local_curvature(points, k=8):
    """Local curvature (1/radius, px^-1) at every point in `points` (Nx2
    array of [y,x] pixel coords), via a Kasa least-squares circle fit through
    each point's k nearest neighbours *within the same point set* (i.e. along
    the same contour). Near-straight/degenerate neighbourhoods -> ~0.
    """
    n = len(points)
    if n < 4:
        return np.zeros(n)
    k = min(k, n - 1)
    d2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)
    nbr_idx = np.argpartition(d2, k, axis=1)[:, :k + 1]  # includes self

    curv = np.zeros(n)
    for i in range(n):
        pts = points[nbr_idx[i]]
        y, x = pts[:, 0], pts[:, 1]
        if len(np.unique(x)) < 2 and len(np.unique(y)) < 2:
            continue
        A = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)
        b = x ** 2 + y ** 2
        try:
            (a_c, b_c, c_c), *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        r2 = a_c ** 2 + b_c ** 2 + c_c
        if r2 <= 1e-6:
            continue
        r = np.sqrt(r2)
        curv[i] = 1.0 / r if r > 1e-3 else 0.0
    return curv


def _chamfer_hausdorff_curvature(mask_a, mask_b, curvature_k=8):
    """Symmetric Chamfer + Hausdorff distance (px), plus curvature_error
    (px^-1) between two boundary pixel masks."""
    ys_a, xs_a = np.nonzero(mask_a)
    ys_b, xs_b = np.nonzero(mask_b)
    if len(ys_a) == 0 or len(ys_b) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    pa = np.stack([ys_a, xs_a], axis=1).astype(np.float64)
    pb = np.stack([ys_b, xs_b], axis=1).astype(np.float64)

    d_a2b, idx_a2b = _nearest_neighbor_indices(pa, pb)   # GT -> nearest pred
    d_b2a, idx_b2a = _nearest_neighbor_indices(pb, pa)   # pred -> nearest GT

    chamfer = float(0.5 * (d_a2b.mean() + d_b2a.mean()))
    hausdorff = float(max(d_a2b.max(), d_b2a.max()))
    # HD95: 95th percentile of the pooled directional distances rather than
    # the true max -- the standard robustified Hausdorff variant (ubiquitous
    # in medical image segmentation for this exact reason) that discards the
    # single worst ~5% of points before taking the extremum, so one outlier
    # pixel on an otherwise well-matched contour doesn't dominate the metric.
    pooled = np.concatenate([d_a2b, d_b2a])
    hausdorff99 = float(np.percentile(pooled, 99))
    hausdorff95 = float(np.percentile(pooled, 95))
    hausdorff90 = float(np.percentile(pooled, 90))

    curv_a = _local_curvature(pa, k=curvature_k)
    curv_b = _local_curvature(pb, k=curvature_k)
    ce_a2b = np.abs(curv_a - curv_b[idx_a2b])
    ce_b2a = np.abs(curv_b - curv_a[idx_b2a])
    curvature_error = float(0.5 * (ce_a2b.mean() + ce_b2a.mean()))

    return chamfer, hausdorff, hausdorff99, hausdorff95, hausdorff90, curvature_error


def shape_metrics(pred, gt, fgbg_t, min_abs_diff=0.5, curvature_k=8):
    """Chamfer + Hausdorff + curvature_error between the prediction's own
    implied boundary contour and GT's, both located via the identical fgbg
    rule used by boundary_metrics() -- applied independently to each map.

    The raw fgbg rule flags every adjacent pixel-pair whose disparity ratio
    exceeds fgbg_t. On a sharp GT edge that's a single pixel-wide contour; on
    an oversmoothed prediction, a ramped transition can have MANY consecutive
    pixel-pairs each individually exceed the threshold, ballooning the "edge"
    into a wide band whose area reflects ramp width, not contour position --
    conflating exactly the smoothness signal boundary_epe/sharp_ratio already
    capture elsewhere. We skeletonize both masks to a single-pixel-wide
    contour before measuring Chamfer/Hausdorff, so the metric isolates
    position/shape and a wide ramp collapses to its centerline rather than
    inflating the distance.

    min_abs_diff (see _fgbg_edge_mask) is essential here, not optional: near
    the background's true disparity of 0, tiny prediction noise produces a
    ratio-only mask that floods the ENTIRE background with false edges
    (verified directly -- unguarded, this turns a ~400px true contour into a
    ~2000px scattered mess), which would otherwise dominate both Chamfer and
    Hausdorff with meaningless distances.
    """
    gt_f, gt_m = _safe_fill(gt)
    pred_f, pred_m = _safe_fill(pred)
    finite = gt_m & pred_m
    if not finite.any():
        return None

    gt_edge = _fgbg_edge_mask(gt_f, fgbg_t, min_abs_diff=min_abs_diff) & finite
    pred_edge = _fgbg_edge_mask(pred_f, fgbg_t, min_abs_diff=min_abs_diff) & finite

    if skeletonize is not None:
        gt_edge = skeletonize(gt_edge)
        pred_edge = skeletonize(pred_edge)

    n_gt, n_pred = int(gt_edge.sum()), int(pred_edge.sum())
    if n_gt == 0 or n_pred == 0:
        return {"chamfer": float("nan"), "hausdorff": float("nan"),
                "hausdorff99": float("nan"), "hausdorff95": float("nan"),
                "hausdorff90": float("nan"), "curvature_error": float("nan"),
                "n_gt_edge": n_gt, "n_pred_edge": n_pred}

    chamfer, hausdorff, hausdorff99, hausdorff95, hausdorff90, curvature_error = _chamfer_hausdorff_curvature(
        gt_edge, pred_edge, curvature_k=curvature_k)
    return {"chamfer": chamfer, "hausdorff": hausdorff,
            "hausdorff99": hausdorff99, "hausdorff95": hausdorff95, "hausdorff90": hausdorff90,
            "curvature_error": curvature_error,
            "n_gt_edge": n_gt, "n_pred_edge": n_pred}


def _shape_one_dir(pred_dir, gt_dir, args):
    """Run the shape/contour-fidelity metric over one pred dir. Same file
    discovery/grouping convention as _boundary_one_dir: one row per
    (stem, seed), not pre-averaged, so it can be joined/aggregated the same
    way as the other per-seed metrics."""
    pred_dir = Path(pred_dir)
    label = pred_dir.name
    files = sorted(pred_dir.rglob(args.pred_glob))
    parsed = [(f, *_base_and_variant(f.name)) for f in files]

    if args.use_median:
        kept = [p for p in parsed if p[2] == "median"]
        if not kept:
            print(f"[skip] {label}: --use-median set but no *_median_disp.npy found")
            return label, []
    else:
        kept = [p for p in parsed if p[2] != "median"]

    groups = defaultdict(list)
    for f, base, variant in kept:
        groups[base].append((f, variant))

    rows = []
    missing_gt, shape_skip = 0, 0
    for base in sorted(groups):
        gtp = _find_gt(gt_dir, base)
        if gtp is None:
            missing_gt += 1
            continue
        gt = _load_disp(gtp)
        for f, variant in groups[base]:
            pred = _load_disp(f)
            if pred.shape != gt.shape:
                shape_skip += 1
                continue
            m = shape_metrics(pred, gt, args.fgbg_t, args.min_abs_diff, args.curvature_k)
            if m is None:
                continue
            seed = variant[len("seed"):] if variant.startswith("seed") else variant
            rows.append({"stem": base, "seed": seed, **m})

    if missing_gt:
        print(f"  [{label}] no GT match for {missing_gt} stem(s)")
    if shape_skip:
        print(f"  [{label}] shape mismatch on {shape_skip} pred file(s) — skipped")
    return label, rows


SHAPE_SHOW = ["chamfer", "hausdorff", "hausdorff99", "hausdorff95", "hausdorff90", "curvature_error", "n_gt_edge", "n_pred_edge"]


def cmd_shape(args):
    gt_dir = Path(args.gt_dir)
    all_out = []
    for pred_dir in args.pred_dir:
        label, rows = _shape_one_dir(pred_dir, gt_dir, args)
        if not rows:
            print(f"[skip] {label}: no scorable pairs\n")
            continue

        n_stems = len({r["stem"] for r in rows})
        hdr = f"  {'metric':<16}{'Mean':>11}{'Std':>11}{'Min':>11}{'Max':>11}"
        print("=" * len(hdr))
        print(f" {label}   ({n_stems} samples x {len(rows) // max(n_stems, 1)} seed(s), "
              f"fgbg_t={args.fgbg_t}"
              f"{'  [median]' if args.use_median else ''})")
        print("=" * len(hdr))
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for c in SHAPE_SHOW:
            vals = [r[c] for r in rows if r.get(c) == r.get(c)]
            if not vals:
                continue
            a = np.array(vals, dtype=np.float64)
            print(f"  {c:<16}{a.mean():>11.4f}{a.std():>11.4f}{a.min():>11.4f}{a.max():>11.4f}")
        print()
        for r in rows:
            all_out.append({"model": label, **r})

    print("Boundary contours located independently in pred and GT via the same "
          "Depth Pro fgbg_depth rule (neighbour ratio > fgbg_t), so neither side "
          "gets a location shortcut. chamfer/hausdorff are in pixels, decoupled "
          "from disparity magnitude -- lower = predicted contour geometry "
          "matches GT more closely. curvature_error (px^-1) compares LOCAL SHAPE "
          "(sharp vs. rounded, straight vs. curved) at Chamfer-matched point pairs, "
          "independent of position -- a contour can score well on chamfer/hausdorff "
          "while still rounding off corners; curvature_error is what catches that.\n")

    if args.out_csv and all_out:
        cols = ["model", "stem", "seed"] + SHAPE_SHOW
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in all_out:
                w.writerow({k: r.get(k, "") for k in cols})
        print(f"Per-sample shape/contour-fidelity metrics saved → {args.out_csv}")


# ============================================================================ #
#  UNCERTAINTY vs ERROR  (per-pixel sample variance vs GT error)
# ============================================================================ #
#
# A diffusion model gets an uncertainty estimate for free: draw K seeds per
# scene, and the per-pixel spread across them is a self-assessed confidence
# signal computed without ever looking at GT. This tests whether that spread
# is meaningful: (1) is it concentrated at object boundaries, where the paper's
# thesis says the true answer is genuinely uncertain, and (2) does it actually
# predict where the model is wrong (correlate with error against GT). Boundary
# pixels are located via the identical _fgbg_edge_mask + win-dilation rule
# boundary_metrics() uses, so "boundary" means the same thing here as
# everywhere else in this file.

def _uncertainty_one_scene(seed_disps, gt, fgbg_t, win, min_abs_diff):
    """seed_disps: (K,H,W) stack of per-seed predicted disparity. Returns a
    dict of per-pixel maps, or None if GT is entirely invalid."""
    gt_f, finite = _safe_fill(gt)
    if not finite.any():
        return None
    std_map = np.std(seed_disps, axis=0)
    median_map = np.median(seed_disps, axis=0)
    err_map = np.abs(median_map - gt_f)
    edge = _fgbg_edge_mask(gt_f, fgbg_t, min_abs_diff=min_abs_diff)
    band = _binary_dilate(edge, win) & finite
    return {"std_map": std_map, "median_map": median_map, "err_map": err_map,
            "boundary_mask": band, "finite_mask": finite,
            "seed_disps": seed_disps, "gt": gt_f, "edge": edge}


def _uncertainty_one_dir(pred_dir, gt_dir, args):
    """Group per-seed prediction files by stem (dropping median/scc/other
    non-seed variants), compute per-scene uncertainty/error/boundary data.
    Returns (label, {stem: result_dict})."""
    pred_dir = Path(pred_dir)
    label = pred_dir.name
    files = sorted(pred_dir.rglob(args.pred_glob))
    parsed = [(f, *_base_and_variant(f.name)) for f in files]
    kept = [p for p in parsed if p[2] not in ("median", "plain")]  # seedN only

    groups = defaultdict(list)
    for f, base, variant in kept:
        groups[base].append(f)

    results = {}
    missing_gt = too_few_seeds = shape_skip = 0
    for base in sorted(groups):
        gtp = _find_gt(gt_dir, base)
        if gtp is None:
            missing_gt += 1
            continue
        gt = _load_disp(gtp)
        seed_files = groups[base]
        if len(seed_files) < 2:
            too_few_seeds += 1
            continue
        seed_disps, ok = [], True
        for f in seed_files:
            d = _load_disp(f)
            if d.shape != gt.shape:
                ok = False
                break
            seed_disps.append(d)
        if not ok:
            shape_skip += 1
            continue
        r = _uncertainty_one_scene(np.stack(seed_disps, axis=0), gt,
                                   args.fgbg_t, args.win, args.min_abs_diff)
        if r is not None:
            results[base] = r

    if missing_gt:
        print(f"  [{label}] no GT match for {missing_gt} stem(s)")
    if too_few_seeds:
        print(f"  [{label}] fewer than 2 seeds for {too_few_seeds} stem(s) — skipped")
    if shape_skip:
        print(f"  [{label}] shape mismatch on {shape_skip} stem(s) — skipped")
    return label, results


def _binned_scatter(ax, unc, err, n_bins, color, label, eps=1e-4):
    """Quantile-bin `unc`, plot mean err per bin, with the bin's std drawn as
    a light band (mean +/- std) running the length of the line -- the spread
    stays visible between points, not just at them, without individual
    error-bar caps cluttering the line itself.
    Values are floored at `eps` so the caller can safely use log-log axes --
    uncertainty/error are heavily right-skewed (a small tail of large-error
    boundary pixels), so linear axes let that tail dominate the plot and
    compress the well-populated low-error region into unreadability."""
    order = np.argsort(unc)
    unc_s, err_s = unc[order], err[order]
    edges = np.linspace(0, len(unc_s), n_bins + 1).astype(int)
    xs, ys, stds = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        u_chunk, e_chunk = unc_s[lo:hi], err_s[lo:hi]
        xs.append(max(eps, u_chunk.mean()))
        ys.append(max(eps, e_chunk.mean()))
        stds.append(float(e_chunk.std()))
    xs, ys, stds = np.array(xs), np.array(ys), np.array(stds)
    lo_band = np.clip(ys - stds, eps, None)
    hi_band = ys + stds
    ax.fill_between(xs, lo_band, hi_band, color=color, alpha=0.15, linewidth=0, zorder=1)
    ax.plot(xs, ys, 'o-', color=color, label=label, markersize=5, linewidth=2, zorder=3)


def _sparsification_curve(unc, err, fractions):
    """For each retention fraction f in `fractions` (e.g. 1.0, 0.95, ..., 0.05),
    compute mean error over the f*N pixels with LOWEST uncertainty (actual --
    what you'd get by discarding the (1-f) least-certain pixels), and
    separately over the f*N pixels with LOWEST true error (oracle -- the best
    any ranking could do, since it cheats and sorts by GT error directly).
    Returns (actual_means, oracle_means), same length as `fractions`. The gap
    between the two curves (AUSE, computed by the caller) is the standard
    summary of how well-calibrated the uncertainty ranking is: 0 = as good as
    the oracle, larger = uncertainty isn't actually tracking where errors are.
    """
    n = len(unc)
    order_unc = np.argsort(unc)   # ascending: most certain (lowest unc) first
    order_err = np.argsort(err)   # ascending: lowest true error first
    err_by_unc = err[order_unc]
    err_by_err = err[order_err]
    actual, oracle = [], []
    for f in fractions:
        k = max(1, int(round(f * n)))
        actual.append(float(err_by_unc[:k].mean()))
        oracle.append(float(err_by_err[:k].mean()))
    return np.array(actual), np.array(oracle)


def cmd_uncertainty(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label, results = _uncertainty_one_dir(args.pred_dir, gt_dir, args)
    if not results:
        print(f"[skip] {label}: no scorable scenes\n")
        return

    unc_all, err_all, is_boundary_all = [], [], []
    per_scene_mean_err = []
    for stem, r in results.items():
        m = r["finite_mask"]
        unc_all.append(r["std_map"][m])
        err_all.append(r["err_map"][m])
        is_boundary_all.append(r["boundary_mask"][m])
        per_scene_mean_err.append((stem, float(r["err_map"][m].mean())))

    unc_all = np.concatenate(unc_all)
    err_all = np.concatenate(err_all)
    is_boundary_all = np.concatenate(is_boundary_all)

    def _corr(u, e):
        if len(u) < 3 or _scipy_stats is None:
            return float("nan"), float("nan")
        return _scipy_stats.pearsonr(u, e)[0], _scipy_stats.spearmanr(u, e)[0]

    pear_pool, spear_pool = _corr(unc_all, err_all)
    pear_bnd, spear_bnd = _corr(unc_all[is_boundary_all], err_all[is_boundary_all])
    pear_int, spear_int = _corr(unc_all[~is_boundary_all], err_all[~is_boundary_all])
    mean_unc_bnd = float(unc_all[is_boundary_all].mean())
    mean_unc_int = float(unc_all[~is_boundary_all].mean())

    print("=" * 70)
    print(f" {label}  uncertainty vs error  ({len(results)} scenes, {len(unc_all):,} pixels)")
    print("=" * 70)
    print(f"  mean uncertainty (std, px):  boundary={mean_unc_bnd:.4f}   "
          f"interior={mean_unc_int:.4f}   ratio={mean_unc_bnd / max(mean_unc_int, 1e-9):.2f}x")
    print(f"  pooled    : pearson r={pear_pool:.3f}  spearman rho={spear_pool:.3f}")
    print(f"  boundary  : pearson r={pear_bnd:.3f}  spearman rho={spear_bnd:.3f}")
    print(f"  interior  : pearson r={pear_int:.3f}  spearman rho={spear_int:.3f}")
    print()

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["region", "n_pixels", "mean_uncertainty", "mean_error", "pearson_r", "spearman_rho"])
            w.writerow(["pooled", len(unc_all), float(unc_all.mean()), float(err_all.mean()), pear_pool, spear_pool])
            w.writerow(["boundary", int(is_boundary_all.sum()), mean_unc_bnd,
                       float(err_all[is_boundary_all].mean()), pear_bnd, spear_bnd])
            w.writerow(["interior", int((~is_boundary_all).sum()), mean_unc_int,
                       float(err_all[~is_boundary_all].mean()), pear_int, spear_int])
        print(f"Summary saved → {args.out_csv}")

    # ---- Combined figure: (a) binned scatter, (b) sparsification ----
    plt.rcParams.update({"font.size": 13})
    fig, (ax_scatter, ax_sparse) = plt.subplots(1, 2, figsize=(12.5, 3.6))

    # Panel (a) plots one scene rather than all pixels pooled across the test
    # set, so the trend isn't smoothed over scene-to-scene variation. Chosen
    # deterministically as the MEDIAN-difficulty scene by its own mean error
    # (same rule used below for the cross-section profile figure), not the
    # best case, so it can't be read as cherry-picked.
    per_scene_mean_err.sort(key=lambda t: t[1])
    example_stem = per_scene_mean_err[len(per_scene_mean_err) // 2][0]
    r_ex = results[example_stem]
    m_ex = r_ex["finite_mask"]
    unc_ex = r_ex["std_map"][m_ex]
    err_ex = r_ex["err_map"][m_ex]
    bnd_ex = r_ex["boundary_mask"][m_ex]

    _binned_scatter(ax_scatter, unc_ex[bnd_ex], err_ex[bnd_ex],
                    args.n_bins, "crimson", "Boundary pixels")
    _binned_scatter(ax_scatter, unc_ex[~bnd_ex], err_ex[~bnd_ex],
                    args.n_bins, "steelblue", "Interior pixels")
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel("Predicted uncertainty (px)")
    ax_scatter.set_ylabel("EPE (px)")
    ax_scatter.legend(frameon=False, fontsize=12, loc="upper left")
    ax_scatter.text(0.97, 0.05,
           f"scene: {example_stem} (median difficulty)\n"
           f"boundary r = {pear_bnd:.2f}  (pooled, all scenes)\n"
           f"interior r = {pear_int:.2f}  (pooled, all scenes)",
           transform=ax_scatter.transAxes, fontsize=9.5, ha="right", va="bottom",
           color="0.25")
    ax_scatter.spines[['top', 'right']].set_visible(False)
    ax_scatter.set_title("(a)", loc="left", fontsize=13)
    # x = fraction of pixels RETAINED (the most-certain ones, by predicted
    # uncertainty), 100% down to 5%; y = mean EPE among the retained pixels.
    # "actual" should fall as retention drops, since we're progressively
    # discarding the least-certain (and, if calibrated, highest-error) pixels.
    # "oracle" cheats by ranking on true error directly -- the best any
    # ranking could do, and the reference "actual" is compared against.
    # "random" is the flat line you'd get from a ranking with no signal at
    # all (dropping a random subset doesn't change the mean of what's left).
    fractions = np.round(np.arange(1.0, 0.0, -0.05), 2)  # 1.00, 0.95, ..., 0.05
    actual_curve, oracle_curve = _sparsification_curve(unc_all, err_all, fractions)
    random_level = float(err_all.mean())
    # fractions is descending, so trapz's sign is flipped relative to the
    # usual ascending-x convention -- abs() gives the (unsigned) area, which
    # is the standard AUSE reporting convention (lower = closer to oracle).
    ause = abs(float(np.trapz(actual_curve - oracle_curve, x=fractions)))

    print("  sparsification (retain most-certain X% -> mean EPE among retained):")
    for f, a, o in zip(fractions, actual_curve, oracle_curve):
        is_decile = np.isclose(round(f * 100) % 10, 0, atol=1e-6)
        if is_decile:
            print(f"    retain {f * 100:5.0f}%:  actual={a:.4f}  oracle={o:.4f}")
    print(f"  AUSE (area between actual and oracle, lower=better-calibrated): {ause:.5f}")
    print(f"  random-ranking baseline (flat, no signal): {random_level:.4f}")
    print()

    if args.out_csv:
        sparsify_csv = out_dir / f"{label}_sparsification.csv"
        with open(sparsify_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["retain_fraction", "actual_mean_epe", "oracle_mean_epe", "random_mean_epe"])
            for frac, a, o in zip(fractions, actual_curve, oracle_curve):
                w.writerow([frac, a, o, random_level])
        print(f"Sparsification table saved → {sparsify_csv}")

    ax_sparse.plot(fractions * 100, actual_curve, 'o-', color="darkorange", label="Predicted uncertainty", markersize=5, linewidth=2)
    ax_sparse.plot(fractions * 100, oracle_curve, 's--', color="0.35", label="True error", markersize=5, linewidth=2)
    RAFT_BEST_EPE = 0.129  # RAFT-Stereo, train-from-scratch, global EPE (Table 1) -- best baseline
    ax_sparse.axhline(RAFT_BEST_EPE, color="0.5", linestyle="--", linewidth=1.3, zorder=1)
    ax_sparse.text(0.45, RAFT_BEST_EPE, f"best baseline (RAFT-Stereo) = {RAFT_BEST_EPE:.3f}",
                   transform=ax_sparse.get_yaxis_transform(), fontsize=9.5,
                   color="0.4", ha="left", va="bottom")
    ax_sparse.invert_xaxis()  # 100% at left, dropping to the right as pixels are discarded
    ax_sparse.set_xlabel("Pixels retained (%)")
    ax_sparse.set_ylabel("Mean EPE (px)")
    ax_sparse.legend(frameon=False, fontsize=12)
    ax_sparse.spines[['top', 'right']].set_visible(False)
    ax_sparse.set_title("(b)", loc="left", fontsize=13)

    fig.tight_layout()
    combined_path = out_dir / f"{label}_uncertainty_combined.png"
    fig.savefig(combined_path, dpi=300)
    plt.close(fig)
    print(f"Combined uncertainty figure saved → {combined_path}")

    # ---- Figure 3: cross-section profile ----
    # Every individual sample's disparity value along one line crossing a
    # boundary, overlaid with GT. This shows the actual mechanism directly,
    # rather than an abstracted heatmap: samples commit to slightly different
    # boundary positions and fan apart exactly at the discontinuity, while
    # collapsing onto essentially the same value in flat regions. The line is
    # chosen automatically as whichever row/column crosses the most GT
    # boundary pixels in a representative (median-difficulty) scene, so it's
    # not cherry-picked.
    per_scene_mean_err.sort(key=lambda x: x[1])
    n = len(per_scene_mean_err)
    median_stem = per_scene_mean_err[n // 2][0]
    r = results[median_stem]
    edge = r["edge"]
    row_counts = edge.sum(axis=1)
    col_counts = edge.sum(axis=0)
    if row_counts.max() >= col_counts.max():
        cut_axis, cut_idx = "row", int(np.argmax(row_counts))
        gt_profile = r["gt"][cut_idx, :]
        seed_profiles = r["seed_disps"][:, cut_idx, :]
    else:
        cut_axis, cut_idx = "col", int(np.argmax(col_counts))
        gt_profile = r["gt"][:, cut_idx]
        seed_profiles = r["seed_disps"][:, :, cut_idx]

    fig, (ax_map, ax_prof) = plt.subplots(
        2, 1, figsize=(6.5, 6.2), gridspec_kw={"height_ratios": [1, 2]})

    im = ax_map.imshow(r["gt"], cmap="turbo")
    if cut_axis == "row":
        ax_map.axhline(cut_idx, color="white", linewidth=1.5)
    else:
        ax_map.axvline(cut_idx, color="white", linewidth=1.5)
    ax_map.set_xticks([]); ax_map.set_yticks([])
    plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

    xs = np.arange(len(gt_profile))
    for k in range(seed_profiles.shape[0]):
        ax_prof.plot(xs, seed_profiles[k], color="steelblue", alpha=0.4, linewidth=1.3,
                    label="Individual samples" if k == 0 else None)
    ax_prof.plot(xs, gt_profile, color="black", linewidth=2.2, label="Ground truth")
    ax_prof.set_xlabel("Position along line (px)")
    ax_prof.set_ylabel("Disparity (px)")
    ax_prof.legend(frameon=False, fontsize=12)
    ax_prof.grid(True, linestyle="-", linewidth=0.4, alpha=0.25)
    ax_prof.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    profile_path = out_dir / f"{label}_uncertainty_profile.png"
    fig.savefig(profile_path, dpi=300)
    plt.close(fig)
    print(f"Cross-section profile figure saved → {profile_path}\n")


# ============================================================================ #
#  CLI
# ============================================================================ #

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("csv", help="aggregate metrics.csv (our method or baselines)",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    c.add_argument("paths", nargs="+", help="metrics.csv files or their dirs")
    c.add_argument("--best-k", type=int, nargs="+", default=[3],
                   help="one or more K for oracle best-K-of-N (e.g. --best-k 1 3)")
    c.add_argument("--rank-metric", default="disp_epe",
                   help="metric used to rank seeds for best-K (default disp_epe; "
                        "falls back to any *epe* column for baselines)")
    c.add_argument("--ensemble", action="store_true",
                   help="also show the map-median row from the sibling "
                        "metrics_ensemble.csv (per-pixel median of seed prediction "
                        "maps; the headline single estimate). Skipped if absent.")
    c.add_argument("--boundary", action="store_true",
                   help="join in boundary_epe from the sibling boundary_metrics.csv "
                        "(matched by stem+seed) so it's aggregated/reported like any "
                        "other metric. Skipped if absent.")
    c.add_argument("--show", nargs="+", default=None,
                   help="metric columns to print (default: common disp/udf/baseline subset)")
    c.add_argument("--out-csv", default=None, help="write long-form results here")
    c.set_defaults(func=cmd_csv)

    b = sub.add_parser("boundary", help="boundary-sharpness metric from disp .npy folders",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    b.add_argument("--pred-dir", nargs="+", required=True,
                   help="one or more dirs of predicted disp .npy (searched recursively)")
    b.add_argument("--gt-dir", required=True, help="dir of GT disp .npy")
    b.add_argument("--pred-glob", default="*_disp.npy",
                   help="glob for predicted disp files (default *_disp.npy)")
    b.add_argument("--use-median", action="store_true",
                   help="score only *_median_disp.npy maps (our multi-seed runs); "
                        "default uses per-seed/flat maps and averages over seeds")
    b.add_argument("--fgbg-t", type=float, default=1.05,
                   help="Depth Pro fgbg ratio threshold defining a GT boundary "
                        "(neighbour disparity ratio > t; default 1.05 = a 5 percent jump)")
    b.add_argument("--win", type=int, default=2,
                   help="half-width (px) of the boundary band / local-max window (default 2)")
    b.add_argument("--min-abs-diff", type=float, default=0.5,
                   help="absolute disparity-difference floor (px) required alongside the "
                        "ratio test (default 0.5); guards against near-zero-disparity "
                        "background pixels being flagged as boundary from ratio noise "
                        "alone. Set 0 to recover the original ratio-only rule.")
    b.add_argument("--out-csv", default=None, help="write per-sample boundary metrics here")
    b.set_defaults(func=cmd_boundary)

    s = sub.add_parser("shape",
                       help="shape/contour-fidelity metric (Chamfer + Hausdorff) from disp .npy folders",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    s.add_argument("--pred-dir", nargs="+", required=True,
                   help="one or more dirs of predicted disp .npy (searched recursively)")
    s.add_argument("--gt-dir", required=True, help="dir of GT disp .npy")
    s.add_argument("--pred-glob", default="*_disp.npy",
                   help="glob for predicted disp files (default *_disp.npy)")
    s.add_argument("--use-median", action="store_true",
                   help="score only *_median_disp.npy maps (our multi-seed runs); "
                        "default uses per-seed/flat maps and averages over seeds")
    s.add_argument("--fgbg-t", type=float, default=1.05,
                   help="Depth Pro fgbg ratio threshold defining a boundary contour "
                        "(neighbour disparity ratio > t; default 1.05), applied "
                        "independently to GT and to each prediction")
    s.add_argument("--min-abs-diff", type=float, default=0.5,
                   help="absolute disparity-difference floor (px) required alongside the "
                        "ratio test (default 0.5); essential here -- without it, near-zero "
                        "background prediction noise floods the mask with false edges. "
                        "Set 0 to recover the original ratio-only rule.")
    s.add_argument("--curvature-k", type=int, default=8,
                   help="neighbours used for the local circle fit that estimates curvature "
                        "at each contour point (default 8); larger = smoother/less local")
    s.add_argument("--out-csv", default=None, help="write per-sample shape metrics here")
    s.set_defaults(func=cmd_shape)

    u = sub.add_parser("uncertainty",
                       help="per-pixel sample-variance uncertainty vs GT error, from multi-seed disp .npy folders",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    u.add_argument("--pred-dir", required=True,
                   help="dir of multi-seed predicted disp .npy (searched recursively; needs >=2 seeds/scene)")
    u.add_argument("--gt-dir", required=True, help="dir of GT disp .npy")
    u.add_argument("--pred-glob", default="*_disp.npy",
                   help="glob for predicted disp files (default *_disp.npy)")
    u.add_argument("--out-dir", required=True, help="dir to save figures (and --out-csv if given)")
    u.add_argument("--out-csv", default=None, help="write pooled/boundary/interior summary stats here")
    u.add_argument("--fgbg-t", type=float, default=1.05,
                   help="Depth Pro fgbg ratio threshold defining a GT boundary (default 1.05)")
    u.add_argument("--win", type=int, default=2,
                   help="half-width (px) of the boundary band (default 2)")
    u.add_argument("--min-abs-diff", type=float, default=0.5,
                   help="absolute disparity-difference floor (px) alongside the ratio test (default 0.5)")
    u.add_argument("--n-bins", type=int, default=15,
                   help="quantile bins for the binned scatter plot (default 15)")
    u.add_argument("--example-percentiles", type=float, nargs="+", default=[10, 40, 70, 95],
                   help="per-scene mean-error percentiles to show in the qualitative figure "
                        "(default 10 40 70 95, i.e. easy to hard)")
    u.set_defaults(func=cmd_uncertainty)

    r = sub.add_parser("recompute",
                       help="compute all metrics (RMSE, EPE, Bad0.5–4) from .npy files",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    r.add_argument("paths", nargs="+", help="pred dirs (searched recursively for *_disp.npy)")
    r.add_argument("--gt-dir", required=True, help="dir of GT disparity .npy files")
    r.add_argument("--best-k", type=int, nargs="+", default=[1],
                   help="one or more K for oracle best-K-of-N (default 1)")
    r.add_argument("--use-median", action="store_true",
                   help="score only *_median_disp.npy maps; default averages over seeds")
    r.set_defaults(func=cmd_recompute)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
