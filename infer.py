#!/usr/bin/env python3
"""
Stereo inference + alternating-UDF experiment.

Single seed:
    python infer.py --checkpoint model.pth \\
                    --left-dir data/left --right-dir data/right \\
                    --gt-field-dir data/fields --gt-disp-dir data/disp \\
                    --out-dir results/

Multi-seed alternating experiment (key scientific use-case):
    python infer.py --checkpoint model.pth \\
                    --left-dir data/left --right-dir data/right \\
                    --gt-field-dir data/fields --gt-disp-dir data/disp \\
                    --seeds 0 1 2 3 4 --out-dir results/

Posterior-spread experiment (needs >=8-10 seeds for a stable std estimate):
    python infer.py --checkpoint model.pth \\
                    --left-dir data/left --right-dir data/right \\
                    --gt-disp-dir data/disp --gt-field-dir data/fields \\
                    --seeds 0 1 2 3 4 5 6 7 8 9 --spread-map --out-dir results/

Per-seed: saves a 3-row plot (inputs | GT UDF + pred UDF | GT disp + pred disp).
Multi-seed: additionally saves an alternating overview plot — all UDF/disp predictions
side-by-side so you can see if the model alternates between depth orderings.
--spread-map: additionally computes per-pixel cross-seed std (disp + UDF), saves it as
    *_disp_std.npy / *_udf_std.npy plus a heatmap figure with the GT boundary contour
    overlaid, and (with GT disp) splits mean std into GT-boundary-band vs interior
    (same band definition as report_metrics.py boundary) into spread_metrics.csv.
    Tests whether the model's sampling uncertainty concentrates at depth boundaries
    (ambiguous: occlusion / edge ownership) vs surface interiors (well-posed matching).
    Also saves the cross-seed MEAN disp (*_mean_disp.npy) — compare its
    report_metrics.py boundary sharp_ratio against the per-seed sharp_ratio to test
    whether averaging samples reproduces SOTA-style edge over-smoothing.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Optional, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import k_diffusion as K

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


# ── loaders ───────────────────────────────────────────────────────────────────

def load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    return transforms.ToTensor()(Image.open(path).convert("RGB")).to(device)


def load_npy(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    while arr.ndim > 2:
        arr = arr[..., 0] if arr.shape[-1] == 1 else arr[0]
    return arr


def make_cond(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.cat([left, right], dim=0)  # (6,H,W)


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    return (t.cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ── tiling ────────────────────────────────────────────────────────────────────
#
# tile_overlap=0.0 / tile_pad_mode="constant" (the defaults) reproduce the
# original non-overlapping, zero-padded grid EXACTLY: stride == patch means
# zero overlap, so every pixel is covered by exactly one tile and the blend
# window cancels out in the weighted average regardless of its shape; and
# pad_mode only matters when pad>0. Callers that don't pass these params (i.e.
# every existing invocation except the SceneFlow script) are unaffected.

def _pad_to_multiple(img: torch.Tensor, pH: int, pW: int, pad_mode: str) -> torch.Tensor:
    C, H, W = img.shape
    pad_h, pad_w = pH - H, pW - W
    if pad_h == 0 and pad_w == 0:
        return img
    if pad_mode == "reflect":
        # reflect requires pad < corresponding input dim; fall back to
        # replicate (edge) padding if the image is too small for that.
        mode = "reflect" if (pad_h < H and pad_w < W) else "replicate"
        return F.pad(img.unsqueeze(0), (0, pad_w, 0, pad_h), mode=mode).squeeze(0)
    return F.pad(img, (0, pad_w, 0, pad_h))  # "constant", old zero-pad behavior


def _tile_positions(total_padded: int, patch: int, stride: int) -> List[int]:
    if total_padded <= patch:
        return [0]
    positions = list(range(0, total_padded - patch + 1, stride))
    if positions[-1] != total_padded - patch:
        positions.append(total_padded - patch)  # ensure exact coverage to the far edge
    return positions


def _blend_window(patch: int, device: torch.device) -> torch.Tensor:
    """Hann-ish 2D window, floor-clamped so weight is never exactly zero.

    With no overlap (stride==patch) each pixel is covered by exactly one tile,
    so this cancels out of the weighted average and has no effect on the
    result. With overlap>0 it cross-fades adjacent tiles smoothly instead of
    the hard-seam placement of the old grid untile().
    """
    if patch == 1:
        return torch.ones(1, 1, device=device)
    n = torch.arange(patch, device=device, dtype=torch.float32)
    w1d = (0.5 - 0.5 * torch.cos(2 * math.pi * n / (patch - 1))).clamp_min(1e-3)
    return w1d[:, None] * w1d[None, :]


def blend_tiles(tiles: torch.Tensor, tile_pos: List[Tuple[int, int]], patch: int,
                pH: int, pW: int, orig_H: int, orig_W: int,
                window: torch.Tensor) -> torch.Tensor:
    """(N,C,patch,patch) tiles at `tile_pos` (y,x) -> blended (C,orig_H,orig_W).

    Weighted-average accumulation: a pixel covered by only one tile (e.g. at
    the image boundary, or anywhere when tile_pos is the old non-overlapping
    grid) reproduces that tile's value exactly, since window cancels out of
    canvas/weight regardless of its value there.
    """
    C = tiles.shape[1]
    canvas = torch.zeros(C, pH, pW, device=tiles.device)
    weight = torch.zeros(1, pH, pW, device=tiles.device)
    for idx, (y, x) in enumerate(tile_pos):
        canvas[:, y:y+patch, x:x+patch] += tiles[idx] * window
        weight[:, y:y+patch, x:x+patch] += window
    return (canvas / weight)[:, :orig_H, :orig_W]


def _make_tile_grid(H: int, W: int, patch: int, overlap: float):
    """Return (pH, pW, tile_pos) for an image of size H,W. overlap=0.0 gives the
    old exact non-overlapping i*patch, j*patch grid (row-major order)."""
    pH = math.ceil(H / patch) * patch
    pW = math.ceil(W / patch) * patch
    stride = max(1, round(patch * (1.0 - overlap)))
    ys = _tile_positions(pH, patch, stride)
    xs = _tile_positions(pW, patch, stride)
    tile_pos = [(y, x) for y in ys for x in xs]
    return pH, pW, tile_pos


def _wide_tiles(padded_left: torch.Tensor, padded_right: torch.Tensor,
                tile_pos: List[Tuple[int, int]], patch: int, wide_px: int,
                device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-tile wide L/R crops for LeanCorrelationConditionerC2F's coarse
    stage -- same centering convention as FoJStereoDataset's _wide_crop
    (used at training time): each tile is horizontally CENTERED within a
    `wide_px`-wide window, edge-replicated where that window runs past the
    (already patch-padded) image's real bounds.
    Returns (L_wide (N,3,patch,wide_px), R_wide (N,3,patch,wide_px)).
    """
    _, pH_full, pW_full = padded_left.shape
    N = len(tile_pos)
    L_wide = torch.zeros(N, 3, patch, wide_px, device=device)
    R_wide = torch.zeros(N, 3, patch, wide_px, device=device)
    for idx, (y, x) in enumerate(tile_pos):
        tile_center = x + patch / 2.0
        wl = int(round(tile_center - wide_px / 2.0))
        wr = wl + wide_px
        pad_left = max(0, -wl)
        pad_right = max(0, wr - pW_full)
        wl_c, wr_c = max(0, wl), min(pW_full, wr)
        cropL = padded_left[:, y:y + patch, wl_c:wr_c]
        cropR = padded_right[:, y:y + patch, wl_c:wr_c]
        if pad_left > 0 or pad_right > 0:
            cropL = F.pad(cropL, (pad_left, pad_right), mode="replicate")
            cropR = F.pad(cropR, (pad_left, pad_right), mode="replicate")
        L_wide[idx] = cropL
        R_wide[idx] = cropR
    return L_wide, R_wide


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model_ema, cond6: torch.Tensor, patch_size: int,
                  input_channels: int, sigma_min: float, sigma_max: float,
                  n_steps: int, device: torch.device, seed: int,
                  tile_overlap: float = 0.0, tile_pad_mode: str = "constant",
                  use_scc_c2f: bool = False, coarse_strip_px: int = 512,
                  det_sigma: Optional[float] = None):
    """Return (pred (C,H,W), tiles_pred (N,C,ps,ps), tiles_cond (N,6,ps,ps), tile_pos, (pH,pW), wide_tiles).
    wide_tiles is (L_wide, R_wide) when use_scc_c2f, else None -- reused by
    run_scc so it isn't rebuilt a second time."""
    _, H, W = cond6.shape

    pH, pW, tile_pos = _make_tile_grid(H, W, patch_size, tile_overlap)
    padded_left  = _pad_to_multiple(cond6[:3], pH, pW, tile_pad_mode)
    padded_right = _pad_to_multiple(cond6[3:], pH, pW, tile_pad_mode)
    padded_cond = torch.cat([padded_left, padded_right], dim=0)

    N = len(tile_pos)
    tiles_cond = torch.zeros(N, 6, patch_size, patch_size, device=device)
    for idx, (y, x) in enumerate(tile_pos):
        tiles_cond[idx] = padded_cond[:, y:y+patch_size, x:x+patch_size]

    extra_args: dict = {"aug_cond": tiles_cond}

    wide_tiles = None
    if use_scc_c2f:
        L_wide, R_wide = _wide_tiles(padded_left, padded_right, tile_pos,
                                     patch_size, coarse_strip_px, device)
        extra_args["scc_coarse_L"] = L_wide
        extra_args["scc_coarse_R"] = R_wide
        wide_tiles = (L_wide, R_wide)

    if det_sigma is not None:
        # deterministic regressor: single forward pass from a zero input (seed is irrelevant)
        x_0 = model_ema.inner_model(
            torch.zeros(N, input_channels, patch_size, patch_size, device=device),
            torch.full((N,), det_sigma, device=device), **extra_args)
    else:
        gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(N, input_channels, patch_size, patch_size,
                        generator=gen, device=device) * sigma_max

        sigmas = K.sampling.get_sigmas_karras(n_steps, sigma_min, sigma_max,
                                              rho=7., device=device)
        x_0 = K.sampling.sample_dpmpp_2m_sde(
            model_ema, x, sigmas,
            extra_args=extra_args,
            eta=0.0, solver_type="heun", disable=True,
        )
    window = _blend_window(patch_size, device)
    pred = blend_tiles(x_0, tile_pos, patch_size, pH, pW, H, W, window)
    return pred, x_0, tiles_cond, tile_pos, (pH, pW), wide_tiles


@torch.no_grad()
def run_scc(inner_model, tiles_cond: torch.Tensor, tile_pos: List[Tuple[int, int]],
            pH: int, pW: int, patch_size: int, orig_H: int, orig_W: int,
            wide_tiles: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
    """Token-level LCC (use_scc): disparity at 32x32 token grid, upsampled to tile size.

    wide_tiles: (L_wide, R_wide) from run_inference -- required when
    inner_model.use_scc_c2f is True (the coarse-to-fine LCC needs a wide
    crop, not just the tile itself; see LeanCorrelationConditionerC2F).

    Returns (disp (H,W) px, valid (H,W) {0,1}) on the full image grid.
    """
    L, R = tiles_cond[:, :3], tiles_cond[:, 3:]
    if getattr(inner_model, "use_scc_c2f", False):
        if wide_tiles is None:
            raise ValueError("inner_model.use_scc_c2f=True requires wide_tiles "
                             "(pass run_inference's returned wide_tiles through).")
        L_wide, R_wide = wide_tiles
        dino_cv = L_dino = R_dino = dino_radius = None
        if getattr(inner_model, "use_scc_dino_cv", False):
            dino_cv = inner_model.dino_cv
            crop_px = inner_model.scc_dino_c2f_crop_px
            off = (L_wide.shape[-1] - crop_px) // 2
            L_dino, R_dino = L_wide[:, :, :, off:off + crop_px], R_wide[:, :, :, off:off + crop_px]
            dino_radius = inner_model.scc_dino_c2f_radius_px
        _, disp_tok, _, valid_tok, _ = inner_model.scc(
            L_wide, R_wide, dino_cv=dino_cv, L_dino=L_dino, R_dino=R_dino, dino_radius_px=dino_radius)
    else:
        dino_cv_feat = None
        if getattr(inner_model, "use_scc_dino_cv", False):
            stride = int(inner_model.scc.px_per_token)
            Ht_lcc = patch_size // stride
            Wt_lcc = patch_size // stride
            dino_cv_feat = inner_model.dino_cv(L, R, Ht_lcc, Wt_lcc)
        _, disp_tok, _, valid_tok, _ = inner_model.scc(L, R, extra_cv=dino_cv_feat)   # (N,1,Ht,Wt), pixels
    up = lambda t: F.interpolate(t, size=(patch_size, patch_size), mode="nearest")
    window = _blend_window(patch_size, tiles_cond.device)
    disp = blend_tiles(up(disp_tok),  tile_pos, patch_size, pH, pW, orig_H, orig_W, window)[0].cpu().numpy()
    valid = blend_tiles(up(valid_tok), tile_pos, patch_size, pH, pW, orig_H, orig_W, window)[0].cpu().numpy()
    return disp, valid


def run_scc_native(inner_model, tiles_cond: torch.Tensor, tile_pos: List[Tuple[int, int]],
                   pH: int, pW: int, patch_size: int, orig_H: int, orig_W: int):
    """Native-resolution LCC (use_scc_native): disparity already at full image resolution.

    No upsampling needed — scc_native output is natively (N,1,H,W) at tile resolution.
    Returns (disp (H,W) px, valid (H,W) {0,1}) on the full image grid.
    """
    L, R = tiles_cond[:, :3], tiles_cond[:, 3:]
    disp_px, _, valid = inner_model.scc_native(L, R)   # (N,1,H,W), already image resolution
    window = _blend_window(patch_size, tiles_cond.device)
    disp = blend_tiles(disp_px.detach(), tile_pos, patch_size, pH, pW, orig_H, orig_W, window)[0].cpu().numpy()
    valid_out = blend_tiles(valid.detach(), tile_pos, patch_size, pH, pW, orig_H, orig_W, window)[0].cpu().numpy()
    return disp, valid_out


# ── decode channels ───────────────────────────────────────────────────────────

def decode_pred(pred: torch.Tensor, udf_scale: float, udf_k: float, disp_norm: float,
                udf_channel_weight: float = 1.0, udf_linear: bool = False):
    """
    2-channel: returns (udf_px, None, disp_px)
    3-channel: returns (udf1_px, udf2_px, disp_px)

    Inverts the UDF channel according to how it was trained:
      exponential: (1 - exp(-k * udf_norm)) * w  →  -log(1 - t/w) / k * u_scale
      linear:       udf_norm                     →  t * u_scale          (udf_linear=True)
      disp: t / disp_norm  →  t * disp_norm
    """
    p = pred.cpu().float().numpy()  # (C, H, W)

    def _udf(ch):
        t = np.clip(p[ch] / udf_channel_weight, 0.0, 1.0 - 1e-6)
        if udf_linear:
            return t * udf_scale
        return -np.log(1.0 - t) / udf_k * udf_scale

    if p.shape[0] == 1:
        return None, None, p[0] * disp_norm
    elif p.shape[0] == 3:
        return _udf(0), _udf(1), p[2] * disp_norm
    else:
        return _udf(0), None, (p[1] * disp_norm if p.shape[0] >= 2 else None)


# ── metrics ───────────────────────────────────────────────────────────────────

# Bad-X thresholds (px). _badkey(2.0) -> "bad2" so existing references keep working.
THRESHOLDS = [0.5, 1.0, 2.0, 3.0, 4.0]
def _badkey(t):
    return f"bad{t:g}"


def compute_metrics(pred: np.ndarray, gt: np.ndarray,
                    seg_mask: Optional[np.ndarray] = None) -> dict:
    mask = np.isfinite(gt) & np.isfinite(pred)
    if seg_mask is not None:
        mask = mask & (seg_mask > 0)
    if not mask.any():
        d = {"mse": float("nan"), "rmse": float("nan"), "epe": float("nan")}
        d.update({_badkey(t): float("nan") for t in THRESHOLDS})
        return d
    diff = pred[mask] - gt[mask]
    abs_diff = np.abs(diff)
    d = {
        "mse":  float(np.mean(diff ** 2)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "epe":  float(np.mean(abs_diff)),
    }
    d.update({_badkey(t): float(np.mean(abs_diff > t) * 100) for t in THRESHOLDS})
    return d


# ── posterior-spread (cross-seed variance) experiment ────────────────────────
#
# Question this answers: does the model's sampling variability concentrate on
# depth boundaries (where stereo is geometrically ambiguous: half-occlusion,
# edge ownership) and vanish in surface interiors (where matching is
# well-posed)? Per sample we compute the per-pixel std of the disparity (and
# UDF) predictions across seeds, then split it into a GT boundary band vs
# interior using the SAME boundary definition as report_metrics.py `boundary`
# (Depth Pro fgbg contours on GT disp, ratio > fgbg_t, dilated by win) so the
# spread numbers line up with the existing boundary_epe / sharp_ratio tables.
# The helpers below are copied from report_metrics.py (kept in sync by hand —
# it lives outside this repo).

def _grey_dilate(a, radius):
    if radius <= 0:
        return a
    pad = np.pad(a, radius, mode="edge")
    H, W = a.shape
    out = a.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out = np.maximum(out, pad[radius + dy:radius + dy + H,
                                      radius + dx:radius + dx + W])
    return out


def _binary_dilate(mask, radius):
    if radius <= 0:
        return mask
    return _grey_dilate(mask.astype(np.float32), radius) > 0.5


def _fgbg_edge_mask(d, t, eps=1e-6):
    """Depth Pro fgbg contours (neighbour disparity ratio > t) collapsed to a
    per-pixel boundary mask; both pixels of each contributing pair are marked."""
    d = np.clip(d.astype(np.float64), eps, None)
    h = ((d[:, 1:] / d[:, :-1]) > t) | ((d[:, :-1] / d[:, 1:]) > t)
    v = ((d[1:, :] / d[:-1, :]) > t) | ((d[:-1, :] / d[1:, :]) > t)
    edge = np.zeros(d.shape, dtype=bool)
    edge[:, :-1] |= h
    edge[:, 1:] |= h
    edge[:-1, :] |= v
    edge[1:, :] |= v
    return edge


def compute_spread_stats(std_map: np.ndarray, gt_disp: np.ndarray,
                         fgbg_t: float, win: int) -> dict:
    """Split a per-pixel cross-seed std map into GT boundary band vs interior.

    Returns bnd/int mean std, their ratio, and the variance concentration:
    share of total variance (sum of std^2) inside the band divided by the
    band's share of pixels (1.0 = spread is uniform; >>1 = spread lives at
    boundaries).
    """
    finite = np.isfinite(gt_disp) & np.isfinite(std_map)
    if not finite.any():
        return {}
    gt_f = gt_disp.astype(np.float64).copy()
    gt_f[~np.isfinite(gt_disp)] = float(np.median(gt_disp[np.isfinite(gt_disp)]))
    edge = _fgbg_edge_mask(gt_f, fgbg_t) & finite
    band = _binary_dilate(edge, win) & finite
    interior = finite & ~band
    var = std_map.astype(np.float64) ** 2
    bnd_std = float(np.mean(std_map[band])) if band.any() else float("nan")
    int_std = float(np.mean(std_map[interior])) if interior.any() else float("nan")
    band_frac = float(band.sum()) / float(finite.sum())
    total_var = float(var[finite].sum())
    var_share = float(var[band].sum()) / total_var if total_var > 0 else float("nan")
    return {
        "band_frac_pct": band_frac * 100.0,
        "bnd_std":       bnd_std,
        "int_std":       int_std,
        "std_ratio":     bnd_std / int_std if int_std > 0 else float("nan"),
        "var_share_pct": var_share * 100.0,
        "concentration": var_share / band_frac if band_frac > 0 else float("nan"),
    }


def save_spread_plot(out_path: Path, stem: str, n_seeds: int,
                     left_np: np.ndarray,
                     disp_std: Optional[np.ndarray],
                     udf_std: Optional[np.ndarray],
                     gt_disp_px: Optional[np.ndarray],
                     fgbg_t: float,
                     stats: dict):
    """One row: Left | GT disp | disp std (+GT edge contour) | UDF std."""
    panels = [("Left", left_np, None)]
    if gt_disp_px is not None:
        panels.append(("GT disp (px)", gt_disp_px, "plasma"))
    if disp_std is not None:
        tag = (f"  bnd/int={stats['std_ratio']:.1f}x  conc={stats['concentration']:.1f}x"
               if stats.get("std_ratio") == stats.get("std_ratio") else "")
        panels.append((f"disp std across {n_seeds} seeds (px){tag}", disp_std, "inferno"))
    if udf_std is not None:
        panels.append((f"UDF std across {n_seeds} seeds (px)", udf_std, "inferno"))

    fig, axes = plt.subplots(1, len(panels), figsize=(len(panels) * 4.2, 4.2),
                             squeeze=False)
    edge = None
    if gt_disp_px is not None and np.isfinite(gt_disp_px).any():
        gt_f = gt_disp_px.astype(np.float64).copy()
        gt_f[~np.isfinite(gt_disp_px)] = float(np.median(gt_disp_px[np.isfinite(gt_disp_px)]))
        edge = _fgbg_edge_mask(gt_f, fgbg_t)
    for ax, (title, img, cmap) in zip(axes[0], panels):
        _ax(ax, img, title, cmap=cmap, colorbar=cmap is not None)
        if cmap == "inferno" and edge is not None:
            ax.contour(edge.astype(np.float32), levels=[0.5], colors="cyan",
                       linewidths=0.5, alpha=0.7)
    fig.suptitle(f"{stem} — posterior spread (cyan = GT depth boundary)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── stem index helper ─────────────────────────────────────────────────────────

def _stem_index(stem: str) -> Optional[int]:
    """Return the last integer in a stem name, e.g. 'two_object_105' -> 105."""
    parts = stem.replace('-', '_').split('_')
    for p in reversed(parts):
        if p.isdigit():
            return int(p)
    return None


# ── t-SNE bimodality plot ─────────────────────────────────────────────────────

def _run_tsne_plot(stem: str, pred_udfs_list: list, seed_ids: list,
                   gt_udf_A: Optional[np.ndarray], gt_udf_B: Optional[np.ndarray],
                   out_path: Path) -> float:
    """t-SNE + KMeans(2) bimodality analysis for one sample. Returns silhouette score."""
    if not _SKLEARN_OK:
        print(f"  [tsne] sklearn not available; skipping {stem}")
        return float("nan")
    n_seeds = len(pred_udfs_list)
    if n_seeds < 3:
        print(f"  [tsne] too few seeds ({n_seeds}) for {stem}; skipping")
        return float("nan")

    udfs = np.stack([u.flatten() for u in pred_udfs_list])  # (n_seeds, HW)

    n_pca = min(n_seeds - 1, 50)
    pca = PCA(n_components=n_pca)
    seed_pca = pca.fit_transform(udfs)

    perplexity = max(2, n_seeds // 3)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                max_iter=1000, init='pca')
    seed_emb = tsne.fit_transform(seed_pca)

    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(seed_emb)
    sil = float(silhouette_score(seed_emb, cluster_labels))

    # Place GT modes by k-NN interpolation in PCA space
    gt_emb, gt_label_list = [], []
    seed_mean = udfs.mean(); seed_std = udfs.std() + 1e-8
    nn_model = NearestNeighbors(n_neighbors=min(3, n_seeds)).fit(seed_pca)
    for gt_arr, lbl in [(gt_udf_A, 'GT A'), (gt_udf_B, 'GT B')]:
        if gt_arr is None:
            continue
        gv = gt_arr.flatten()
        gv_rescaled = (gv - gv.mean()) / (gv.std() + 1e-8) * seed_std + seed_mean
        gv_pca = pca.transform(gv_rescaled[None])
        dists, idxs = nn_model.kneighbors(gv_pca)
        weights = 1.0 / (dists[0] + 1e-8); weights /= weights.sum()
        gt_emb.append((seed_emb[idxs[0]] * weights[:, None]).sum(axis=0))
        gt_label_list.append(lbl)

    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ['#e6194b', '#4363d8']
    for c in [0, 1]:
        mask = cluster_labels == c
        ax.scatter(seed_emb[mask, 0], seed_emb[mask, 1], color=colors[c], s=90,
                   alpha=0.85, edgecolors='k', linewidths=0.5, label=f'cluster {c}')
    for sid, (x, y) in zip(seed_ids, seed_emb):
        ax.annotate(str(sid), (x, y), fontsize=6, ha='center', va='center', color='white')
    gt_colors = ['#f58231', '#3cb44b']
    for i, (xy, lbl) in enumerate(zip(gt_emb, gt_label_list)):
        ax.scatter(xy[0], xy[1], color=gt_colors[i], s=300, marker='*',
                   edgecolors='k', linewidths=0.8, zorder=5, label=lbl)
    ax.set_title(f"{stem}  |  silhouette = {sil:.3f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8)
    fig.suptitle("t-SNE of UDF outputs\nRed/blue = KMeans(2). Stars = GT modes.", fontsize=9)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [tsne] {stem}: silhouette={sil:.3f} → {out_path}")
    return sil


# ── per-split aggregate CSV ───────────────────────────────────────────────────

def _write_agg_csv(csv_rows: list, path: Path):
    """Write mean / median / best1-of-N summary CSV from per-(stem,seed) rows."""
    if not csv_rows:
        return
    samples: dict = {}
    for r in csv_rows:
        s = r["stem"]
        if s not in samples:
            samples[s] = []
        samples[s].append({k: v for k, v in r.items() if k not in ("stem", "seed")})

    metric_keys = [k for k in csv_rows[0] if k not in ("stem", "seed")]
    n_seeds = max(len(v) for v in samples.values()) if samples else 0
    rank = "disp_epe" if "disp_epe" in metric_keys else (metric_keys[0] if metric_keys else "")

    def _nanmean(vals):
        v = [x for x in vals if x == x and not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(v)) if v else float("nan")

    def _nanmedian(vals):
        v = [x for x in vals if x == x and not (isinstance(x, float) and np.isnan(x))]
        return float(np.median(v)) if v else float("nan")

    results = []
    for mode_name, mode in [("mean", "mean"), ("median", "median"),
                             (f"best1_of_{n_seeds}", "best1")]:
        per_sample = []
        for seeds_data in samples.values():
            if mode == "mean":
                m = {c: _nanmean([s.get(c, float("nan")) for s in seeds_data])
                     for c in metric_keys}
            elif mode == "median":
                m = {c: _nanmedian([s.get(c, float("nan")) for s in seeds_data])
                     for c in metric_keys}
            else:  # best1
                valid = [s for s in seeds_data
                         if rank and not (isinstance(s.get(rank, float("nan")), float)
                                          and np.isnan(s.get(rank, float("nan"))))]
                best = min(valid or list(seeds_data),
                           key=lambda s: float(s.get(rank, float("inf"))))
                m = {c: best.get(c, float("nan")) for c in metric_keys}
            per_sample.append(m)
        agg = {c: _nanmean([s[c] for s in per_sample]) for c in metric_keys}
        results.append({"aggregation": mode_name, "n_samples": len(per_sample),
                        "n_seeds": n_seeds, **agg})

    fields = ["aggregation", "n_samples", "n_seeds"] + metric_keys
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nAggregate metrics (mean / median / best1-of-{n_seeds}) → {path}")
    show_cols = [c for c in ["disp_epe", "disp_rmse", "disp_bad0.5", "disp_bad1",
                              "disp_bad2", "disp_bad3", "disp_bad4", "udf_epe", "udf_bad2"]
                 if c in metric_keys]
    if show_cols:
        hdr = f"  {'':>18}  " + " ".join(f"{c.replace('disp_', ''):>8}" for c in show_cols)
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for r in results:
            vals = " ".join(
                (f"{float(r.get(c, float('nan'))):8.2f}" if "bad" in c
                 else f"{float(r.get(c, float('nan'))):8.4f}")
                for c in show_cols)
            print(f"  {r['aggregation']:>18}  {vals}")


# ── cross-model comparison table (shared summary CSV across runs) ──────────────

def _mean_field(rows, key):
    vals = [r[key] for r in rows if key in r and r[key] == r[key]]  # drop NaN
    return float(np.mean(vals)) if vals else float("nan")


def _summary_fields():
    bad = [f"disp_{_badkey(t)}" for t in THRESHOLDS]
    return ["model", "n_seeds", "disp_epe", "disp_epe_med", "disp_rmse"] + bad \
           + ["udf_epe", "udf_bad2"]


def append_summary_row(summary_csv, model_name, csv_rows, ensemble_rows, n_seeds):
    """Append/replace this model's aggregate row in a shared summary CSV."""
    row = {"model": model_name, "n_seeds": n_seeds,
           "disp_epe":     _mean_field(csv_rows, "disp_epe"),
           "disp_epe_med": _mean_field(ensemble_rows, "disp_epe"),
           "disp_rmse":    _mean_field(csv_rows, "disp_rmse"),
           "udf_epe":      _mean_field(csv_rows, "udf_epe"),
           "udf_bad2":     _mean_field(csv_rows, "udf_bad2")}
    for t in THRESHOLDS:
        row[f"disp_{_badkey(t)}"] = _mean_field(csv_rows, f"disp_{_badkey(t)}")

    fields = _summary_fields()
    existing = {}
    if Path(summary_csv).exists():
        with open(summary_csv) as f:
            for r in csv.DictReader(f):
                existing[r["model"]] = r
    existing[model_name] = row
    Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in existing.values():
            w.writerow({k: r.get(k, "") for k in fields})


def print_summary_table(summary_csv):
    """Print the cross-model comparison table from the shared summary CSV."""
    if not Path(summary_csv).exists():
        print(f"\n[summary] no summary CSV at {summary_csv}")
        return
    with open(summary_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    def fv(r, k):
        try:
            return float(r.get(k, "nan"))
        except (TypeError, ValueError):
            return float("nan")

    rows.sort(key=lambda r: fv(r, "disp_epe"))
    cols = ["disp_epe", "disp_epe_med"] + [f"disp_{_badkey(t)}" for t in THRESHOLDS]
    hdr = f"{'model':54s} {'sd':>3} " + " ".join(
        f"{c.replace('disp_', ''):>8}" for c in cols) + f" {'udf_epe':>8}"
    bar = "=" * len(hdr)
    print(f"\n{bar}\n FINAL MODEL COMPARISON   (sorted by disp EPE; Bad-X = % |err| > X px)\n{bar}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cells = " ".join(
            (f"{fv(r, c):8.4f}" if "epe" in c else f"{fv(r, c):8.2f}") for c in cols)
        print(f"{r['model']:54s} {fv(r, 'n_seeds'):3.0f} {cells} {fv(r, 'udf_epe'):8.4f}")
    print(bar)


# ── display-only despeckle (visualization aid; never applied to saved .npy
#    predictions or metrics -- those still see the model's real, raw output) ──

def _despeckle_for_display(arr, size=7, iso_factor=8.0, range_factor=2.5):
    """Replace isolated salt-and-pepper outlier pixels with their local median,
    for plotting only.

    A pixel must fail BOTH checks to get touched:
      1. locally inconsistent -- its value disagrees sharply with its own
         local neighborhood (measured against the image's typical local
         residual, a MAD-like scale);
      2. globally extreme -- that disagreement is also large relative to the
         image's OWN normal dynamic range (robust p1-p99 spread), not just
         the tiny local noise floor.
    Condition 2 is what keeps genuine sharp object silhouettes safe: a real
    corner/edge pixel can look "locally anomalous" too (a corner's window is
    dominated by the other side of the edge), but its value is still within
    a few multiples of the image's own legitimate range. A decode-singularity
    or sampling-instability spike is many times beyond it. (Verified
    directly: a synthetic sharp 200-vs-50 object corner is left untouched
    while an injected 1177-value spike gets cleaned.)

    size=7 (not 3): real artifacts observed in practice aren't lone single
    pixels but small clusters (checked on an actual sample: UDF ceiling
    pixels form ~500 connected components, mostly 4-5px each) -- a 3x3
    window can be outvoted by a cluster that size, a 7x7 one can't. Even so
    this is a heuristic, not a guarantee: a diffuse multi-pixel noisy patch
    (seen once on a disp map, several nearby pixels all somewhat elevated
    rather than one lone spike against a clean background) can still slip
    through if it doesn't stand out sharply enough from its neighborhood.
    """
    from scipy.ndimage import median_filter
    arr = np.asarray(arr, dtype=np.float64)
    med = median_filter(arr, size=size, mode="nearest")
    resid = np.abs(arr - med)
    local_scale = np.median(resid) + 1e-6
    p1, p99 = np.percentile(arr, [1, 99])
    global_range = (p99 - p1) + 1e-6
    is_outlier = (resid > iso_factor * local_scale) & (resid > range_factor * global_range)
    out = arr.copy()
    out[is_outlier] = med[is_outlier]
    return out


# ── per-seed 3-row plot ───────────────────────────────────────────────────────

def _ax(ax, img, title, cmap=None, vmin=None, vmax=None, colorbar=True, contour_levels=None):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if colorbar and cmap is not None:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if contour_levels is not None and img.ndim == 2:
        ax.contour(img, levels=contour_levels, colors="white", linewidths=0.6, alpha=0.8)
    return im


def save_per_seed_plot(
    out_path: Path,
    stem: str,
    seed: int,
    left_np: np.ndarray,
    right_np: np.ndarray,
    pred_udf_px: np.ndarray,           # obj1 UDF (or composite in 2-ch mode)
    pred_disp_px: Optional[np.ndarray],
    gt_udf_px: Optional[np.ndarray],
    gt_disp_px: Optional[np.ndarray],
    pred_udf2_px: Optional[np.ndarray] = None,  # obj2 UDF (3-ch layered mode)
    gt_udf2_px:   Optional[np.ndarray] = None,
    seg_mask:        Optional[np.ndarray] = None,  # (H, W) foreground mask; nonzero = object
    no_plot: bool = False,                          # compute + return the metrics only (no figure, much faster)
):
    """Plot rows: inputs | UDF obj1 | [UDF obj2 if 3-ch] | [disp].
    Each GT/pred pair: left=GT, right=Pred.
    """
    if no_plot:
        m_u1 = compute_metrics(pred_udf_px, gt_udf_px, seg_mask) if (pred_udf_px is not None and gt_udf_px is not None) else None
        m_u2 = compute_metrics(pred_udf2_px, gt_udf2_px, seg_mask) if (pred_udf2_px is not None and gt_udf2_px is not None) else None
        valid = [m for m in (m_u1, m_u2) if m is not None]
        mu = {k: float(np.mean([m[k] for m in valid])) for k in valid[0]} if valid else None
        md = compute_metrics(pred_disp_px, gt_disp_px, seg_mask) if (pred_disp_px is not None and gt_disp_px is not None) else None
        return mu, md
    has_udf1   = pred_udf_px is not None or gt_udf_px is not None
    has_udf2   = pred_udf2_px is not None
    has_disp   = pred_disp_px is not None or gt_disp_px is not None
    nrows = 1 + int(has_udf1) + int(has_udf2) + int(has_disp)
    fig, axes = plt.subplots(nrows, 2, figsize=(8, nrows * 4), squeeze=False)
    udf1_label = "UDF obj1" if has_udf2 else "UDF"
    fig.suptitle(f"{stem}  seed={seed}", fontsize=11)

    # row 0 — inputs
    _ax(axes[0][0], left_np,  "Left",  colorbar=False)
    _ax(axes[0][1], right_np, "Right", colorbar=False)

    udf_levels = [0, 1]

    def _udf_row(row_idx, pred_u, gt_u, label):
        all_u = [u for u in [pred_u, gt_u] if u is not None]
        vmin = min(u.min() for u in all_u) if all_u else 0
        vmax = max(u.max() for u in all_u) if all_u else 1
        if gt_u is not None:
            _ax(axes[row_idx][0], gt_u, f"GT {label} (px)", cmap="magma",
                vmin=vmin, vmax=vmax, contour_levels=udf_levels)
        else:
            axes[row_idx][0].text(0.5, 0.5, "No GT", ha="center", va="center")
            axes[row_idx][0].axis("off")
            axes[row_idx][0].set_title(f"GT {label}")
        m = compute_metrics(pred_u, gt_u, seg_mask) if (pred_u is not None and gt_u is not None) else None
        tag = f"  EPE={m['epe']:.3f} Bad2={m['bad2']:.1f}%" if m is not None else ""
        if pred_u is not None:
            _ax(axes[row_idx][1], pred_u, f"Pred {label} (px){tag}", cmap="magma",
                vmin=vmin, vmax=vmax, contour_levels=udf_levels)
        else:
            axes[row_idx][1].text(0.5, 0.5, "No Pred", ha="center", va="center")
            axes[row_idx][1].axis("off")
            axes[row_idx][1].set_title(f"Pred {label}")
        return m

    row = 1
    m_udf1 = None
    if has_udf1:
        m_udf1 = _udf_row(row, pred_udf_px, gt_udf_px, udf1_label)
        row += 1

    m_udf2 = None
    if has_udf2:
        m_udf2 = _udf_row(row, pred_udf2_px, gt_udf2_px, "UDF obj2")
        row += 1

    # disparity
    if has_disp:
        show_disp = pred_disp_px if pred_disp_px is not None else np.zeros_like(pred_udf_px)
        all_d = [show_disp] + ([gt_disp_px] if gt_disp_px is not None else [])
        d_vmin = min(d.min() for d in all_d)
        d_vmax = max(d.max() for d in all_d)
        if gt_disp_px is not None:
            _ax(axes[row][0], gt_disp_px, "GT disp (px)", cmap="plasma", vmin=d_vmin, vmax=d_vmax)
        else:
            axes[row][0].text(0.5, 0.5, "No GT", ha="center", va="center")
            axes[row][0].axis("off")
            axes[row][0].set_title("GT disp")
        m_d = compute_metrics(show_disp, gt_disp_px, seg_mask) if gt_disp_px is not None else None
        tag_d = f"  EPE={m_d['epe']:.3f} Bad2={m_d['bad2']:.1f}%" if m_d is not None else ""
        _ax(axes[row][1], show_disp, f"Pred disp (px){tag_d}", cmap="plasma", vmin=d_vmin, vmax=d_vmax)
        row += 1

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    def _avg_metrics(*ms):
        valid = [m for m in ms if m is not None]
        if not valid:
            return None
        return {k: float(np.mean([m[k] for m in valid])) for k in valid[0]}

    metrics_udf  = _avg_metrics(m_udf1, m_udf2)
    metrics_disp = compute_metrics(pred_disp_px, gt_disp_px, seg_mask) \
                   if (pred_disp_px is not None and gt_disp_px is not None) else None
    return metrics_udf, metrics_disp


# ── multi-seed alternating overview ──────────────────────────────────────────

def save_alternating_plot(
    out_path: Path,
    stem: str,
    seeds: List[int],
    left_np: np.ndarray,
    right_np: np.ndarray,
    pred_udfs:  List[np.ndarray],                   # udf obj1, one per seed
    pred_disps: List[Optional[np.ndarray]],
    gt_udf_px:  Optional[np.ndarray],
    gt_disp_px: Optional[np.ndarray],
    pred_udfs2: Optional[List[Optional[np.ndarray]]] = None,  # udf obj2, one per seed
    gt_udf2_px: Optional[np.ndarray] = None,
):
    """
    Overview grid for the alternating experiment.

    Columns: Left | Right | GT | seed-0 | seed-1 | …
    Rows: UDF obj1 | [UDF obj2 if layered] | [disp]
    """
    # Display-only cleanup: despeckle isolated outlier pixels in the
    # *predictions* before they're used for vmin/vmax or plotted. GT is left
    # untouched (it has no such artifacts), and these local copies never
    # propagate back to the caller -- the raw .npy predictions and any
    # metrics computed from them elsewhere are unaffected.
    pred_udfs  = [_despeckle_for_display(u) if u is not None else None for u in pred_udfs]
    pred_disps = [_despeckle_for_display(d) if d is not None else None for d in pred_disps]
    if pred_udfs2 is not None:
        pred_udfs2 = [_despeckle_for_display(u) if u is not None else None for u in pred_udfs2]

    has_udf1 = any(u is not None for u in pred_udfs)
    has_udf2 = pred_udfs2 is not None and any(u is not None for u in pred_udfs2)
    has_disp = any(d is not None for d in pred_disps)
    # row 0 = inputs; then one row per content type
    nrows = 1 + int(has_udf1) + int(has_udf2) + int(has_disp)
    ncols = 3 + len(seeds)   # left, right, GT, then one per seed

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3, nrows * 3.2),
                             squeeze=False)
    fig.suptitle(f"{stem} — alternating experiment ({len(seeds)} seeds)", fontsize=11)

    udf_levels = [0, 1]

    def hide(ax):
        ax.axis("off")

    # row 0: inputs
    _ax(axes[0][0], left_np,  "Left",  colorbar=False)
    _ax(axes[0][1], right_np, "Right", colorbar=False)
    for c in range(2, ncols):
        hide(axes[0][c])

    def _content_row(row_idx, preds, gt, label, cmap, vmin, vmax, contour_levels=None):
        hide(axes[row_idx][0]); hide(axes[row_idx][1])
        if gt is not None:
            _ax(axes[row_idx][2], gt, f"GT {label}", cmap=cmap,
                vmin=vmin, vmax=vmax, contour_levels=contour_levels)
        else:
            hide(axes[row_idx][2]); axes[row_idx][2].set_title(f"GT {label}")
        for i, (s, p) in enumerate(zip(seeds, preds)):
            if p is not None:
                _ax(axes[row_idx][3 + i], p, f"seed {s}", cmap=cmap,
                    vmin=vmin, vmax=vmax, contour_levels=contour_levels)
            else:
                hide(axes[row_idx][3 + i])

    row = 1
    udf1_label = "UDF obj1" if has_udf2 else "UDF"
    if has_udf1:
        valid = [u for u in pred_udfs if u is not None]
        all_u = valid + ([gt_udf_px] if gt_udf_px is not None else [])
        vmin = min(u.min() for u in all_u) if all_u else 0
        vmax = max(u.max() for u in all_u) if all_u else 1
        _content_row(row, pred_udfs, gt_udf_px, udf1_label, "magma", vmin, vmax, udf_levels)
        row += 1

    if has_udf2:
        valid2 = [u for u in pred_udfs2 if u is not None]
        all_u2 = valid2 + ([gt_udf2_px] if gt_udf2_px is not None else [])
        vmin2 = min(u.min() for u in all_u2) if all_u2 else 0
        vmax2 = max(u.max() for u in all_u2) if all_u2 else 1
        _content_row(row, pred_udfs2, gt_udf2_px, "UDF obj2", "magma", vmin2, vmax2, udf_levels)
        row += 1

    if has_disp:
        all_disp = [d for d in pred_disps if d is not None]
        if gt_disp_px is not None:
            all_disp.append(gt_disp_px)
        d_vmin = min(d.min() for d in all_disp) if all_disp else 0
        d_vmax = max(d.max() for d in all_disp) if all_disp else 1
        _content_row(row, pred_disps, gt_disp_px, "disp", "plasma", d_vmin, d_vmax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("here1")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=False,
                   help="Model checkpoint (.pth). Optional only with --summary-only.")
    p.add_argument("--config",     help="Config JSON (read from checkpoint if omitted)")
    p.add_argument("--summary-csv", default=None,
                   help="Shared CSV across runs; this model's aggregate row is appended "
                        "and the full cross-model comparison table is printed.")
    p.add_argument("--model-name", default=None,
                   help="Row label in the summary table (default: out-dir basename).")
    p.add_argument("--summary-only", action="store_true",
                   help="Skip inference; just print the comparison table from --summary-csv.")
    p.add_argument("--use-ema",    action="store_true", default=True)
    p.add_argument("--steps",      type=int, default=50)
    p.add_argument("--seeds",      type=int, nargs="+", default=[0],
                   help="One or more seeds. Multiple seeds run the alternating experiment.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip the per-seed comparison PNGs and the alternating overview (metrics, .npy "
                        "predictions and csvs are unchanged). Use for many-seed runs.")
    p.add_argument("--min-index", type=int, default=None,
                   help="Only process samples whose stem index >= min-index (last int in name).")
    p.add_argument("--max-index", type=int, default=None,
                   help="Only process samples whose stem index <= max-index (last int in name).")
    p.add_argument("--tile-overlap", type=float, default=0.0,
                   help="Fractional overlap between adjacent inference tiles, only relevant "
                        "when an image is larger than the model's patch size (e.g. SceneFlow "
                        "at native resolution). 0.0 (default) = old non-overlapping grid, "
                        "identical output to before this flag existed. E.g. 0.5 = 50%% overlap "
                        "with cross-fade blending between tiles, removing hard tile-grid seams.")
    p.add_argument("--tile-pad-mode", choices=["constant", "reflect"], default="constant",
                   help="Padding mode when image dims aren't a multiple of patch size. "
                        "constant (default, old behavior) zero-pads, which creates "
                        "out-of-distribution all-black edge tiles the model never saw in "
                        "training (crops are always sampled fully inside the real image). "
                        "reflect mirrors real image content into the pad region instead.")
    p.add_argument("--bimodal-tsne", action="store_true",
                   help="Run t-SNE + silhouette analysis for bimodal samples "
                        "(triggered when *_field_A.npy / *_field_B.npy GT files are found).")
    p.add_argument("--spread-map", action="store_true",
                   help="Posterior-spread experiment (needs multiple --seeds): per-pixel "
                        "std of predictions across seeds, saved as *_disp_std.npy / "
                        "*_udf_std.npy + a heatmap figure with GT boundary contours. "
                        "If GT disp is given, spread is split into GT boundary band vs "
                        "interior (same band definition as report_metrics.py boundary) "
                        "and written to spread_metrics.csv. Also saves the cross-seed "
                        "MEAN disparity (*_mean_disp.npy) for the self-averaging "
                        "sharpness comparison.")
    p.add_argument("--spread-fgbg-t", type=float, default=1.05,
                   help="Depth Pro fgbg ratio threshold defining a GT boundary for the "
                        "spread split (default 1.05, matching report_metrics.py boundary).")
    p.add_argument("--spread-win", type=int, default=2,
                   help="Half-width (px) of the boundary band for the spread split "
                        "(default 2, matching report_metrics.py boundary).")
    p.add_argument("--scc-coarse-strip-px", type=int, default=512,
                   help="Width (px) of the wide L/R crop fed to LeanCorrelationConditionerC2F's "
                        "coarse stage, when the checkpoint has use_scc_c2f=True. Should match "
                        "(or exceed) whatever coarse_strip_px the training dataset config used -- "
                        "ignored entirely for checkpoints without use_scc_c2f.")

    # single pair
    p.add_argument("--left");  p.add_argument("--right")
    p.add_argument("--gt-disp",  help="GT disparity .npy")
    p.add_argument("--gt-field", help="GT UDF .npy")
    p.add_argument("--seg",      help="Segmentation mask .npy for single pair (nonzero = object)")

    # directories
    p.add_argument("--left-dir");  p.add_argument("--right-dir")
    p.add_argument("--gt-disp-dir");  p.add_argument("--gt-field-dir")
    p.add_argument("--gt-udf-obj1-dir", help="Dir of GT per-object UDF obj1 .npy (layered model)")
    p.add_argument("--gt-udf-obj2-dir", help="Dir of GT per-object UDF obj2 .npy (layered model)")
    p.add_argument("--seg-dir", help="Dir of segmentation masks .npy (nonzero = object); "
                                     "if provided, metrics are computed on foreground only")
    p.add_argument("--glob", default="*.png")

    p.add_argument("--out-dir", default="infer_out")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()

    # summary-only: just reprint the cross-model table and exit (no inference).
    if args.summary_only:
        if not args.summary_csv:
            p.error("--summary-only requires --summary-csv.")
        print_summary_table(args.summary_csv)
        return
    if not args.checkpoint:
        p.error("--checkpoint is required (unless --summary-only).")

    device    = torch.device(args.device)
    out_dir   = Path(args.out_dir)
    seeds     = args.seeds

    # ── load checkpoint ───────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt.get("config") or (
        json.load(open(args.config)) if args.config
        else (_ for _ in ()).throw(ValueError("Pass --config (no config in checkpoint)."))
    )

    model_config   = config["model"]
    patch_size     = model_config["input_size"][0]
    input_channels = model_config["input_channels"]
    sigma_min      = model_config["sigma_min"]
    sigma_max      = model_config["sigma_max"]
    udf_scale           = config["dataset"].get("config", {}).get("u_scale", 64.0)
    disp_norm_val       = model_config.get("disp_norm", 64.0)
    # Encoding of the UDF channel is inferred from the checkpoint config:
    #   exponential  (1 - exp(-k * udf/u_scale)) * w   <- models trained with `udf_k`
    #   linear        udf / u_scale                    <- older "generic"/"refine" models (no udf_k)
    # Decoding with the wrong inverse keeps the field shape but wrecks its scale,
    # which is what made the generic models' UDF look catastrophic (EPE ~9 vs ~0.35).
    udf_k_raw           = model_config.get("udf_k", None)
    udf_linear          = udf_k_raw is None
    udf_k               = udf_k_raw if udf_k_raw is not None else 3.0
    udf_channel_weight  = model_config.get("udf_channel_weight", None) or 1.0

    inner_model = K.config.make_model(config).to(device).eval()
    weights = ckpt["model_ema"] if (args.use_ema and "model_ema" in ckpt) else ckpt["model"]
    # Older checkpoints (pre-RAFT-conditioning) don't have the `raft_disp_norm`
    # buffer, which is a fixed constant (not learned) added unconditionally to
    # the model later. Tolerate only that specific missing key.
    missing, unexpected = inner_model.load_state_dict(weights, strict=False)
    allowed_missing = {"raft_disp_norm"}
    bad_missing = set(missing) - allowed_missing
    if bad_missing or unexpected:
        raise RuntimeError(
            f"Error(s) in loading state_dict for {type(inner_model).__name__}:\n"
            f"\tMissing key(s): {sorted(bad_missing)}\n"
            f"\tUnexpected key(s): {sorted(unexpected)}"
        )
    model_ema = K.config.make_denoiser_wrapper(config)(inner_model)

    has_scc        = getattr(inner_model, "use_scc",        False)  # token-injection LCC
    has_scc_native = getattr(inner_model, "use_scc_native", False)  # native-res channel-concat LCC
    # disp_left: LCC matches in the left-view frame (fL fixed, fR shifted by -d) instead
    # of cyclopean. Only meaningful when has_scc — read off the built module, since it's
    # a forward-time behavior flag with no dedicated params (checkpoint-compatible either way).
    disp_left = has_scc and getattr(inner_model.scc, "disp_left", False)

    step = ckpt.get("step", 0)
    lcc_mode = ("token-inject" if has_scc else "") + ("native" if has_scc_native else "") or "none"
    print(f"  step={step}  channels={input_channels}  patch={patch_size}  "
          f"udf_scale={udf_scale}  udf_k={udf_k}  udf_decode={'linear' if udf_linear else 'exp'}  "
          f"disp_norm={disp_norm_val}  "
          f"udf_channel_weight={udf_channel_weight}  lcc={lcc_mode}  "
          f"lcc_frame={'left-view' if disp_left else 'cyclopean'}  seeds={seeds}")

    # ── collect pairs ─────────────────────────────────────────────────────────
    pairs = []

    if args.left and args.right:
        pairs.append((Path(args.left), Path(args.right), Path(args.left).stem,
                      Path(args.gt_disp)  if args.gt_disp  else None,
                      Path(args.gt_field) if args.gt_field else None,
                      None, None,
                      Path(args.seg) if args.seg else None,
                      None, None))

    gt_udf_obj1_dir  = Path(args.gt_udf_obj1_dir)  if args.gt_udf_obj1_dir  else None
    gt_udf_obj2_dir  = Path(args.gt_udf_obj2_dir)  if args.gt_udf_obj2_dir  else None
    seg_dir          = Path(args.seg_dir)           if args.seg_dir          else None

    if args.left_dir and args.right_dir:
        left_dir  = Path(args.left_dir)
        right_dir = Path(args.right_dir)
        gt_disp_dir  = Path(args.gt_disp_dir)  if args.gt_disp_dir  else None
        gt_field_dir = Path(args.gt_field_dir) if args.gt_field_dir else None

        def find_gt(d, stems):
            if d is None: return None
            for s in stems:
                c = d / s
                if c.exists(): return c
            return None

        for lp in sorted(left_dir.glob(args.glob)):
            base = lp.stem[:-5] if lp.stem.endswith("_left") else lp.stem
            # index range filter
            if args.min_index is not None or args.max_index is not None:
                idx = _stem_index(base)
                if idx is None:
                    continue
                if args.min_index is not None and idx < args.min_index:
                    continue
                if args.max_index is not None and idx > args.max_index:
                    continue
            rp = right_dir / f"{base}_right{lp.suffix}"
            if not rp.exists(): rp = right_dir / lp.name
            if not rp.exists(): print(f"  [warn] no right match for {lp.name}"); continue
            pairs.append((lp, rp, base,
                          find_gt(gt_disp_dir,      [f"{base}_disp.npy",  f"{base}.npy"]),
                          find_gt(gt_field_dir,      [f"{base}_field.npy", f"{base}.npy"]),
                          find_gt(gt_udf_obj1_dir,   [f"{base}_udf.npy"]),
                          find_gt(gt_udf_obj2_dir,   [f"{base}_udf.npy"]),
                          find_gt(seg_dir,           [f"{base}_seg.npy", f"{base}.npy"]),
                          find_gt(gt_field_dir,      [f"{base}_field_A.npy"]),
                          find_gt(gt_field_dir,      [f"{base}_field_B.npy"])))

    if not pairs:
        raise ValueError("No pairs found.")

    print(f"Running on {len(pairs)} pair(s) × {len(seeds)} seed(s) → {out_dir}/")

    csv_rows = []
    ensemble_rows = []   # one median-over-seeds row per sample (multi-seed runs only)
    scc_epe_list = []    # SCC cost-volume disparity EPE per sample (SCC models only)
    tsne_rows = []       # bimodal t-SNE silhouette scores, one per sample
    spread_rows = []     # posterior-spread boundary/interior stats, one per sample

    for lp, rp, stem, gt_disp_path, gt_field_path, \
            gt_udf1_path, gt_udf2_path, seg_path, gt_field_a_path, gt_field_b_path \
            in tqdm(pairs, desc="pairs"):
        left  = load_rgb(lp, device)
        right = load_rgb(rp, device)
        left_np  = tensor_to_uint8(left)
        right_np = tensor_to_uint8(right)
        cond6    = make_cond(left, right)

        gt_udf_px  = load_npy(gt_field_path) if gt_field_path else None
        gt_disp_px = load_npy(gt_disp_path)  if gt_disp_path  else None
        gt_udf1_px = load_npy(gt_udf1_path)  if gt_udf1_path  else None
        gt_udf2_px = load_npy(gt_udf2_path)  if gt_udf2_path  else None
        seg_mask   = load_npy(seg_path)       if seg_path      else None
        # bimodal GT: field_A and field_B are the two ambiguous boundary interpretations
        gt_udf_A_px = load_npy(gt_field_a_path) if gt_field_a_path else None
        gt_udf_B_px = load_npy(gt_field_b_path) if gt_field_b_path else None
        is_bimodal  = gt_udf_A_px is not None and gt_udf_B_px is not None
        if seg_mask is not None:
            print(f"  [{stem}] seg mask loaded — evaluating on {int((seg_mask > 0).sum())} foreground pixels")
        if is_bimodal:
            print(f"  [{stem}] bimodal GT (A+B) — UDF metrics use oracle min(EPE_A, EPE_B)")

        pred_udfs  = []
        pred_udfs2 = []
        pred_disps = []

        _, orig_H, orig_W = cond6.shape
        scc_base = None   # (H,W) px, seed-independent; d_hat added back when residual

        use_scc_c2f = getattr(inner_model, "use_scc_c2f", False)
        coarse_strip_px = args.scc_coarse_strip_px

        for seed in tqdm(seeds, desc="seeds", leave=False):
            pred, _, tiles_cond, tile_pos, (pH, pW), wide_tiles = run_inference(
                model_ema, cond6,
                patch_size=patch_size,
                input_channels=input_channels,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                n_steps=args.steps,
                device=device,
                seed=seed,
                tile_overlap=args.tile_overlap,
                tile_pad_mode=args.tile_pad_mode,
                det_sigma=(config.get("deterministic") or {}).get("sigma"),
                use_scc_c2f=use_scc_c2f,
                coarse_strip_px=coarse_strip_px,
            )

            seed_stem = f"{stem}_seed{seed}"
            seed_dir  = out_dir / stem
            seed_dir.mkdir(parents=True, exist_ok=True)

            # LCC disparity — depends only on L/R images, so compute once per image (not per seed).
            # Saved as *_scc_disp.npy; EPE printed if GT disparity is available.
            if (has_scc or has_scc_native) and scc_base is None:
                if has_scc_native:
                    scc_disp, scc_valid = run_scc_native(inner_model, tiles_cond, tile_pos,
                                                          pH, pW, patch_size, orig_H, orig_W)
                else:
                    scc_disp, scc_valid = run_scc(inner_model, tiles_cond, tile_pos,
                                                   pH, pW, patch_size, orig_H, orig_W,
                                                   wide_tiles=wide_tiles)
                scc_base = scc_disp * (scc_valid > 0)
                np.save(seed_dir / f"{stem}_scc_disp.npy", scc_disp.astype(np.float32))
                if gt_disp_px is not None:
                    vmask = (scc_valid > 0)
                    if seg_mask is not None:
                        vmask = vmask & (seg_mask > 0)
                    m_scc = compute_metrics(scc_disp, gt_disp_px, vmask.astype(np.float32))
                    scc_epe_list.append(m_scc["epe"])
                    print(f'    [SCC] {stem} cost-volume disp EPE={m_scc["epe"]:.4f} (valid region)')

            udf1_px, udf2_px, disp_px = decode_pred(pred, udf_scale, udf_k, disp_norm_val,
                                                     udf_channel_weight, udf_linear=udf_linear)
            pred_udfs.append(udf1_px)
            pred_udfs2.append(udf2_px)
            pred_disps.append(disp_px)

            if udf1_px is not None:
                np.save(seed_dir / f"{seed_stem}_udf_obj1.npy", udf1_px.astype(np.float32))
            if udf2_px is not None:
                np.save(seed_dir / f"{seed_stem}_udf_obj2.npy", udf2_px.astype(np.float32))
            if disp_px is not None:
                np.save(seed_dir / f"{seed_stem}_disp.npy", disp_px.astype(np.float32))

            # per-seed plot
            # use gt_udf1/2 if available (layered mode), else composite field GT;
            # for bimodal samples show GT_A as reference (oracle metric computed separately)
            plot_gt_udf  = (gt_udf1_px if gt_udf1_px is not None
                            else (gt_udf_A_px if is_bimodal else gt_udf_px))
            plot_gt_udf2 = gt_udf2_px  # None in 2-ch mode
            metrics_udf, metrics_disp = save_per_seed_plot(
                out_path  = seed_dir / f"{seed_stem}_comparison.png",
                stem=stem, seed=seed,
                left_np=left_np, right_np=right_np,
                pred_udf_px=udf1_px,
                pred_disp_px=disp_px,
                gt_udf_px=plot_gt_udf,
                gt_disp_px=gt_disp_px,
                pred_udf2_px=udf2_px,
                gt_udf2_px=plot_gt_udf2,
                seg_mask=seg_mask,
                no_plot=args.no_plots,
            )
            row_data = {"stem": stem, "seed": seed}
            if metrics_udf  is not None:
                row_data.update({f"udf_{k}":  v for k, v in metrics_udf.items()})
            if metrics_disp is not None:
                row_data.update({f"disp_{k}": v for k, v in metrics_disp.items()})
            # bimodal oracle: replace UDF metrics with min(EPE vs GT_A, EPE vs GT_B)
            if is_bimodal and udf1_px is not None:
                m_A = compute_metrics(udf1_px, gt_udf_A_px, seg_mask)
                m_B = compute_metrics(udf1_px, gt_udf_B_px, seg_mask)
                oracle = m_A if m_A.get("epe", float("inf")) <= m_B.get("epe", float("inf")) else m_B
                row_data.update({f"udf_{k}": v for k, v in oracle.items()})
            csv_rows.append(row_data)

        # multi-seed alternating overview
        if len(seeds) > 1 and not args.no_plots:
            alt_gt_udf  = gt_udf1_px if gt_udf1_px is not None else gt_udf_px
            alt_gt_udf2 = gt_udf2_px
            save_alternating_plot(
                out_path  = out_dir / f"{stem}_alternating.png",
                stem=stem,
                seeds=seeds,
                left_np=left_np, right_np=right_np,
                pred_udfs=pred_udfs,
                pred_udfs2=pred_udfs2,
                pred_disps=pred_disps,
                gt_udf_px=alt_gt_udf,
                gt_udf2_px=alt_gt_udf2,
                gt_disp_px=gt_disp_px,
            )

        # seed-ensembled prediction (median over seeds) — a robust single estimate.
        # Diffusion samples vary per seed; the per-sample median is consistently
        # lower-error than any single seed at no extra inference cost.
        if len(seeds) > 1:
            def _median_stack(lst):
                arrs = [a for a in lst if a is not None]
                return np.median(np.stack(arrs, axis=0), axis=0) if arrs else None

            disp_ens = _median_stack(pred_disps)
            udf1_ens = _median_stack(pred_udfs)
            udf2_ens = _median_stack(pred_udfs2)
            ens_gt_udf = gt_udf1_px if gt_udf1_px is not None else gt_udf_px

            m_disp_e = (compute_metrics(disp_ens, gt_disp_px, seg_mask)
                        if (disp_ens is not None and gt_disp_px is not None) else None)
            mu1 = (compute_metrics(udf1_ens, ens_gt_udf, seg_mask)
                   if (udf1_ens is not None and ens_gt_udf is not None) else None)
            mu2 = (compute_metrics(udf2_ens, gt_udf2_px, seg_mask)
                   if (udf2_ens is not None and gt_udf2_px is not None) else None)
            valid_u = [m for m in (mu1, mu2) if m is not None]
            m_udf_e = ({k: float(np.mean([m[k] for m in valid_u])) for k in valid_u[0]}
                       if valid_u else None)

            erow = {"stem": stem, "seed": "median"}
            if m_udf_e:  erow.update({f"udf_{k}":  v for k, v in m_udf_e.items()})
            if m_disp_e: erow.update({f"disp_{k}": v for k, v in m_disp_e.items()})
            ensemble_rows.append(erow)

            seed_dir = out_dir / stem
            if disp_ens is not None:
                np.save(seed_dir / f"{stem}_median_disp.npy", disp_ens.astype(np.float32))
            if udf1_ens is not None:
                np.save(seed_dir / f"{stem}_median_udf_obj1.npy", udf1_ens.astype(np.float32))
            if udf2_ens is not None:
                np.save(seed_dir / f"{stem}_median_udf_obj2.npy", udf2_ens.astype(np.float32))

        # posterior-spread experiment: per-pixel std across seeds
        if args.spread_map and len(seeds) > 1:
            def _std_stack(lst):
                arrs = [a for a in lst if a is not None]
                return np.std(np.stack(arrs, axis=0), axis=0) if len(arrs) > 1 else None

            def _mean_stack(lst):
                arrs = [a for a in lst if a is not None]
                return np.mean(np.stack(arrs, axis=0), axis=0) if arrs else None

            disp_std = _std_stack(pred_disps)
            udf_std  = _std_stack(pred_udfs)
            disp_mean = _mean_stack(pred_disps)

            seed_dir = out_dir / stem
            seed_dir.mkdir(parents=True, exist_ok=True)
            if disp_std is not None:
                np.save(seed_dir / f"{stem}_disp_std.npy", disp_std.astype(np.float32))
            if udf_std is not None:
                np.save(seed_dir / f"{stem}_udf_std.npy", udf_std.astype(np.float32))
            if disp_mean is not None:
                np.save(seed_dir / f"{stem}_mean_disp.npy", disp_mean.astype(np.float32))

            stats = {}
            if disp_std is not None and gt_disp_px is not None:
                stats = compute_spread_stats(disp_std, gt_disp_px,
                                             args.spread_fgbg_t, args.spread_win)
                if stats:
                    udf_stats = (compute_spread_stats(udf_std, gt_disp_px,
                                                      args.spread_fgbg_t, args.spread_win)
                                 if udf_std is not None else {})
                    spread_rows.append({
                        "stem": stem, "n_seeds": len(seeds), **stats,
                        **{f"udf_{k}": v for k, v in udf_stats.items()
                           if k in ("bnd_std", "int_std", "std_ratio", "concentration")},
                    })
                    print(f"  [spread] {stem}: bnd_std={stats['bnd_std']:.3f} "
                          f"int_std={stats['int_std']:.3f} "
                          f"ratio={stats['std_ratio']:.1f}x "
                          f"({stats['var_share_pct']:.0f}% of variance in "
                          f"{stats['band_frac_pct']:.1f}% of pixels, "
                          f"concentration={stats['concentration']:.1f}x)")

            if disp_std is not None or udf_std is not None:
                save_spread_plot(
                    out_path=out_dir / f"{stem}_spread.png",
                    stem=stem, n_seeds=len(seeds),
                    left_np=left_np,
                    disp_std=disp_std, udf_std=udf_std,
                    gt_disp_px=gt_disp_px,
                    fgbg_t=args.spread_fgbg_t,
                    stats=stats,
                )

        # t-SNE bimodality plot (bimodal samples only, requires --bimodal-tsne)
        if is_bimodal and args.bimodal_tsne:
            valid_pairs = [(u, s) for u, s in zip(pred_udfs, seeds) if u is not None]
            if valid_pairs:
                udf_list, sid_list = zip(*valid_pairs)
                sil = _run_tsne_plot(
                    stem=stem,
                    pred_udfs_list=list(udf_list),
                    seed_ids=list(sid_list),
                    gt_udf_A=gt_udf_A_px,
                    gt_udf_B=gt_udf_B_px,
                    out_path=out_dir / stem / f"{stem}_tsne_udf.png",
                )
                tsne_rows.append({"stem": stem, "silhouette": sil,
                                  "n_seeds": len(udf_list)})

    # ── summary ───────────────────────────────────────────────────────────────
    if csv_rows:
        metric_keys = []
        for r in csv_rows:
            for k in r:
                if k not in ("stem", "seed") and k not in metric_keys:
                    metric_keys.append(k)
        fieldnames = ["stem", "seed"] + metric_keys
        csv_path = out_dir / "metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in csv_rows:
                writer.writerow({k: r.get(k, float("nan")) for k in fieldnames})
        print(f"\nMetrics saved → {csv_path}")

        for prefix, label in [("disp", "Disp"), ("udf", "UDF")]:
            epe_vals  = [r[f"{prefix}_epe"]  for r in csv_rows if f"{prefix}_epe"  in r]
            if not epe_vals:
                continue
            print(f"\n=== {label} ({len(epe_vals)} evals) ===")
            print(f"  EPE   Mean={np.mean(epe_vals):.4f}  Std={np.std(epe_vals):.4f}  "
                  f"Min={np.min(epe_vals):.4f}  Max={np.max(epe_vals):.4f}")
            for t in THRESHOLDS:
                bv = [r[f"{prefix}_{_badkey(t)}"] for r in csv_rows
                      if f"{prefix}_{_badkey(t)}" in r]
                if bv:
                    print(f"  Bad{t:<4g} Mean={np.mean(bv):6.2f}%  Std={np.std(bv):5.2f}  "
                          f"Min={np.min(bv):6.2f}  Max={np.max(bv):6.2f}")

    # ── seed-ensembled (median-over-seeds) summary ─────────────────────────────
    if ensemble_rows:
        ekeys = []
        for r in ensemble_rows:
            for k in r:
                if k not in ("stem", "seed") and k not in ekeys:
                    ekeys.append(k)
        efields = ["stem", "seed"] + ekeys
        epath = out_dir / "metrics_ensemble.csv"
        with open(epath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=efields, extrasaction="ignore")
            writer.writeheader()
            for r in ensemble_rows:
                writer.writerow({k: r.get(k, float("nan")) for k in efields})
        print(f"\nSeed-ensembled (median-of-{len(seeds)}) metrics saved → {epath}")

        for prefix, label in [("disp", "Disp"), ("udf", "UDF")]:
            epe_vals  = [r[f"{prefix}_epe"]  for r in ensemble_rows if f"{prefix}_epe"  in r]
            if not epe_vals:
                continue
            print(f"\n=== {label} ENSEMBLE median-of-{len(seeds)} ({len(epe_vals)} samples) ===")
            print(f"  EPE   Mean={np.mean(epe_vals):.4f}  Std={np.std(epe_vals):.4f}  "
                  f"Min={np.min(epe_vals):.4f}  Max={np.max(epe_vals):.4f}")
            for t in THRESHOLDS:
                bv = [r[f"{prefix}_{_badkey(t)}"] for r in ensemble_rows
                      if f"{prefix}_{_badkey(t)}" in r]
                if bv:
                    print(f"  Bad{t:<4g} Mean={np.mean(bv):6.2f}%")

    # ── SCC cost-volume disparity summary (SCC models only) ────────────────────
    if scc_epe_list:
        print(f"\n=== SCC cost-volume disp ({len(scc_epe_list)} samples, valid region) ===")
        print(f"  EPE   Mean={np.mean(scc_epe_list):.4f}  Std={np.std(scc_epe_list):.4f}  "
              f"Min={np.min(scc_epe_list):.4f}  Max={np.max(scc_epe_list):.4f}")

    # ── aggregate CSV: mean / median / best1-of-N ─────────────────────────────
    if csv_rows:
        _write_agg_csv(csv_rows, out_dir / "metrics_agg.csv")

    # ── posterior-spread summary ──────────────────────────────────────────────
    if spread_rows:
        sfields = []
        for r in spread_rows:
            for k in r:
                if k not in sfields:
                    sfields.append(k)
        spath = out_dir / "spread_metrics.csv"
        with open(spath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sfields, extrasaction="ignore")
            w.writeheader()
            for r in spread_rows:
                w.writerow({k: r.get(k, float("nan")) for k in sfields})
        print(f"\nPosterior-spread metrics ({len(spread_rows)} samples) → {spath}")

        def _sm(key):
            vals = [r[key] for r in spread_rows if key in r and r[key] == r[key]]
            return float(np.mean(vals)) if vals else float("nan")

        print(f"=== Posterior spread (std across {len(seeds)} seeds, "
              f"boundary band = GT fgbg edges ±{args.spread_win}px) ===")
        print(f"  disp: boundary std={_sm('bnd_std'):.4f}  interior std={_sm('int_std'):.4f}  "
              f"ratio={_sm('std_ratio'):.2f}x")
        print(f"  variance concentration: {_sm('var_share_pct'):.1f}% of total variance "
              f"in {_sm('band_frac_pct'):.1f}% of pixels "
              f"(concentration={_sm('concentration'):.2f}x; 1.0 = uniform)")
        if any("udf_bnd_std" in r for r in spread_rows):
            print(f"  udf:  boundary std={_sm('udf_bnd_std'):.4f}  "
                  f"interior std={_sm('udf_int_std'):.4f}  "
                  f"ratio={_sm('udf_std_ratio'):.2f}x")

    # ── bimodal t-SNE silhouette CSV ──────────────────────────────────────────
    if tsne_rows:
        tsne_csv = out_dir / "tsne_silhouette.csv"
        with open(tsne_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["stem", "silhouette", "n_seeds"])
            w.writeheader()
            for r in tsne_rows:
                w.writerow(r)
        sil_vals = [r["silhouette"] for r in tsne_rows
                    if r["silhouette"] == r["silhouette"]]  # drop NaN
        print(f"\nBimodal t-SNE silhouette scores ({len(tsne_rows)} samples) → {tsne_csv}")
        if sil_vals:
            print(f"  Mean={np.mean(sil_vals):.3f}  Std={np.std(sil_vals):.3f}  "
                  f"Min={np.min(sil_vals):.3f}  Max={np.max(sil_vals):.3f}")

    # ── append this model's row to the shared summary + print comparison table ──
    if args.summary_csv and csv_rows:
        model_name = args.model_name or out_dir.name
        append_summary_row(args.summary_csv, model_name, csv_rows,
                           ensemble_rows, len(seeds))
        print(f"\nSummary row written for '{model_name}' → {args.summary_csv}")
        print_summary_table(args.summary_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
