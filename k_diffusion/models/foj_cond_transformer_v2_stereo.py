# foj_cond_transformer_v2_stereo_dual.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_transformer_v2 import (
    ImageTransformerDenoiserModelV2,
    TokenMerge,
    TokenSplitWithoutSkip,
    downscale_pos,
)
from .axial_rope import make_axial_pos


class LeanCorrelationConditioner(nn.Module):
    """Correlation cost volume built from the L/R images.

    Default: symmetric (cyclopean) matching. The prediction grid is cyclopean: a
    cyclopean column x has its left match at x + d/2 and its right match at x - d/2
    (disparity d >= 0). We build a symmetric correlation volume over d in [0, dmax],
    read a soft-argmax disparity, warp both views onto the cyclopean grid, and fuse
    into a per-token joint feature.

    disp_left=True: left-view matching instead. fL stays on its native grid and only
    fR is sampled at x - d, so d_hat is a disparity directly on the left-image pixel
    grid — matching the GT disparity convention and the main model's own output
    channel (both are asserted to align with the left image's pixel grid). This
    removes the frame mismatch the cyclopean mode has against GT-based supervision.

    Everything runs at the token grid (image_size / px_per_token), so disparity is
    in token units internally; multiply by px_per_token to get pixels.

    Returns:
      joint_tokens: (B, Ht, Wt, width0)  additive conditioning in token layout
      disp_px:      (B, 1, Ht, Wt)       soft-argmax disparity in pixels
      conf:         (B, 1, Ht, Wt)       match confidence in [0, 1]
      valid:        (B, 1, Ht, Wt)       border-validity mask in {0, 1}
      dxy:          (B, 2, Ht, Wt)       disparity gradient (px/px), or None if use_plane=False
    """

    def __init__(self, feat_ch=64, dmax=16, width0=128, px_per_token=4, disp_step=1.0,
                 n_extra_cand=0, use_plane=False, use_nowrap=False, disp_left=False,
                 use_own_cv=True):
        super().__init__()
        self.dmax = int(dmax)
        self.px_per_token = float(px_per_token)
        self.disp_step = float(disp_step)
        self.use_plane = bool(use_plane)
        self.use_nowrap = bool(use_nowrap)
        self.disp_left = bool(disp_left)
        # use_own_cv=False: DINO-cost-volume-only mode -- skip LCC's own CNN
        # correlation loop in forward() entirely; agg/slant then read only the
        # externally supplied extra_cv (e.g. DINO). The encoder below still runs
        # unconditionally, since fL/fR are also needed for the appearance
        # warp+fuse step at the end of forward(), independent of where the
        # disparity estimate itself came from.
        self.use_own_cv = bool(use_own_cv)
        if not self.use_own_cv and n_extra_cand == 0:
            raise ValueError(
                "LeanCorrelationConditioner: use_own_cv=False (DINO-cost-volume-only) "
                "requires n_extra_cand > 0 -- with no own CNN correlation and no extra "
                "(DINO) cost volume, there would be no cost-volume input at all.")

        # Disparity candidates in TOKEN units: 0, step, 2*step, ..., dmax.
        cands = torch.arange(0.0, self.dmax + 1e-6, self.disp_step, dtype=torch.float32)
        self.n_cand = int(cands.numel())

        # Shared L/R encoder: total stride = px_per_token.
        # px_per_token=4 → stride-2 × stride-2; px_per_token=2 → stride-2 × stride-1.
        # px_per_token>4 (must be a power of 2, e.g. 8, 16 for bigger native tiles):
        # extra feat_ch->feat_ch stride-2 layers make up the remaining factor beyond
        # the base pair's max total stride of 4 -- bit-identical to the old fixed
        # 3-layer encoder for px_per_token in {2,4} (the only values any existing
        # checkpoint was trained with), only different for px_per_token>4, which
        # previously silently clamped to stride-4 and produced a wrong-resolution
        # cost volume (mismatched against anything else deriving its grid from the
        # real px_per_token, e.g. DINOCostVolume's target_H/target_W).
        s2 = 2 if int(px_per_token) >= 4 else 1
        base_stride = 2 * s2
        extra_down = 0
        if int(px_per_token) > 4:
            extra_needed = px_per_token / base_stride
            extra_down = round(math.log2(extra_needed))
            if base_stride * (2 ** extra_down) != int(px_per_token):
                raise ValueError(f"px_per_token must be a power of 2 (>=2), got {px_per_token}")
        enc_layers = [
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, feat_ch, 3, stride=s2, padding=1), nn.ReLU(inplace=True),
        ]
        for _ in range(extra_down):
            enc_layers += [nn.Conv2d(feat_ch, feat_ch, 3, stride=2, padding=1), nn.ReLU(inplace=True)]
        enc_layers += [nn.Conv2d(feat_ch, feat_ch, 3, padding=1)]
        self.encoder = nn.Sequential(*enc_layers)
        # Cost-volume aggregation: accepts CNN + optional extra (e.g. DINO) candidates as
        # input channels, but always outputs n_cand channels for soft-argmax over CNN disparities.
        # use_own_cv=False drops LCC's own CNN correlation channels from the input entirely
        # (agg_in_ch shrinks to just n_extra_cand), but the OUTPUT stays n_cand (LCC's own
        # token-grid scale) either way, so d_hat/disp_px/valid-mask semantics downstream are
        # unaffected by this flag.
        agg_in_ch = (self.n_cand if self.use_own_cv else 0) + n_extra_cand
        self.agg = nn.Sequential(
            nn.Conv2d(agg_in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, self.n_cand, 3, padding=1),
        )
        # Plane mode (A1+B1): slant head regresses (dx, dy) disparity gradients (px/px)
        # from the fused cost volume; fuse additionally sees the full probability
        # distribution P (n_cand) and the slant (2) instead of only the collapsed d_hat.
        fuse_in = 3 * feat_ch + 2
        if self.use_plane:
            self.slant = nn.Sequential(
                nn.Conv2d(agg_in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 2, 3, padding=1),
            )
            nn.init.zeros_(self.slant[-1].weight)
            nn.init.zeros_(self.slant[-1].bias)
            fuse_in += self.n_cand + 2
        # Fuse [fL_cyc, fR_cyc, fL_cyc - fR_cyc, d_hat, conf (, P, dx, dy)] -> width0.
        # Skipped (not built) when use_nowrap=True: that mode uses direct_proj instead.
        if not self.use_nowrap:
            self.fuse = nn.Sequential(
                nn.Conv2d(fuse_in, width0, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(width0, width0, 3, padding=1),
            )
        # scc_nowrap: skip the warp entirely, project the fused distribution P
        # (agg's output is always n_cand channels, regardless of n_extra_cand)
        # directly to width0 with a single conv.
        if self.use_nowrap:
            direct_in = self.n_cand + (2 if self.use_plane else 0)
            self.direct_proj = nn.Conv2d(direct_in, width0, 3, padding=1)
        self.register_buffer("disp_index", cands)   # candidate disparities (tokens)

    def _base_grid(self, B, H, W, device, dtype):
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
            indexing="ij",
        )
        grid = torch.stack([xs, ys], dim=-1)            # (H, W, 2)
        return grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

    def _shift_x(self, feat, dx, base_grid):
        """Sample feat at column (x + dx). dx in token units: scalar or (B,1,H,W)."""
        W = feat.shape[-1]
        grid = base_grid.clone()
        off = 2.0 * dx / max(W - 1, 1)
        if torch.is_tensor(off):           # per-pixel map (B,1,H,W) -> (B,H,W)
            off = off[:, 0]
        grid[..., 0] = grid[..., 0] + off
        return F.grid_sample(feat, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=True)

    def forward(self, L_img, R_img, extra_cv=None):
        """
        extra_cv: optional (B, n_extra_cand, H, W) cost volume to fuse before agg,
                  e.g. a DINOCostVolume already upsampled to (H, W). Required when
                  use_own_cv=False (DINO-cost-volume-only mode).
        """
        fL = F.normalize(self.encoder(L_img), dim=1)
        fR = F.normalize(self.encoder(R_img), dim=1)
        B, C, H, W = fL.shape
        base = self._base_grid(B, H, W, fL.device, fL.dtype)

        # Correlation over the (possibly sub-token) disparity candidates.
        # disp_left: left-view convention (matches GT/main-model frame) — fL stays put,
        #   fR is sampled at x - d, so d_hat is a disparity directly on the left pixel grid.
        # else: symmetric/cyclopean — fL at x + d/2 against fR at x - d/2.
        # use_own_cv=False skips this loop entirely (the expensive per-candidate
        # grid_sample pass) -- agg then reads only extra_cv (e.g. DINO's cost
        # volume). fL/fR from the encoder above are still used below for the
        # appearance warp+fuse step regardless of use_own_cv.
        if self.use_own_cv:
            costs = []
            for d in self.disp_index.tolist():
                if self.disp_left:
                    fL_d = fL
                    fR_d = self._shift_x(fR, -d, base)
                else:
                    fL_d = self._shift_x(fL, +0.5 * d, base)
                    fR_d = self._shift_x(fR, -0.5 * d, base)
                costs.append((fL_d * fR_d).sum(dim=1, keepdim=True))   # (B,1,H,W)
            cost = torch.cat(costs, dim=1)                             # (B,n_cand,H,W)

            # Concatenate extra cost volume (e.g. DINO semantic correlations) before agg
            if extra_cv is not None:
                cost = torch.cat([cost, extra_cv.to(cost.dtype)], dim=1)  # (B,n_cand+n_extra,H,W)
        else:
            assert extra_cv is not None, "use_own_cv=False requires extra_cv (DINO cost volume)"
            cost = extra_cv.to(fL.dtype)                                # (B,n_extra,H,W)

        P = self.agg(cost).softmax(dim=1)                         # (B,n_cand,H,W)
        d_idx = self.disp_index.view(1, -1, 1, 1).to(P.dtype)
        d_hat = (P * d_idx).sum(dim=1, keepdim=True)              # (B,1,H,W) tokens
        conf = P.max(dim=1, keepdim=True).values                  # (B,1,H,W)

        disp_px = d_hat * self.px_per_token                       # (B,1,H,W)

        # Border validity: columns where the R sample stays in-bounds.
        valid = torch.ones_like(d_hat)
        if self.disp_left:
            # Only fR is shifted (by up to -dmax), so only the left edge can go
            # out of bounds; the right edge is unaffected (fL is never shifted).
            b = self.dmax
            if b > 0:
                valid[..., :b] = 0.0
        else:
            # Symmetric half-shifts: both edges can go out of bounds.
            b = (self.dmax + 1) // 2
            if b > 0:
                valid[..., :b] = 0.0
                valid[..., -b:] = 0.0

        # Plane mode: slant (dx, dy) = disparity gradient in px/px, from the full cost volume.
        dxy = None
        if self.use_plane:
            dxy = self.slant(cost)                                # (B,2,H,W)

        if self.use_nowrap:
            # No warp, no fL/fR appearance, no d_hat/conf: project the fused
            # distribution P (+ slant, if plane mode) directly to width0.
            direct_in = torch.cat([P, dxy], dim=1) if self.use_plane else P
            joint = self.direct_proj(direct_in)                       # (B,width0,H,W)
        else:
            if self.disp_left:
                # Left-view: fL is already on the target grid; only warp fR onto it.
                fL_cyc = fL
                fR_cyc = self._shift_x(fR, -d_hat, base)
            else:
                # Dual half-warp onto the cyclopean grid.
                fL_cyc = self._shift_x(fL, +0.5 * d_hat, base)
                fR_cyc = self._shift_x(fR, -0.5 * d_hat, base)
            fused = torch.cat([fL_cyc, fR_cyc, fL_cyc - fR_cyc, d_hat, conf], dim=1)
            if self.use_plane:
                # B1: keep the full distribution P (multimodal hypotheses survive);
                # A1: append the slant so the token describes a plane, not a constant.
                fused = torch.cat([fused, P, dxy], dim=1)
            joint = self.fuse(fused)                                  # (B,width0,H,W)
        joint_tokens = joint.movedim(1, -1)                       # (B,H,W,width0)
        return joint_tokens, disp_px, conf, valid, dxy


class LeanCorrelationConditionerC2F(nn.Module):
    """Coarse-to-fine LCC: fixes the case where the target disparity range
    (e.g. 192px for SceneFlow) exceeds what a single fixed-window search can
    cover from a tile-sized crop -- see the "degenerate valid mask" issue in
    the base LeanCorrelationConditioner (dmax large relative to the token
    grid makes `valid` all-zero, so SCC supervision silently never fires).

    Instead of one global window searched from d=0, this does:
      Stage 1 (coarse): a separate, coarse-stride encoder over a WIDE L/R
        crop (wider than the tile of interest, same width for L and R, tile
        horizontally centered in it -- caller's responsibility). Searches
        the *entire* provided width (derived from the crop's own size, not
        a hand-picked cap) for a rough disparity anchor `d_coarse`.
      Stage 2 (fine): a fine-stride encoder over the SAME wide crop (needed
        because `d_coarse` can point anywhere within it -- the fine search
        can't be limited to the narrow tile itself). Searches only a small,
        fixed local radius around `d_coarse` (per-pixel, since d_coarse
        varies spatially) -- cheap, and correct regardless of how large the
        true disparity turns out to be, because the coarse stage already
        localized it.

    An optional extra_cv (an *anchored* DINOCostVolume output, i.e. called
    with anchor=d_coarse) can be fused in at the fine stage, same convention
    as the base LCC's extra_cv.

    disp_left=True: left-view convention (fL fixed, only fR shifts).
    Otherwise: symmetric/cyclopean (fL at +d/2, fR at -d/2), applied at
    both stages.

    Returns the same 5-tuple as LeanCorrelationConditioner:
      (joint_tokens, disp_px, conf, valid, dxy)
    so injection into the rest of the model is unchanged.
    """

    def __init__(self, feat_ch=64, fine_radius=8, width0=128, px_per_token=2,
                 coarse_px_per_token=32, disp_step=1.0, n_extra_cand=0,
                 use_plane=False, disp_left=False):
        super().__init__()
        self.px_per_token = float(px_per_token)
        self.coarse_px_per_token = float(coarse_px_per_token)
        self.fine_radius = float(fine_radius)
        self.disp_step = float(disp_step)
        self.use_plane = bool(use_plane)
        self.disp_left = bool(disp_left)

        fine_cands = torch.arange(-self.fine_radius, self.fine_radius + 1e-6,
                                   self.disp_step, dtype=torch.float32)
        self.n_cand = int(fine_cands.numel())
        self.register_buffer("fine_delta", fine_cands)

        # Fine encoder: same architecture/stride as the base LCC's encoder.
        s2 = 2 if int(px_per_token) >= 4 else 1
        self.fine_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, feat_ch, 3, stride=s2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
        )

        # Coarse encoder: stride-2 convs down to coarse_px_per_token total stride.
        n_down = round(math.log2(max(self.coarse_px_per_token, 1.0)))
        if 2 ** n_down != int(self.coarse_px_per_token):
            raise ValueError(f"coarse_px_per_token must be a power of 2, got {coarse_px_per_token}")
        coarse_layers = [nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True)]
        ch = 32
        for _ in range(1, n_down):
            out_ch = min(ch * 2, feat_ch)
            coarse_layers += [nn.Conv2d(ch, out_ch, 3, stride=2, padding=1), nn.ReLU(inplace=True)]
            ch = out_ch
        coarse_layers += [nn.Conv2d(ch, feat_ch, 3, padding=1)]
        self.coarse_encoder = nn.Sequential(*coarse_layers)

        # Learned "convex upsampling" (RAFT-style) for lifting d_coarse from
        # the coarse grid to the fine output grid. Plain bilinear upsampling
        # blends a coarse cell's value with its NEIGHBORS using fixed
        # position-based weights -- if one neighbor saw the true match and
        # another saw only background, bilinear still blends them by a fixed
        # geometric rule and can land far from either. Here, a small head
        # predicts (from fine-resolution content) a per-fine-pixel softmax
        # weighting over the local 3x3 coarse neighborhood instead, so the
        # network can learn to lean on the informative neighbor rather than
        # average blindly -- this is what RAFT/RAFT-Stereo use to go from
        # their 1/8-resolution working grid to full resolution.
        self.coarse_up_mask = nn.Sequential(
            nn.Conv2d(feat_ch, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 9, 1),
        )

        self.agg = nn.Sequential(
            nn.Conv2d(self.n_cand + n_extra_cand, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, self.n_cand, 3, padding=1),
        )
        # No learned aggregation for the coarse stage: its candidate count
        # varies at runtime with the wide crop's width (bigger crop near
        # image center, smaller near edges where it had to be clipped), so
        # it can't be a fixed-channel-count Conv2d like `agg` above. Instead,
        # a learnable scalar temperature sharpens the softmax -- raw cosine
        # similarities between L2-normalized features are bounded to [-1,1]
        # and often cluster in a narrow band (e.g. 0.47 vs 0.87 peak), so an
        # UNSCALED softmax is too flat and the soft-argmax weighted sum gets
        # dragged toward the candidate range's center instead of the true
        # peak, even when the peak is completely unambiguous. Init small
        # (sharp) since coarse candidates are plain dot products, not logits.
        self.coarse_temp = nn.Parameter(torch.tensor(0.05))
        # Coarse
        # candidates go through a plain softmax instead (see forward()).

        fuse_in = 3 * feat_ch + 2
        if self.use_plane:
            self.slant = nn.Sequential(
                nn.Conv2d(self.n_cand + n_extra_cand, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 2, 3, padding=1),
            )
            nn.init.zeros_(self.slant[-1].weight)
            nn.init.zeros_(self.slant[-1].bias)
            fuse_in += self.n_cand + 2
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, width0, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width0, width0, 3, padding=1),
        )

    @staticmethod
    def _roi_base_grid(B, H_out, W_out, W_src, offset_tok, device, dtype):
        """Grid representing the ROI's own H_out x W_out output positions,
        expressed in normalized [-1,1] coords of a WIDER source of width
        W_src, where the ROI starts at column `offset_tok` within that
        source. When W_src == W_out and offset_tok == 0 this is identical
        to the base grid the un-widened LCC builds from its own shape."""
        ys = torch.linspace(-1.0, 1.0, H_out, device=device, dtype=dtype)
        # output col i sits at source col (offset_tok + i); normalize by (W_src-1).
        xs = (2.0 * (offset_tok + torch.arange(W_out, device=device, dtype=dtype))
              / max(W_src - 1, 1)) - 1.0
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)
        return grid.unsqueeze(0).expand(B, -1, -1, -1)

    @staticmethod
    def _shift_x(feat, dx, base_grid):
        """Sample feat at column (x + dx). dx: scalar or (B,1,H,W) map, in
        TOKEN units of `feat`'s own width."""
        W = feat.shape[-1]
        grid = base_grid.clone()
        off = 2.0 * dx / max(W - 1, 1)
        if torch.is_tensor(off):
            off = off[:, 0]
        grid[..., 0] = grid[..., 0] + off
        return F.grid_sample(feat, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=True)

    @staticmethod
    def _convex_upsample(coarse, mask, up_h, up_w):
        """RAFT-style learned convex upsampling.
        coarse: (B,C,Hc,Wc).
        mask:   (B,9,Hc*up_h,Wc*up_w), softmax already applied over dim=1 --
                 a per-fine-pixel weighting over its enclosing coarse cell's
                 3x3 neighborhood.
        Returns (B,C,Hc*up_h,Wc*up_w): each fine pixel = weighted sum of
        that neighborhood, using the LEARNED (not fixed-bilinear) weights.
        """
        B, C, Hc, Wc = coarse.shape
        # 3x3 neighborhood per coarse cell: (B,C*9,Hc*Wc) -> (B,C,9,Hc,Wc)
        nb = F.unfold(coarse, kernel_size=3, padding=1).view(B, C, 9, Hc, Wc)
        # Broadcast (nearest) to fine resolution: every fine pixel inside a
        # given coarse cell shares that cell's (unchanged) 3x3 neighborhood
        # values -- only the MASK varies per fine pixel, not the candidates.
        nb = nb.reshape(B, C * 9, Hc, Wc)
        nb = F.interpolate(nb, size=(Hc * up_h, Wc * up_w), mode="nearest")
        nb = nb.view(B, C, 9, Hc * up_h, Wc * up_w)
        return (nb * mask.unsqueeze(1)).sum(dim=2)                # (B,C,Hfine,Wfine)

    def forward(self, L_wide, R_wide, dino_cv=None, L_dino=None, R_dino=None,
               dino_radius_px=None):
        """
        L_wide, R_wide: (B,3,H,Ww) -- a crop WIDER than the tile of interest
            (same Ww for both), with the tile of interest horizontally
            CENTERED within it. H is the tile's own height (no vertical
            widening needed -- disparity search is horizontal only).
        dino_cv, L_dino, R_dino, dino_radius_px: optional -- when given,
            fuses an ANCHORED DINOCostVolume (dino_cv) into the fine stage.
            Unlike the base LeanCorrelationConditioner (where extra_cv can
            be precomputed externally and just passed in), DINO's anchored
            search needs the coarse anchor computed *inside* this forward
            pass, so it's called here directly rather than accepted as a
            precomputed tensor. L_dino/R_dino: DINO's own modest crop
            (narrower than L_wide/R_wide -- see DINOCostVolume.forward's
            anchored-mode docstring for why it shouldn't see the huge coarse
            strip). Its `valid` output is folded into this module's own.
        """
        B, _, H, Ww_px = L_wide.shape
        device, dtype = L_wide.device, L_wide.dtype
        tile_px = round(H)  # tile is square: height == its own width in pixels

        # ---------------- Coarse stage ----------------
        fLc = F.normalize(self.coarse_encoder(L_wide), dim=1)   # (B,C,Hc,Wc)
        fRc = F.normalize(self.coarse_encoder(R_wide), dim=1)
        Hc, Wc = fLc.shape[-2], fLc.shape[-1]
        tile_tok_c = max(1, round(tile_px / self.coarse_px_per_token))
        offset_c = (Wc - tile_tok_c) / 2.0
        base_c = self._roi_base_grid(B, Hc, tile_tok_c, Wc, offset_c, device, dtype)
        # Search the entire coarse width available: max half-shift such that
        # (roi center +/- shift) still touches the coarse source's real
        # extent -- derived from Wc, not a hand-picked constant.
        max_shift_c = max(offset_c, Wc - offset_c - tile_tok_c, 1.0)
        coarse_cands = torch.arange(-max_shift_c, max_shift_c + 1e-6, 1.0, device=device, dtype=dtype)
        costs_c = []
        for d in coarse_cands.tolist():
            if self.disp_left:
                fLc_d = self._shift_x(fLc, 0.0, base_c)
                fRc_d = self._shift_x(fRc, -d, base_c)
            else:
                fLc_d = self._shift_x(fLc, +0.5 * d, base_c)
                fRc_d = self._shift_x(fRc, -0.5 * d, base_c)
            costs_c.append((fLc_d * fRc_d).sum(dim=1, keepdim=True))
        cost_c = torch.cat(costs_c, dim=1)                      # (B,n_coarse,Hc,tile_tok_c)
        # No learned aggregation here (see __init__ note) -- temperature-
        # scaled softmax instead (plain/unscaled softmax on raw cosine
        # similarities is too flat -- see coarse_temp's docstring above).
        # max_shift_c is derived from the coarse source's own width, so this
        # search always stays in-bounds by construction (no separate
        # coarse-stage validity check needed).
        Pc = (cost_c / self.coarse_temp.clamp(min=1e-3)).softmax(dim=1)  # (B,n_coarse,Hc,tile_tok_c)
        idx_c = coarse_cands.view(1, -1, 1, 1)
        d_coarse_c = (Pc * idx_c).sum(dim=1, keepdim=True)       # (B,1,Hc,tile_tok_c), coarse-token units

        # ---------------- Fine encoder (needed both for the learned
        # upsample mask below, and for the fine correlation afterward) ----
        Ht_out = round(tile_px / self.px_per_token)
        Wt_out = Ht_out
        fLf = F.normalize(self.fine_encoder(L_wide), dim=1)     # (B,C,Hf,Wf)
        fRf = F.normalize(self.fine_encoder(R_wide), dim=1)
        Hf, Wf = fLf.shape[-2], fLf.shape[-1]
        offset_f = (Wf - Wt_out) / 2.0
        base_f = self._roi_base_grid(B, Ht_out, Wt_out, Wf, offset_f, device, dtype)

        # Lift d_coarse to the fine grid with LEARNED convex upsampling
        # (RAFT-style) instead of plain bilinear -- see coarse_up_mask's
        # docstring for why: fixed bilinear weights blend an informative
        # coarse cell with an uninformative neighbor regardless of content,
        # which silently destroys the coarse stage's otherwise-correct
        # estimate whenever tile_tok_c is small relative to Ht_out/Wt_out.
        up_h, up_w = Ht_out // Hc, Wt_out // tile_tok_c
        if up_h * Hc != Ht_out or up_w * tile_tok_c != Wt_out:
            raise ValueError(
                "coarse_px_per_token must be an integer multiple of px_per_token "
                f"for convex upsampling (got coarse={self.coarse_px_per_token}, "
                f"fine={self.px_per_token})")
        fL_roi = self._shift_x(fLf, torch.zeros(B, 1, Ht_out, Wt_out, device=device, dtype=dtype), base_f)
        up_mask = self.coarse_up_mask(fL_roi).softmax(dim=1)      # (B,9,Ht_out,Wt_out)
        d_coarse_fine = self._convex_upsample(d_coarse_c, up_mask, up_h, up_w)  # (B,1,Ht_out,Wt_out)
        d_coarse_fine = d_coarse_fine * (self.coarse_px_per_token / self.px_per_token)

        # ---------------- Anchored DINO (optional), now that d_coarse exists ----------------
        dino_valid = None
        extra_cv = None
        if dino_cv is not None:
            assert L_dino is not None and R_dino is not None and dino_radius_px is not None
            extra_cv, dino_valid = dino_cv(L_dino, R_dino, Ht_out, Wt_out,
                                           anchor_px=d_coarse_fine * self.px_per_token,
                                           radius=dino_radius_px)

        # ---------------- Fine stage (anchored on d_coarse) ----------------
        costs = []
        for delta in self.fine_delta.tolist():
            d_map = d_coarse_fine + delta                        # (B,1,Ht_out,Wt_out)
            if self.disp_left:
                fL_d = self._shift_x(fLf, torch.zeros_like(d_map), base_f)
                fR_d = self._shift_x(fRf, -d_map, base_f)
            else:
                fL_d = self._shift_x(fLf, +0.5 * d_map, base_f)
                fR_d = self._shift_x(fRf, -0.5 * d_map, base_f)
            costs.append((fL_d * fR_d).sum(dim=1, keepdim=True))
        cost = torch.cat(costs, dim=1)                            # (B,n_cand,Ht_out,Wt_out)

        if extra_cv is not None:
            cost_agg_in = torch.cat([cost, extra_cv.to(cost.dtype)], dim=1)
        else:
            cost_agg_in = cost
        P = self.agg(cost_agg_in).softmax(dim=1)
        delta_idx = self.fine_delta.view(1, -1, 1, 1).to(P.dtype)
        delta_hat = (P * delta_idx).sum(dim=1, keepdim=True)     # (B,1,Ht_out,Wt_out)
        conf = P.max(dim=1, keepdim=True).values

        d_hat = d_coarse_fine + delta_hat                         # (B,1,Ht_out,Wt_out), fine-token units
        disp_px = d_hat * self.px_per_token

        # Validity, computed PER PIXEL (not a single batch-wide gate): does
        # the actual shift range used for fL and fR -- coarse anchor +/- the
        # fine radius -- stay within the wide source's real extent at this
        # specific output position? With a generously-sized wide crop this
        # is ~all-1 across the interior; it correctly drops to 0 only where
        # the wide crop itself ran out of real content (true image edges),
        # not as an artifact of dmax exceeding the tile -- that's the fix.
        col_idx = torch.arange(Wt_out, device=device, dtype=dtype).view(1, 1, 1, Wt_out)
        src_pos = offset_f + col_idx                              # (1,1,1,Wt_out)

        def _in_bounds(lo, hi):
            return (lo >= 0) & (hi <= (Wf - 1))

        if self.disp_left:
            fL_lo = fL_hi = src_pos.expand(B, 1, Ht_out, Wt_out)
            fR_lo = src_pos - (d_coarse_fine + self.fine_radius)
            fR_hi = src_pos - (d_coarse_fine - self.fine_radius)
        else:
            fL_lo = src_pos + 0.5 * (d_coarse_fine - self.fine_radius)
            fL_hi = src_pos + 0.5 * (d_coarse_fine + self.fine_radius)
            fR_lo = src_pos - 0.5 * (d_coarse_fine + self.fine_radius)
            fR_hi = src_pos - 0.5 * (d_coarse_fine - self.fine_radius)
        valid = (_in_bounds(fL_lo, fL_hi) & _in_bounds(fR_lo, fR_hi)).to(d_hat.dtype)
        if dino_valid is not None:
            valid = valid * dino_valid.to(valid.dtype)

        dxy = None
        if self.use_plane:
            dxy = self.slant(cost_agg_in)

        if self.disp_left:
            fL_cyc = self._shift_x(fLf, torch.zeros_like(d_hat), base_f)
            fR_cyc = self._shift_x(fRf, -d_hat, base_f)
        else:
            fL_cyc = self._shift_x(fLf, +0.5 * d_hat, base_f)
            fR_cyc = self._shift_x(fRf, -0.5 * d_hat, base_f)
        fused = torch.cat([fL_cyc, fR_cyc, fL_cyc - fR_cyc, d_hat, conf], dim=1)
        if self.use_plane:
            fused = torch.cat([fused, P, dxy], dim=1)
        joint = self.fuse(fused)
        joint_tokens = joint.movedim(1, -1)
        return joint_tokens, disp_px, conf, valid, dxy


class LeanCorrelationConditionerNative(nn.Module):
    """Pixel-resolution symmetric cost-volume LCC for channel-concat injection.

    Unlike LeanCorrelationConditioner (which downsamples 128→32 before matching),
    this class uses a stride-1 encoder so features and d_hat stay at full image
    resolution. Disparity candidates are in pixel units; the output is concat'd
    directly into the input without any upsampling.

    Returns:
      disp_px: (B, 1, H, W)  soft-argmax disparity in pixels, at image resolution
      conf:    (B, 1, H, W)  match confidence in [0, 1]
      valid:   (B, 1, H, W)  border-validity mask in {0, 1}
    """

    def __init__(self, feat_ch=32, dmax=64, disp_step=4.0):
        super().__init__()
        self.dmax = int(dmax)
        self.disp_step = float(disp_step)

        cands = torch.arange(0.0, self.dmax + 1e-6, self.disp_step, dtype=torch.float32)
        self.n_cand = int(cands.numel())

        # Stride-1 encoder: features stay at full image resolution.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, feat_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
        )
        # Cost-volume aggregation (same structure as token-level LCC).
        self.agg = nn.Sequential(
            nn.Conv2d(self.n_cand, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, self.n_cand, 3, padding=1),
        )
        self.register_buffer("disp_cands", cands)   # pixel-unit candidates

    def _base_grid(self, B, H, W, device, dtype):
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([xs, ys], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    def _shift_x(self, feat, dx, base_grid):
        """Sample feat at column (x + dx). dx in pixel units (feature map is pixel-res)."""
        W = feat.shape[-1]
        grid = base_grid.clone()
        off = 2.0 * dx / max(W - 1, 1)
        if torch.is_tensor(off):
            off = off[:, 0]
        grid[..., 0] = grid[..., 0] + off
        return F.grid_sample(feat, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=True)

    def forward(self, L_img, R_img):
        fL = F.normalize(self.encoder(L_img), dim=1)   # (B, feat_ch, H, W)
        fR = F.normalize(self.encoder(R_img), dim=1)
        B, C, H, W = fL.shape
        base = self._base_grid(B, H, W, fL.device, fL.dtype)

        # Symmetric correlation at full resolution: fL at x + d/2 pixels, fR at x - d/2 pixels.
        costs = []
        for d in self.disp_cands.tolist():
            fL_d = self._shift_x(fL, +0.5 * d, base)
            fR_d = self._shift_x(fR, -0.5 * d, base)
            costs.append((fL_d * fR_d).sum(dim=1, keepdim=True))
        cost = torch.cat(costs, dim=1)                        # (B, n_cand, H, W)

        P = self.agg(cost).softmax(dim=1)                    # (B, n_cand, H, W)
        d_idx = self.disp_cands.view(1, -1, 1, 1).to(P.dtype)
        disp_px = (P * d_idx).sum(dim=1, keepdim=True)       # (B, 1, H, W) pixels, native res
        conf = P.max(dim=1, keepdim=True).values             # (B, 1, H, W)

        # Border validity: columns where x ± dmax/2 pixels stays in-bounds.
        valid = torch.ones_like(disp_px)
        b = (self.dmax + 1) // 2   # pixel border width
        if b > 0 and b < W:
            valid[..., :b] = 0.0
            valid[..., -b:] = 0.0
        return disp_px, conf, valid


class DINOCostVolume(nn.Module):
    """Stereo cost volume built from frozen DINOv2 features.

    Extracts patch features from L and R images via a frozen DINOv2 ViT,
    applies a learnable 1×1 projection to calibrate for stereo correlation,
    then builds a symmetric correlation cost volume over disparity candidates
    (same cyclopean convention as LeanCorrelationConditioner).

    The cost volume is computed at DINO's native 16×16 spatial resolution
    (ViT-S/14 on 224² → 16×16 patches; each patch ≈ 8 real px at 128² input)
    and upsampled to the requested target size for concatenation with the CNN
    cost volume inside LeanCorrelationConditioner.agg.

    Args:
        model_name:  DINOv2 variant ('dinov2_vits14' → 384d, 'dinov2_vitb14' → 768d).
        proj_ch:     Channels after learnable projection (before correlation).
        dmax:        Max disparity in DINO grid units (1 unit ≈ 8 real px at 128² input).
        disp_step:   Step between candidates in DINO grid units.
    """

    def __init__(self, model_name='dinov2_vits14', proj_ch=64, dmax=8, disp_step=1.0,
                 disp_left=False):
        super().__init__()
        self.disp_left = bool(disp_left)
        self.disp_step = float(disp_step)
        self.dino = torch.hub.load('facebookresearch/dinov2', model_name,
                                   pretrained=True, verbose=False)
        for p in self.dino.parameters():
            p.requires_grad = False

        dino_dim = self.dino.embed_dim              # 384 (vits14) or 768 (vitb14)
        self.proj = nn.Conv2d(dino_dim, proj_ch, 1) # 1×1 learnable, calibrates for correlation

        cands = torch.arange(0.0, dmax + 1e-6, disp_step, dtype=torch.float32)
        self.n_cand = int(cands.numel())
        self.register_buffer('disp_cands', cands)

        self.register_buffer('dino_mean',
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('dino_std',
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _extract(self, img):
        """img: (B,3,H,W) in [0,1].  Returns (B, proj_ch, g, g) where g=16."""
        img_224 = F.interpolate(img.float(), (224, 224),
                                mode='bilinear', align_corners=False)
        img_n = (img_224 - self.dino_mean) / self.dino_std
        with torch.no_grad():
            feats = self.dino.get_intermediate_layers(img_n, n=1)[0]  # (B, N, D)
        B, N, D = feats.shape
        g = int(N ** 0.5)
        feats = feats.reshape(B, g, g, D).movedim(-1, 1)             # (B, D, g, g)
        return self.proj(feats.to(self.proj.weight.dtype))            # (B, proj_ch, g, g)

    def _base_grid(self, B, H, W, device, dtype):
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
            indexing='ij',
        )
        return torch.stack([xs, ys], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    def _shift_x(self, feat, dx, base_grid):
        W = feat.shape[-1]
        grid = base_grid.clone()
        off = 2.0 * dx / max(W - 1, 1)
        if torch.is_tensor(off):           # per-pixel map (B,1,H,W) -> (B,H,W)
            off = off[:, 0]
        grid[..., 0] = grid[..., 0] + off
        return F.grid_sample(feat, grid, mode='bilinear',
                             padding_mode='zeros', align_corners=True)

    def forward(self, L_img, R_img, target_H, target_W, anchor_px=None, radius=None):
        """
        anchor_px: optional (B,1,target_H,target_W) disparity anchor IN REAL
            PIXELS (same convention as LeanCorrelationConditionerC2F's
            disp_px), evaluated at the requested output resolution. When
            given, this module switches to an ANCHORED local search --
            candidates are `anchor + delta` for a small radius around the
            anchor (per-pixel, since the anchor varies spatially) instead of
            the fixed global range disp_cands -- and additionally returns a
            per-pixel `valid` mask (there is none in the unanchored mode).
            This is the fine-stage-only role DINO plays in the coarse-to-fine
            design: the CNN coarse stage alone is responsible for finding
            the right neighborhood (see LeanCorrelationConditionerC2F); DINO
            only ever refines locally once anchored, so it never needs to
            see more than a modest, close-to-square crop (avoids distorting
            DINOv2's frozen features via an extreme forced-square resize of
            a very wide strip).
        radius: local search radius in REAL PIXELS, required when anchor_px
            is given.
        Returns:
          cost only, (B, n_cand, target_H, target_W)          -- anchor_px is None (unchanged)
          (cost, valid), both (B, n_cand_or_1, target_H, target_W) -- anchor_px given
        """
        fL = F.normalize(self._extract(L_img), dim=1)  # (B, proj_ch, g, g)
        fR = F.normalize(self._extract(R_img), dim=1)
        B, C, H, W = fL.shape
        base = self._base_grid(B, H, W, fL.device, fL.dtype)

        if anchor_px is None:
            costs = []
            for d in self.disp_cands.tolist():
                if self.disp_left:
                    fL_d = fL
                    fR_d = self._shift_x(fR, -d, base)
                else:
                    fL_d = self._shift_x(fL, +0.5 * d, base)
                    fR_d = self._shift_x(fR, -0.5 * d, base)
                costs.append((fL_d * fR_d).sum(dim=1, keepdim=True))  # (B, 1, g, g)
            cost = torch.cat(costs, dim=1)                             # (B, n_cand, g, g)
            # Upsample to LCC token grid for concatenation in agg
            return F.interpolate(cost.float(), (target_H, target_W),
                                 mode='bilinear', align_corners=False).to(fL.dtype)

        # ---- Anchored (coarse-to-fine fine-stage) mode ----
        assert radius is not None and radius > 0
        # DINO always resizes to a fixed 224x224 regardless of the input
        # crop's real width, so real-px-per-DINO-unit depends on whatever
        # crop the caller supplied -- computed dynamically here rather than
        # assuming the old hardcoded "~8px at 128px input" (which was only
        # ever an approximation for that one specific input size).
        px_per_unit = L_img.shape[-1] / float(W)
        anchor_g = F.interpolate(anchor_px, size=(H, W), mode="bilinear",
                                 align_corners=False).to(fL.dtype) / px_per_unit  # DINO units
        local = torch.arange(-radius / px_per_unit, radius / px_per_unit + 1e-6,
                             self.disp_step, device=fL.device, dtype=fL.dtype)

        col_idx = torch.arange(W, device=fL.device, dtype=fL.dtype).view(1, 1, 1, W)

        def _in_bounds(lo, hi):
            return (lo >= 0) & (hi <= (W - 1))

        costs = []
        for delta in local.tolist():
            d_map = anchor_g + delta                                  # (B,1,H,W)
            if self.disp_left:
                fL_d = fL
                fR_d = self._shift_x(fR, -d_map, base)
            else:
                fL_d = self._shift_x(fL, +0.5 * d_map, base)
                fR_d = self._shift_x(fR, -0.5 * d_map, base)
            costs.append((fL_d * fR_d).sum(dim=1, keepdim=True))
        cost = torch.cat(costs, dim=1)                                 # (B, n_local, g, g)

        # Per-pixel validity, same style as LeanCorrelationConditionerC2F's
        # fine-stage bounds check: does the actual shift range used (anchor
        # +/- radius) stay within this crop's real extent?
        if self.disp_left:
            fL_lo = fL_hi = col_idx.expand(B, 1, H, W)
            fR_lo = col_idx - (anchor_g + radius / px_per_unit)
            fR_hi = col_idx - (anchor_g - radius / px_per_unit)
        else:
            fL_lo = col_idx + 0.5 * (anchor_g - radius / px_per_unit)
            fL_hi = col_idx + 0.5 * (anchor_g + radius / px_per_unit)
            fR_lo = col_idx - 0.5 * (anchor_g + radius / px_per_unit)
            fR_hi = col_idx - 0.5 * (anchor_g - radius / px_per_unit)
        valid = (_in_bounds(fL_lo, fL_hi) & _in_bounds(fR_lo, fR_hi)).to(fL.dtype)

        cost_up = F.interpolate(cost.float(), (target_H, target_W),
                                mode='bilinear', align_corners=False).to(fL.dtype)
        valid_up = F.interpolate(valid, (target_H, target_W), mode='nearest')
        return cost_up, valid_up


class FoJCondTransformerV2StereoDual(ImageTransformerDenoiserModelV2):
    """
    Stereo FoJ denoiser with TWO independent image encoders (left & right).

    Inputs:
      x:        (B, 2, H, W)  where channel 0 = UDF (already normalized by dataset u_scale),
                              and channel 1 = disparity in PIXELS.
      aug_cond: (B, 6, H, W)  [Left RGB (3), Right RGB (3)]

    Outputs:
      (B, 2, H, W)  same channel order; returns UDF in dataset units, disparity in PIXELS.

    Internally:
      - Disparity is normalized by disp_norm for stable training and scaled back at the end.
      - UDF passes through unchanged (already ~O(1) from the dataset).
    """

    def __init__(
        self,
        levels,
        mapping,
        in_channels,      # expect 2
        out_channels,     # expect 2
        patch_size,       # int or (Ph, Pw); must be square & divide H/W
        num_classes=0,
        mapping_cond_dim=128,
        cond_channels=6,  # 3 (left) + 3 (right)
        disp_norm=64.0,   # internal normalization for disparity (pixels)
        use_amodal_head=False,
        use_cross_view_diff=False,
        use_refine_head=False,
        refine_head_ch=64,      # hidden channels in the refinement CNN
        refine_head_depth=3,    # total conv layers (min 2: first + last)
        use_scc=False,
        scc_dmax=16,
        scc_feat_ch=64,
        scc_disp_step=1.0,
        scc_plane=False,            # A1+B1: slanted-plane head + full-distribution injection in LCC
        scc_nowrap=False,           # skip warp/fuse; project fused P (+slant) directly to width0
        disp_left=False,            # left-view matching (fL fixed, fR shifted by -d) instead of cyclopean
        use_scc_c2f=False,          # coarse-to-fine LCC (LeanCorrelationConditionerC2F) instead of the
                                     # fixed-window LCC -- fixes the degenerate valid-mask bug that occurs
                                     # when scc_dmax is large relative to the token grid (see
                                     # LeanCorrelationConditionerC2F's docstring); requires a WIDE L/R
                                     # crop from the caller (forward()'s scc_coarse_L/scc_coarse_R
                                     # extra_args), not just the usual tile-sized aug_cond.
        scc_c2f_coarse_px_per_token=32,  # coarse-stage stride (must be a multiple of patch_size[0])
        scc_c2f_fine_radius=8,      # fine-stage local search radius, in FINE tokens
        scc_dino_c2f_radius_px=16,  # DINO's local search radius in REAL PIXELS (c2f mode only)
        scc_dino_c2f_crop_px=192,   # FIXED width of the modest crop fed to DINO in c2f mode -- must
                                     # match exactly what forward() actually supplies (enforced there),
                                     # since it determines DINO's anchored candidate count at init time
        use_scc_native=False,       # channel-concat LCC at native image resolution (stride-1 encoder)
        scc_native_dmax=64,         # max disparity in pixels for native LCC
        scc_native_feat_ch=32,      # encoder channels for native LCC
        scc_native_disp_step=4.0,   # pixel step between disparity candidates
        use_scc_dino_cv=False,      # fuse DINOv2 semantic cost volume into LCC before agg
        scc_dino_dmax=8,            # max DINO disparity shift (1 unit ≈ 8 real px at 128² input)
        scc_dino_proj_ch=64,        # learnable projection channels before DINO correlation
        scc_dino_model_name='dinov2_vits14',
        scc_use_own_cv=True,        # include LCC's own CNN-based correlation channels as input to
                                     # agg/slant. False = DINO-cost-volume-only mode: agg/slant read
                                     # only the DINO cost volume (requires use_scc_dino_cv=True); LCC's
                                     # own encoder still runs for the appearance warp+fuse step. Only
                                     # supported by the plain LeanCorrelationConditioner, not the C2F
                                     # variant (see the use_scc_c2f validation below).
        use_raft_disp=False,        # channel-concat externally-provided sharp disparity (e.g. RAFT or GT)
        raft_disp_norm=64.0,        # divide incoming disparity (px) by this before concatenation
        use_img_chan_concat=False,  # concat raw L/R RGB (6ch) to x before patch_in
        **kw
    ):
        # use_scc_native / use_raft_disp each add one channel before patch_in.
        # use_img_chan_concat adds 6 (L+R RGB) before patch_in.
        k_chancat = (1 if use_scc_native else 0) + (1 if use_raft_disp else 0) + (6 if use_img_chan_concat else 0)
        super().__init__(levels, mapping, in_channels + k_chancat, out_channels,
                         patch_size, num_classes, mapping_cond_dim, **kw)

        # --- Validate channels ---
        # if in_channels != 2 or out_channels != 2:
        #     raise ValueError(f"in/out channels must be 2 (UDF, DISP). Got in={in_channels}, out={out_channels}.")
        # if cond_channels != 6:
        #     raise ValueError(f"cond_channels must be 6 (L/R RGB). Got {cond_channels}.")
        # self.cond_channels = cond_channels

        # # --- Scales as buffers so they move with .to(device) and save in state dict ---
        # self.register_buffer("in_scale",  torch.tensor([1.0, 1.0 / float(disp_norm)]))  # multiply inputs by this
        # self.register_buffer("out_scale", torch.tensor([1.0, float(disp_norm)]))        # multiply outputs by this

        if not (in_channels in (1, 2, 3) and out_channels in (1, 2, 3)):
            raise AssertionError(f"in/out channels must be 1, 2, or 3; got in={in_channels}, out={out_channels}")

        self.in_channels = in_channels      # raw input channels (before chancat expansion)
        self.out_channels = out_channels
        self.use_scc_native = use_scc_native
        self.use_raft_disp = use_raft_disp
        self.k_chancat = k_chancat
        self.register_buffer('raft_disp_norm', torch.tensor(float(raft_disp_norm)))

        # make scale buffers match out/in channels
        # (these are commonly registered buffers that checkpoint stores with shape [C])
        # in_scale covers only the original input channels; SCC channels are appended unscaled.
        self.register_buffer("in_scale",  torch.ones(in_channels, dtype=torch.float32))
        self.register_buffer("out_scale", torch.ones(out_channels, dtype=torch.float32))

        # --- Patch size handling ---
        if isinstance(patch_size, int):
            P = (patch_size, patch_size)
        else:
            assert isinstance(patch_size, (tuple, list)) and len(patch_size) == 2, \
                "patch_size must be int or (Ph, Pw)."
            assert patch_size[0] == patch_size[1], "Use square patch_size."
            P = (int(patch_size[0]), int(patch_size[1]))

        width0 = levels[0].width

        # ---------- GLOBAL encoders (independent L/R) ----------
        def make_global_enc():
            return nn.Sequential(
                nn.Conv2d(3, mapping_cond_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mapping_cond_dim, mapping_cond_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),  # -> [B, D, 1, 1]
            )
        self.image_encoder_global_L = make_global_enc()
        self.image_encoder_global_R = make_global_enc()
        self.global_fuse = nn.Linear(2 * mapping_cond_dim, mapping_cond_dim)

        # ---------- SPATIAL encoders (independent L/R), patch-aligned ----------
        def make_spatial_enc():
            return nn.Sequential(
                nn.Conv2d(3, width0, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.cond_pre_L = make_spatial_enc()
        self.cond_pre_R = make_spatial_enc()
        self.cond_patch_in_L = TokenMerge(width0, width0, P)
        self.cond_patch_in_R = TokenMerge(width0, width0, P)

        # Learnable gates for additive fusion
        self.cond_gate_L = nn.Parameter(torch.tensor(1.0))
        self.cond_gate_R = nn.Parameter(torch.tensor(1.0))

        self.use_cross_view_diff = use_cross_view_diff
        if use_cross_view_diff:
            # Shared encoder so that identical L/R patches produce tL_d == tR_d
            # → tL_d - tR_d == 0 exactly for no-texture camouflage interiors.
            # Kept separate from cond_pre_L/R to preserve backward compatibility.
            self.cond_pre_diff      = make_spatial_enc()
            self.cond_patch_in_diff = TokenMerge(width0, width0, P)
            # Zero-init: starts as a no-op, learned from scratch
            self.cond_gate_diff = nn.Parameter(torch.tensor(0.0))
        
        self.use_img_chan_concat = use_img_chan_concat
        self.in_channels  = int(in_channels)
        self.out_channels = int(out_channels)
        self.expects_image_aug_cond = True

        # --- Optional amodal mask auxiliary head ---
        self.use_amodal_head = use_amodal_head
        if use_amodal_head:
            self.amodal_patch_out = TokenSplitWithoutSkip(width0, 2, P)
            nn.init.zeros_(self.amodal_patch_out.proj.weight)

        # --- Optional conv refinement head ---
        # Zero-init residual: starts as identity, learns sub-patch spatial correction.
        # Uses the stereo images as high-res guidance to resolve within-patch gradients.
        self.use_refine_head = use_refine_head
        if use_refine_head:
            refine_in = out_channels + 6  # prediction channels + L_img + R_img
            n_hidden = max(1, refine_head_depth - 2)
            layers = [nn.Conv2d(refine_in, refine_head_ch, 3, padding=1), nn.ReLU(inplace=True)]
            for _ in range(n_hidden):
                layers += [nn.Conv2d(refine_head_ch, refine_head_ch, 3, padding=1), nn.ReLU(inplace=True)]
            layers += [nn.Conv2d(refine_head_ch, out_channels, 3, padding=1)]
            self.refine_head = nn.Sequential(*layers)
            nn.init.zeros_(self.refine_head[-1].weight)
            nn.init.zeros_(self.refine_head[-1].bias)

        # --- Optional Lean Correlation Conditioner (symmetric cost-volume joint structure) ---
        # use_scc (token injection): zero-init gate, starts as no-op, added after patch_in.
        # use_scc_dino_cv: augments the CNN cost volume with DINOv2 semantic correlations before agg.
        # use_scc_native (native-res chancat): stride-1 encoder, d_hat at full image resolution.
        # use_scc_c2f: coarse-to-fine LCC (LeanCorrelationConditionerC2F) -- see its docstring.
        # Independent of use_scc_c2f: it changes what `self.scc` IS (fixed-window vs
        # coarse-to-fine) and, when DINO is also enabled, switches DINOCostVolume to
        # its anchored local-search mode (same module, different forward() call).
        self.use_scc_c2f = bool(use_scc_c2f)
        self.scc_dino_c2f_radius_px = float(scc_dino_c2f_radius_px)
        self.scc_dino_c2f_crop_px = int(scc_dino_c2f_crop_px)
        self.use_scc_dino_cv = use_scc_dino_cv and use_scc
        n_dino_cand = 0
        if use_scc and use_scc_dino_cv:
            self.dino_cv = DINOCostVolume(
                model_name=scc_dino_model_name,
                proj_ch=scc_dino_proj_ch,
                dmax=scc_dino_dmax,
                disp_left=disp_left,
            )
            if self.use_scc_c2f:
                # Anchored mode's candidate count depends on radius/crop_px
                # (fixed at init time -- forward() must supply a crop of
                # exactly scc_dino_c2f_crop_px width, enforced there), NOT
                # dmax, which only governs the (unused, in this mode) fixed
                # global-range search.
                px_per_unit = self.scc_dino_c2f_crop_px / 16.0  # DINOv2 ViT-S/14 @ 224/14=16 grid
                n_dino_cand = int(torch.arange(
                    -self.scc_dino_c2f_radius_px / px_per_unit,
                    self.scc_dino_c2f_radius_px / px_per_unit + 1e-6,
                    self.dino_cv.disp_step).numel())
            else:
                n_dino_cand = self.dino_cv.n_cand

        self.use_scc = use_scc
        if use_scc:
            if self.use_scc_c2f:
                if not scc_use_own_cv:
                    raise ValueError(
                        "scc_use_own_cv=False (DINO-cost-volume-only) is not supported "
                        "with use_scc_c2f=True -- only the plain LeanCorrelationConditioner "
                        "supports this toggle.")
                self.scc = LeanCorrelationConditionerC2F(
                    feat_ch=scc_feat_ch, fine_radius=scc_c2f_fine_radius, width0=width0,
                    px_per_token=P[0], coarse_px_per_token=scc_c2f_coarse_px_per_token,
                    disp_step=scc_disp_step, n_extra_cand=n_dino_cand,
                    use_plane=scc_plane, disp_left=disp_left)
            else:
                self.scc = LeanCorrelationConditioner(
                    feat_ch=scc_feat_ch, dmax=scc_dmax, width0=width0,
                    px_per_token=P[0], disp_step=scc_disp_step,
                    n_extra_cand=n_dino_cand, use_plane=scc_plane, use_nowrap=scc_nowrap,
                    disp_left=disp_left, use_own_cv=scc_use_own_cv)
            self.scc_gate = nn.Parameter(torch.tensor(0.0))
        if use_scc_native:
            self.scc_native = LeanCorrelationConditionerNative(
                feat_ch=scc_native_feat_ch, dmax=scc_native_dmax, disp_step=scc_native_disp_step)

        # Safety: ensure mapping_cond is wired in the base class
        assert self.mapping_cond_in_proj is not None, (
            "mapping_cond_dim=0 in config; set a positive value (e.g., 128)."
        )

    def forward(self, x, sigma, aug_cond=None, class_cond=None, mapping_cond=None, return_amodal=False, return_scc=False, raft_disp=None,
               scc_coarse_L=None, scc_coarse_R=None):
        # ---- Input checks ----
        if aug_cond is None:
            raise ValueError("Require aug_cond (B,6,H,W) = [L_RGB(3), R_RGB(3)].")
        if not (x.ndim == 4 and x.shape[1] == self.in_channels):
            raise ValueError(f"FoJ input shape must be (B, {self.in_channels}, H, W). Got {tuple(x.shape)}.")
        if not (aug_cond.ndim == 4 and aug_cond.shape[1] == 6):
            raise ValueError(f"aug_cond must be (B, 6, H, W). Got {tuple(aug_cond.shape)}.")

        B = aug_cond.size(0)

        # ---------- Split conditioning into left/right ----------
        L_img, R_img = aug_cond[:, :3], aug_cond[:, 3:]   # (B,3,H,W) each

        # ---------- GLOBAL vectors (L/R), fuse ----------
        # Skipped when use_img_chan_concat=True: L/R enters via channel-concat at patch_in instead.
        if not self.use_img_chan_concat:
            gL = self.image_encoder_global_L(L_img).view(B, -1)  # (B, D)
            gR = self.image_encoder_global_R(R_img).view(B, -1)  # (B, D)
            g_fused = self.global_fuse(torch.cat([gL, gR], dim=1))  # (B, D)

        # ---------- Normalize inputs channel-wise for internal stability ----------
        # NOTE: in_scale covers only the original in_channels; SCC channels are appended unscaled.
        x = x * self.in_scale.view(1, -1, 1, 1).to(dtype=x.dtype)

        # ---------- LCC: token injection ----------
        scc_disp_px = scc_conf = scc_valid = scc_dxy = None
        _joint_tokens = None
        if self.use_scc:
            if self.use_scc_c2f:
                if scc_coarse_L is None or scc_coarse_R is None:
                    raise ValueError(
                        "use_scc_c2f=True requires scc_coarse_L/scc_coarse_R (a wide L/R crop, "
                        "tile-of-interest horizontally centered) passed to forward().")
                dino_cv = L_dino = R_dino = dino_radius = None
                if self.use_scc_dino_cv:
                    dino_cv = self.dino_cv
                    # DINO only ever needs a MODEST, close-to-square crop in
                    # c2f mode (anchored locally on the CNN coarse stage's
                    # own estimate) -- center-crop it out of the same wide
                    # strip rather than requiring a second caller-supplied
                    # tensor. Must match scc_dino_c2f_crop_px exactly (that's
                    # what dino_cv's anchored candidate count was sized to).
                    crop_px = self.scc_dino_c2f_crop_px
                    Ww = scc_coarse_L.shape[-1]
                    off = (Ww - crop_px) // 2
                    if off < 0:
                        raise ValueError(
                            f"scc_coarse_L is narrower ({Ww}px) than scc_dino_c2f_crop_px "
                            f"({crop_px}px) -- widen the coarse crop or shrink scc_dino_c2f_crop_px.")
                    L_dino = scc_coarse_L[:, :, :, off:off + crop_px]
                    R_dino = scc_coarse_R[:, :, :, off:off + crop_px]
                    dino_radius = self.scc_dino_c2f_radius_px
                _joint_tokens, scc_disp_px, scc_conf, scc_valid, scc_dxy = self.scc(
                    scc_coarse_L, scc_coarse_R,
                    dino_cv=dino_cv, L_dino=L_dino, R_dino=R_dino, dino_radius_px=dino_radius)
            else:
                dino_cv_feat = None
                if self.use_scc_dino_cv:
                    # LCC's own cost-volume resolution = H // px_per_token (stride matches
                    # px_per_token: stride-4 total when px_per_token=4, stride-2 when =2).
                    stride = int(self.scc.px_per_token)
                    Ht_lcc = L_img.shape[-2] // stride
                    Wt_lcc = L_img.shape[-1] // stride
                    dino_cv_feat = self.dino_cv(L_img, R_img, Ht_lcc, Wt_lcc)
                _joint_tokens, scc_disp_px, scc_conf, scc_valid, scc_dxy = self.scc(
                    L_img, R_img, extra_cv=dino_cv_feat)

        # ---------- LCC: native-res channel-concat (d_hat prepended before patch_in) ----------
        native_disp_px = native_conf = native_valid = None
        if self.use_scc_native:
            native_disp_px, native_conf, native_valid = self.scc_native(L_img, R_img)
            x = torch.cat([x, native_disp_px], dim=1)   # (B, in_channels+1, H, W)

        # ---------- RAFT / external sharp disparity channel-concat ----------
        if self.use_raft_disp:
            if raft_disp is not None:
                rd = (raft_disp.to(dtype=x.dtype) / self.raft_disp_norm.to(dtype=x.dtype)).clamp(0, 1)
            else:
                rd = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], dtype=x.dtype, device=x.device)
            x = torch.cat([x, rd], dim=1)

        # ---------- Early L/R RGB channel-concat ----------
        if self.use_img_chan_concat:
            x = torch.cat([x, L_img.to(dtype=x.dtype), R_img.to(dtype=x.dtype)], dim=1)

        # ---------- Patch FoJ input to tokens ----------
        x = x.movedim(-3, -1)      # (B, H, W, C)
        x = self.patch_in(x)       # (B, H/P, W/P, width0)

        # ---------- SPATIAL cond tokens (L/R), fuse additively ----------
        # Skipped when use_img_chan_concat=True: L/R already provided as input channels to patch_in.
        if not self.use_img_chan_concat:
            tL = self.cond_patch_in_L(self.cond_pre_L(L_img).movedim(-3, -1))
            tR = self.cond_patch_in_R(self.cond_pre_R(R_img).movedim(-3, -1))
            x = x + self.cond_gate_L.to(dtype=x.dtype) * tL + self.cond_gate_R.to(dtype=x.dtype) * tR
        if self.use_cross_view_diff:
            # Shared encoder: identical patches → tL_d == tR_d → diff == 0 exactly
            tL_d = self.cond_patch_in_diff(self.cond_pre_diff(L_img).movedim(-3, -1))
            tR_d = self.cond_patch_in_diff(self.cond_pre_diff(R_img).movedim(-3, -1))
            x = x + self.cond_gate_diff.to(dtype=x.dtype) * (tL_d - tR_d)

        # ---------- Token injection (v2 style): add into token stream after patch_in ----------
        if self.use_scc and _joint_tokens is not None:
            x = x + self.scc_gate.to(dtype=x.dtype) * _joint_tokens.to(dtype=x.dtype)

        # ---------- Positional encodings ----------
        pos = make_axial_pos(x.shape[-3], x.shape[-2], device=x.device).view(
            x.shape[-3], x.shape[-2], 2
        )

        # ---------- Mapping net (time + aug + class + global image vec) ----------
        if self.class_emb is not None and class_cond is None:
            raise ValueError("class_cond must be specified if num_classes > 0")

        c_noise   = torch.log(sigma) / 4
        time_emb  = self.time_in_proj(self.time_emb(c_noise[..., None]))
        aug_emb   = self.aug_in_proj(self.aug_emb(x.new_zeros([x.shape[0], 9])))
        class_emb = self.class_emb(class_cond) if self.class_emb is not None else 0
        img_cond  = 0 if self.use_img_chan_concat else self.mapping_cond_in_proj(g_fused)
        cond      = self.mapping(time_emb + aug_emb + class_emb + img_cond)

        # ---------- Hourglass ----------
        skips, poses = [], []
        for down_level, merge in zip(self.down_levels, self.merges):
            x = down_level(x, pos, cond)
            skips.append(x); poses.append(pos)
            x = merge(x); pos = downscale_pos(pos)

        x = self.mid_level(x, pos, cond)

        for up_level, split, skip, pos in reversed(list(zip(self.up_levels, self.splits, skips, poses))):
            x = split(x, skip)
            x = up_level(x, pos, cond)

        # ---------- Unpatch & restore original units ----------
        x = self.out_norm(x)
        out = self.patch_out(x).movedim(-1, -3)  # (B, 2, H, W) in normalized space

        # Undo internal disparity normalization; UDF passes through unchanged
        out = out * self.out_scale.view(1, -1, 1, 1).to(dtype=out.dtype)

        if self.use_refine_head:
            out = out + self.refine_head(
                torch.cat([out, L_img, R_img], dim=1).to(dtype=out.dtype))

        scc_aux = None
        if return_scc:
            if self.use_scc_native and native_disp_px is not None:
                scc_aux = {"disp": native_disp_px, "conf": native_conf, "valid": native_valid}
            elif self.use_scc and scc_disp_px is not None:
                scc_aux = {"disp": scc_disp_px, "conf": scc_conf, "valid": scc_valid,
                           "dxy": scc_dxy}

        if return_amodal and self.use_amodal_head:
            amodal = torch.sigmoid(self.amodal_patch_out(x).movedim(-1, -3))  # (B, 2, H, W)
            if scc_aux is not None:
                return out, amodal, scc_aux
            return out, amodal

        if scc_aux is not None:
            return out, scc_aux

        return out
