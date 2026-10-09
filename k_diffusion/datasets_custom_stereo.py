from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import random


def _wide_crop(img, top, left, ps, wide_px, H, W):
    """Extract a `wide_px`-wide (height `ps`, unchanged) crop from `img`,
    horizontally CENTERED on the tile at [top:top+ps, left:left+ps] -- must
    match LeanCorrelationConditionerC2F's own assumption of exact centering
    (offset = (Wc - tile_tok_c) / 2 in the model code). Pads with edge-
    replication where the window would run past the real image bounds
    (near true image edges); the model's own bounds check (comparing the
    coarse/fine search range against the crop's real extent) still catches
    genuinely out-of-frame content, since replicated padding doesn't
    fabricate a plausible disparity match, it just keeps shapes valid."""
    tile_center = left + ps / 2.0
    wl = int(round(tile_center - wide_px / 2.0))
    wr = wl + wide_px
    pad_left = max(0, -wl)
    pad_right = max(0, wr - W)
    wl_c, wr_c = max(0, wl), min(W, wr)
    crop = img[:, top:top + ps, wl_c:wr_c]
    if pad_left > 0 or pad_right > 0:
        crop = F.pad(crop, (pad_left, pad_right), mode="replicate")
    return crop


class FoJStereoDataset(Dataset):
    def __init__(
        self,
        root,
        left_img_path,
        right_img_path,
        disp_path,
        field_path=None,
        channels=None,
        size=128,
        image_glob="*.png",
        clip_max=None,
        u_scale=1.0,
        # Optional: per-object amodal mask dirs (both must be set to enable)
        mask_obj1_path=None,
        mask_obj2_path=None,
        # Optional: per-object UDF dirs for 3-channel layered prediction
        udf_obj1_path=None,
        udf_obj2_path=None,
        # If True, skip all one_object_* samples (useful for layered UDF training)
        two_object_only=False,
        # If True, also return a per-pixel disp-validity mask (1=valid, 0=invalid)
        # as the last element of the batch tuple. Disp .npy files may contain NaN
        # for pixels masked out upstream (e.g. prepare_sceneflow.py marking
        # disparities beyond disp_max as invalid rather than clipping them) --
        # this surfaces that mask so the training loss can exclude them instead
        # of silently training on NaN-derived zeros. Off by default so existing
        # configs/consumers (fixed-length tuple unpacking) are unaffected.
        return_disp_valid=False,
        # If True, also return a wide L/R crop (scc_coarse_L, scc_coarse_R)
        # centered on the same tile -- for LeanCorrelationConditionerC2F's
        # coarse stage, which needs real image content wider than the tile
        # to search across (see its docstring for why). Off by default.
        return_coarse_strip=False,
        coarse_strip_px=512,
        **kwargs,
    ):
        self.root = Path(root)
        self.size = int(size)
        self.clip_max = clip_max
        self.u_scale = float(u_scale) if u_scale is not None else 1.0
        self.channels = channels
        self.two_object_only = two_object_only
        self.return_disp_valid = bool(return_disp_valid)
        self.return_coarse_strip = bool(return_coarse_strip)
        self.coarse_strip_px = int(coarse_strip_px)

        self.left_dir  = Path(left_img_path)
        self.right_dir = Path(right_img_path)
        self.disp_dir  = Path(disp_path)
        self.field_dir = Path(field_path) if field_path is not None else None

        self.use_amodal = (mask_obj1_path is not None and mask_obj2_path is not None)
        if self.use_amodal:
            self.mask_obj1_dir = Path(mask_obj1_path)
            self.mask_obj2_dir = Path(mask_obj2_path)

        self.use_layered_udf = (udf_obj1_path is not None and udf_obj2_path is not None)
        if self.use_layered_udf:
            self.udf_obj1_dir = Path(udf_obj1_path)
            self.udf_obj2_dir = Path(udf_obj2_path)

        self.to_tensor = transforms.ToTensor()

        # ------------------------
        # Build valid item list
        # ------------------------
        items = []
        skipped_small = 0

        skipped_one_obj = 0
        left_paths = sorted(self.left_dir.glob(image_glob))
        for pL in left_paths:
            stem = pL.stem
            base = stem[:-5] if stem.endswith("_left") else stem

            if self.two_object_only and base.startswith("one_object_"):
                skipped_one_obj += 1
                continue

            pR = self.right_dir / f"{base}_right.png"
            pD = self.disp_dir  / f"{base}_disp.npy"

            if not (pR.exists() and pD.exists()):
                continue

            # Support bimodal fields (_field_A / _field_B) and single fields (_field).
            if self.field_dir is not None:
                pF_A = self.field_dir / f"{base}_field_A.npy"
                pF_B = self.field_dir / f"{base}_field_B.npy"
                pF_single = self.field_dir / f"{base}_field.npy"
                if pF_A.exists() and pF_B.exists():
                    pF_options = [pF_A, pF_B]
                elif pF_single.exists():
                    pF_options = [pF_single]
                else:
                    continue
            else:
                pF_options = None

            # Verify amodal mask files exist when enabled
            if self.use_amodal:
                pM1 = self.mask_obj1_dir / f"{base}_mask.npy"
                pM2 = self.mask_obj2_dir / f"{base}_mask.npy"
                if not (pM1.exists() and pM2.exists()):
                    continue
                amodal_item = (pM1, pM2)
            else:
                amodal_item = None

            # Verify per-object UDF files exist when enabled
            if self.use_layered_udf:
                pU1 = self.udf_obj1_dir / f"{base}_udf.npy"
                pU2 = self.udf_obj2_dir / f"{base}_udf.npy"
                if not (pU1.exists() and pU2.exists()):
                    continue
                is_one_object = base.startswith("one_object_")
                layered_item = (pU1, pU2, is_one_object)
            else:
                layered_item = None

            # Check spatial size BEFORE adding
            with Image.open(pL) as img:
                W, H = img.size

            if H < self.size or W < self.size:
                skipped_small += 1
                continue

            items.append((pL, pR, pD, pF_options, amodal_item, layered_item))

        self.items = items

        msg = (
            f"[FoJStereoDataset] Loaded {len(self.items)} samples | "
            f"Skipped {skipped_small} smaller than {self.size} | "
            f"Amodal masks: {'enabled' if self.use_amodal else 'disabled'} | "
            f"Layered UDF: {'enabled' if self.use_layered_udf else 'disabled'}"
        )
        if self.two_object_only:
            msg += f" | Skipped {skipped_one_obj} one-object samples (two_object_only=True)"
        print(msg)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pL, pR, pD, pF_options, amodal_item, layered_item = self.items[idx]
        pF = random.choice(pF_options) if pF_options is not None else None

        # ---------- Load tensors (NO resize) ----------
        L = self.to_tensor(Image.open(pL).convert("RGB"))  # (3,H,W)
        R = self.to_tensor(Image.open(pR).convert("RGB"))  # (3,H,W)

        disp_np = np.load(pD)
        if disp_np.ndim == 3:
            disp_np = disp_np[..., 0]
        # NaN marks pixels invalid upstream (e.g. disparity beyond disp_max in
        # prepare_sceneflow.py). Record validity before filling, so the filled
        # value (never used as a real target once masked) can't be mistaken for
        # a real disparity by any code that skips the mask.
        disp_valid_np = np.isfinite(disp_np)
        if not disp_valid_np.all():
            disp_np = np.nan_to_num(disp_np, nan=0.0)
        disp = torch.from_numpy(disp_np).float()[None]     # (1,H,W)
        disp_valid = torch.from_numpy(disp_valid_np).float()[None]  # (1,H,W)

        if pF is not None:
            udf_np = np.load(pF)
            if udf_np.ndim == 3:
                udf_np = udf_np[..., 0]
            udf = torch.from_numpy(udf_np).float()[None]   # (1,H,W)
            if self.clip_max is not None:
                udf = udf.clamp_(0.0, float(self.clip_max))
            if self.u_scale and self.u_scale != 1.0:
                udf = udf / self.u_scale
        else:
            udf = None

        _, H, W = disp.shape
        assert L.shape[-2:] == disp.shape[-2:], "L / disp misaligned!"
        if udf is not None:
            assert udf.shape[-2:] == disp.shape[-2:], "UDF / disp misaligned!"
        ps = self.size

        # ---------- Load amodal masks (optional) ----------
        if amodal_item is not None:
            pM1, pM2 = amodal_item
            m1 = torch.from_numpy(np.load(pM1).astype(np.float32))  # (H,W) or (H,W,1)
            m2 = torch.from_numpy(np.load(pM2).astype(np.float32))
            if m1.ndim == 3:
                m1 = m1[..., 0]
            if m2.ndim == 3:
                m2 = m2[..., 0]
            amodal_masks = torch.stack([m1, m2], dim=0)  # (2, H, W)
        else:
            amodal_masks = None

        # ---------- Load per-object UDFs for layered prediction (optional) ----------
        if layered_item is not None:
            pU1, pU2, is_one_object = layered_item

            def _load_udf(path):
                arr = np.load(path).astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[..., 0]
                t = torch.from_numpy(arr)[None]  # (1, H, W)
                if self.clip_max is not None:
                    t = t.clamp_(0.0, float(self.clip_max))
                if self.u_scale and self.u_scale != 1.0:
                    t = t / self.u_scale
                return t

            udf1 = _load_udf(pU1)
            if is_one_object:
                # No second object — fill with max distance (background)
                fill = float(self.clip_max) / self.u_scale if (self.clip_max is not None and self.u_scale and self.u_scale != 1.0) else float(self.clip_max or 1.0)
                udf2 = torch.full_like(udf1, fill)
            else:
                udf2 = _load_udf(pU2)
        else:
            udf1 = udf2 = None

        # ---------- Random aligned crop ----------
        top  = random.randint(0, H - ps)
        left = random.randint(0, W - ps)

        # Wide L/R crop for the coarse stage, BEFORE L/R get overwritten by
        # the narrow tile crop below -- must read from the full-resolution
        # tensors still in scope here.
        if self.return_coarse_strip:
            scc_coarse_L = _wide_crop(L, top, left, ps, self.coarse_strip_px, H, W)
            scc_coarse_R = _wide_crop(R, top, left, ps, self.coarse_strip_px, H, W)

        L = L[:, top:top+ps, left:left+ps]
        R = R[:, top:top+ps, left:left+ps]
        disp = disp[:, top:top+ps, left:left+ps]
        disp_valid = disp_valid[:, top:top+ps, left:left+ps]
        if udf is not None:
            udf = udf[:, top:top+ps, left:left+ps]
        if amodal_masks is not None:
            amodal_masks = amodal_masks[:, top:top+ps, left:left+ps]
        if udf1 is not None:
            udf1 = udf1[:, top:top+ps, left:left+ps]
            udf2 = udf2[:, top:top+ps, left:left+ps]

        # Stereo conditioning
        cond6 = torch.cat([L, R], dim=0)  # (6,ps,ps)

        aug_vec_9 = torch.zeros(9)

        # Layered UDF mode: 3-channel target [udf_obj1, udf_obj2, disp]
        if udf1 is not None:
            target = torch.cat([udf1, udf2, disp], dim=0)  # (3, H, W)
        elif self.channels is None:
            target = torch.cat([udf, disp], dim=0) if udf is not None else disp
        else:
            chs = list(self.channels)
            target_parts = []
            for ch in chs:
                if ch == 0:
                    if udf is None:
                        raise ValueError("channels includes 0 (UDF) but field_path/UDF is missing.")
                    target_parts.append(udf)
                elif ch == 1:
                    target_parts.append(disp)
                else:
                    raise ValueError(f"Unsupported channel index {ch}; expected 0 (UDF) or 1 (DISP).")
            target = torch.cat(target_parts, dim=0)

        # disp_valid, then scc_coarse_L/R (if enabled) come LAST, after
        # amodal_masks if present, in that fixed order -- so existing
        # fixed-length/positional consumers (e.g. train.py's
        # `reals, _, aug_cond, _disp_gt = batch[image_key]`) are unaffected
        # when both flags default to False, and enabling one doesn't shift
        # the position of the other.
        extra = (disp_valid,) if self.return_disp_valid else ()
        if self.return_coarse_strip:
            extra = extra + (scc_coarse_L, scc_coarse_R)
        if amodal_masks is not None:
            return (target, aug_vec_9, cond6, disp, amodal_masks) + extra, torch.tensor(0)
        return (target, aug_vec_9, cond6, disp) + extra, torch.tensor(0)
