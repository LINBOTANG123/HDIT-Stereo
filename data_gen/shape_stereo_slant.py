#!/usr/bin/env python3
"""
Generate synthetic stereo training data: rectified left/right image pairs with
ground-truth disparity, unsigned-distance boundary fields (UDF), and segmentation,
for scenes of simple shapes (triangles, squares, regular polygons, circles).

Geometry convention (cyclopean):
    Ground-truth disparity and UDF live on the cyclopean (midpoint) grid. A
    cyclopean column x matches column x + d/2 in the left image and x - d/2 in the
    right image, with disparity d >= 0 (nearer objects have larger d).

Generation modes (--mode):
    one         single object per image (always unambiguous).
    two         two objects (the workhorse). Supports same/different color and
                same/different depth. A same-color, same-depth, untextured pair is
                genuine figure-ground camouflage: which object owns the shared
                boundary is ambiguous, so TWO UDF fields are saved (_field_A /
                _field_B). Every other case saves a single _field.
    both        run 'one' then 'two'.
    multi       3+ objects per image (--multi-objects).
    comparison  paired conditions: {camouflage, non-camouflage} x {equal,
                non-equal disparity}, written to four subdirectories.

Stereo synthesis (--stereo-mode):
    slanted     (default) per-object homography tilt; GT defined on the tilted
                cyclopean grid.
    shift       simple +/- d/2 horizontal translation.

Shapes are chosen from --shapes (default: all of ALL_SHAPES). Textures are
procedural (checker/stripes/dots) or sampled from --texture-dir.

Outputs (subdirectories of --output-dir):
    left/ right/           rectified stereo images
    imgs/                  cyclopean image
    disp/                  GT disparity on the cyclopean grid (float32 .npy)
    disp_left/             GT disparity on the left-image grid
    fields/                UDF boundary field(s) on the cyclopean grid: _field.npy,
                            or _field_A/_field_B for the same-disp/same-color camouflage case
    fields_left/           UDF boundary field on the LEFT-image grid: _field.npy only
                            (the camouflage A/B ambiguity is a cyclopean-grid construct --
                            see _save_two_object_result -- so this is always single-valued,
                            built from the occlusion-composited left segmentation)
    SEG/                   integer segmentation map
    udf_vis/               UDF overlay visualization (unless --no-viz)
    amodal_mask_obj{1,2}/  per-object amodal masks   (two-object scenes)
    disp_obj{1,2}/         per-object disparity       (two-object scenes)
    udf_obj{1,2}/          per-object UDF             (two-object scenes)
"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw
import math
import cv2
from typing import Optional, List, Tuple, Dict, Any
import argparse

# -----------------------------------------------------------------------------
# Defaults (can be overridden via CLI)
# -----------------------------------------------------------------------------
OUTPUT_DIR = "data/generated"
VIS_DIR    = "udf_vis"
FIELD_DIR  = "fields"
SEG_DIR    = "SEG"
IMG_DIR    = "imgs"

LEFT_DIR      = "left"
RIGHT_DIR     = "right"
DISP_DIR      = "disp"
DISP_LEFT_DIR = "disp_left"
FIELD_LEFT_DIR = "fields_left"

# per-object amodal outputs (two-object scenes only)
MASK_OBJ1_DIR = "amodal_mask_obj1"
MASK_OBJ2_DIR = "amodal_mask_obj2"
DISP_OBJ1_DIR = "disp_obj1"
DISP_OBJ2_DIR = "disp_obj2"
UDF_OBJ1_DIR  = "udf_obj1"
UDF_OBJ2_DIR  = "udf_obj2"

# per-object outputs for three-strip folded-paper scenes
MASK_OBJ3_DIR = "amodal_mask_obj3"
DISP_OBJ3_DIR = "disp_obj3"
UDF_OBJ3_DIR  = "udf_obj3"

# Comparison mode: four paired subdirectories (created under out_root)
CAMO_EQ_DIR     = "camouflage_equal_disp"
CAMO_NEQ_DIR    = "camouflage_nonequal_disp"
NONCAMO_EQ_DIR  = "noncamo_equal_disp"
NONCAMO_NEQ_DIR = "noncamo_nonequal_disp"
COMPARISON_SUBDIRS = [CAMO_EQ_DIR, CAMO_NEQ_DIR, NONCAMO_EQ_DIR, NONCAMO_NEQ_DIR]

IMAGE_SIZE = (128, 128)

NUM_ONE_OBJECT       = 10000
NUM_TWO_OBJECT_DIFF  = 10000
NUM_TWO_OBJECT_SAME  = 10000

MIN_TRIANGLE_AREA = 500
REF_IMAGE_SIZE = 128.0  # resolution the pixel-based shape-size constants below were tuned for

UDF_VIZ_ALPHA = 0.55
UDF_VIZ_DMAX  = 64.0

# -----------------------------------------------------------------------------
# Shape helpers
# -----------------------------------------------------------------------------
def random_circle_pts(image_height, image_width, min_r=None, max_r=None, n_approx=64):
    H, W = image_height, image_width
    scale = min(H, W) / REF_IMAGE_SIZE
    if min_r is None:
        min_r = 20 * scale
    if max_r is None:
        max_r = 50 * scale
    margin = max_r + 5
    cx = random.uniform(margin, W - margin)
    cy = random.uniform(margin, H - margin)
    r  = random.uniform(min_r, max_r)
    phi0 = random.uniform(0, 2*math.pi)
    return [(cx + r*math.cos(phi0 + k*2*math.pi/n_approx),
             cy + r*math.sin(phi0 + k*2*math.pi/n_approx)) for k in range(n_approx)]

def _use_texture_for_object(mode_texture, p_texture):
    if mode_texture == "all":
        return True
    if mode_texture == "none":
        return False
    return random.random() < p_texture

def make_checker(H, W, fg=(30,30,30), bg=(220,220,220), cell=8):
    yy, xx = np.indices((H, W))
    mask = ((xx // cell) + (yy // cell)) % 2
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[mask == 0] = bg
    img[mask == 1] = fg
    return Image.fromarray(img, mode="RGB")

def make_stripes(H, W, fg=(30,30,30), bg=(220,220,220), period=12, angle_deg=0):
    yy = np.arange(H)[:, None]
    band = ((yy // period) % 2).astype(np.uint8)
    base = np.repeat(band, W, axis=1)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[base == 0] = bg
    img[base == 1] = fg
    pil = Image.fromarray(img, mode="RGB")
    if angle_deg % 360 != 0:
        pil = pil.rotate(angle_deg, resample=Image.NEAREST, expand=False, fillcolor=bg)
    return pil

def make_dots(H, W, fg=(30,30,30), bg=(220,220,220), spacing=14, radius=3):
    img = Image.new("RGB", (W, H), color=bg)
    d = ImageDraw.Draw(img)
    for y in range(spacing//2, H, spacing):
        for x in range(spacing//2, W, spacing):
            d.ellipse((x-radius, y-radius, x+radius, y+radius), fill=fg)
    return img

def random_pattern(H, W, fg, bg, kind: Optional[str] = None):
    if kind is None:
        kind = random.choice(["checker", "stripes", "dots"])
    if kind == "checker":
        cell = random.choice([6, 8, 10, 12])
        return make_checker(H, W, fg=fg, bg=bg, cell=cell)
    elif kind == "stripes":
        period = random.choice([8, 10, 12, 14, 16])
        ang = random.choice([0, 15, 30, 45, 60, 75])
        return make_stripes(H, W, fg=fg, bg=bg, period=period, angle_deg=ang)
    else:
        spacing = random.choice([12, 14, 16, 18])
        radius  = random.choice([2, 3, 4])
        return make_dots(H, W, fg=fg, bg=bg, spacing=spacing, radius=radius)

def get_texture_paths(tex_dir: str) -> List[str]:
    """Recursively collect all image paths from a directory."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(tex_dir)
        for f in files
        if os.path.splitext(f)[1].lower() in exts
    ]
    if not paths:
        raise ValueError(f"No image files found under '{tex_dir}'")
    return paths

def load_random_texture(tex_source: List, min_size: int = 256, max_size: int = 1024) -> np.ndarray:
    """Pick a random texture. tex_source can be a list of paths or preloaded numpy arrays."""
    item = random.choice(tex_source)
    if isinstance(item, np.ndarray):
        return item  # already loaded and resized by preload_textures()
    img = Image.open(item).convert("RGB")
    w, h = img.size
    short, long_ = min(w, h), max(w, h)
    if short < min_size:
        scale = min_size / short
        img = img.resize((round(w * scale), round(h * scale)), resample=Image.LANCZOS)
        w, h = img.size
        long_ = max(w, h)
    if long_ > max_size:
        scale = max_size / long_
        img = img.resize((round(w * scale), round(h * scale)), resample=Image.LANCZOS)
    return np.array(img, dtype=np.uint8)

def preload_textures(tex_paths: List[str], min_size: int = 256, max_size: int = 1024) -> List[np.ndarray]:
    """Load all texture images into memory once. Returns list of H×W×3 uint8 arrays."""
    arrays = []
    for p in tex_paths:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        short, long_ = min(w, h), max(w, h)
        if short < min_size:
            scale = min_size / short
            img = img.resize((round(w * scale), round(h * scale)), resample=Image.LANCZOS)
            w, h = img.size
            long_ = max(w, h)
        if long_ > max_size:
            scale = max_size / long_
            img = img.resize((round(w * scale), round(h * scale)), resample=Image.LANCZOS)
        arrays.append(np.array(img, dtype=np.uint8))
    return arrays

def polygon_bbox(pts: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)

def make_texture_spec(
    pts: List[Tuple[float, float]],
    color: Tuple[int, int, int],
    *,
    same_texture_field: Optional[Dict[str, Any]] = None,
    texture_hw: Tuple[int, int] = (192, 192),
    anchor_mode: str = "object",
    image_hw: Tuple[int, int] = IMAGE_SIZE,
    pattern_kind: Optional[str] = None,
    tex_array: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if same_texture_field is not None:
        spec = dict(same_texture_field)
        spec["anchor_mode"] = anchor_mode
    elif tex_array is not None:
        # External texture: crop-and-fit with no tiling; random rotation for variety.
        spec = {
            "tex_rgb": tex_array,
            "pattern_kind": "external",
            "repeat_u": 1.0,
            "repeat_v": 1.0,
            "phase_u": 0.0,
            "phase_v": 0.0,
            "theta": random.uniform(0.0, 2.0 * math.pi),
            "anchor_mode": anchor_mode,
            "image_hw": image_hw,
        }
    else:
        fg = tuple(max(0, c - 40) for c in color)
        bg = tuple(min(255, c + 60) for c in color)
        tex_h, tex_w = texture_hw
        if pattern_kind is None:
            pattern_kind = random.choice(["checker", "stripes", "dots"])
        tex_rgb = np.array(random_pattern(tex_h, tex_w, fg=fg, bg=bg, kind=pattern_kind), dtype=np.uint8)
        spec = {
            "tex_rgb": tex_rgb,
            "pattern_kind": pattern_kind,
            "repeat_u": random.uniform(1.5, 4.0),
            "repeat_v": random.uniform(1.5, 4.0),
            "phase_u": random.uniform(0.0, 1.0),
            "phase_v": random.uniform(0.0, 1.0),
            "theta": random.uniform(0.0, 2.0 * math.pi),
            "anchor_mode": anchor_mode,
            "image_hw": image_hw,
        }

    x0, y0, x1, y1 = polygon_bbox(pts)
    spec["bbox"] = (x0, y0, x1, y1)
    spec["image_hw"] = image_hw
    return spec

def sample_texture_map(texture_spec: Dict[str, Any], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    tex = texture_spec["tex_rgb"]
    tex_h, tex_w = tex.shape[:2]
    repeat_u = float(texture_spec["repeat_u"])
    repeat_v = float(texture_spec["repeat_v"])
    phase_u = float(texture_spec["phase_u"])
    phase_v = float(texture_spec["phase_v"])
    theta = float(texture_spec["theta"])
    x0, y0, x1, y1 = texture_spec["bbox"]

    if texture_spec.get("anchor_mode", "object") == "global":
        image_h, image_w = texture_spec.get("image_hw", IMAGE_SIZE)
        cx = 0.5 * image_w
        cy = 0.5 * image_h
        sx = max(8.0, 0.35 * image_w)
        sy = max(8.0, 0.35 * image_h)
    else:
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        sx = max(1.0, x1 - x0)
        sy = max(1.0, y1 - y0)

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = xs - cx
    dy = ys - cy
    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy

    uu = (xr / sx) * repeat_u + phase_u
    vv = (yr / sy) * repeat_v + phase_v

    # Bilinear interpolation with tiling
    u_px = np.mod(uu, 1.0) * (tex_w - 1)
    v_px = np.mod(vv, 1.0) * (tex_h - 1)

    ix0 = np.floor(u_px).astype(np.int32)
    iy0 = np.floor(v_px).astype(np.int32)
    ix1 = (ix0 + 1) % tex_w
    iy1 = (iy0 + 1) % tex_h

    fx = (u_px - ix0).astype(np.float32)[..., None]
    fy = (v_px - iy0).astype(np.float32)[..., None]

    c00 = tex[iy0, ix0].astype(np.float32)
    c10 = tex[iy0, ix1].astype(np.float32)
    c01 = tex[iy1, ix0].astype(np.float32)
    c11 = tex[iy1, ix1].astype(np.float32)

    result = (c00 * (1.0 - fx) * (1.0 - fy) +
              c10 * fx           * (1.0 - fy) +
              c01 * (1.0 - fx)   * fy         +
              c11 * fx           * fy)
    return np.clip(result, 0, 255).astype(np.uint8)

def make_background_texture_spec(
    H: int, W: int,
    tex_paths: Optional[List] = None,
    exclude_arrays: Optional[List[np.ndarray]] = None,
    exclude_colors: Optional[List[Tuple[int, int, int]]] = None,
    pattern_kind: Optional[str] = None,
    min_color_gap: float = 60.0,
) -> Dict[str, Any]:
    """
    Build a texture_spec covering the FULL image, for use as a textured
    backdrop behind the object(s). Reuses make_texture_spec/sample_texture_map
    (anchor_mode="global") so the backdrop gets the same tiling/rotation
    machinery as object textures -- no new sampling code needed.

    To guarantee the background never looks like an object's texture:
      - external textures (tex_paths given): exclude_arrays filters out any
        image array already chosen for an object in this scene (by identity,
        since preload_textures() returns a fixed list of arrays reused across
        samples), then picks randomly from what's left.
      - procedural fallback (no tex_paths): exclude_colors biases the backdrop's
        base color away from any object's color by at least min_color_gap
        (same gap logic as pick_two_colors), so its fg/bg tint reads as
        distinct even though the pattern kind may coincide.
    """
    pts_full = [(0.0, 0.0), (float(W), 0.0), (float(W), float(H)), (0.0, float(H))]

    tex_array = None
    if tex_paths:
        candidates = list(tex_paths)
        if exclude_arrays:
            filtered = [t for t in candidates if not any(t is e for e in exclude_arrays)]
            if filtered:
                candidates = filtered
        tex_array = random.choice(candidates)
        if not isinstance(tex_array, np.ndarray):
            tex_array = load_random_texture([tex_array])

    base_color = tuple(random.randint(50, 200) for _ in range(3))
    if tex_array is None and exclude_colors:
        tries = 0
        while tries < 50 and any(
            np.linalg.norm(np.array(base_color, dtype=np.float32) - np.array(c, dtype=np.float32)) < min_color_gap
            for c in exclude_colors
        ):
            base_color = tuple(random.randint(50, 200) for _ in range(3))
            tries += 1

    return make_texture_spec(
        pts_full, base_color,
        anchor_mode="global", image_hw=(H, W),
        pattern_kind=pattern_kind, tex_array=tex_array,
    )

def render_background_rgb(H: int, W: int, bg_spec: Dict[str, Any]) -> np.ndarray:
    """Rasterize a background texture_spec (see make_background_texture_spec)
    over the full (H,W) grid -> uint8 RGB array."""
    yy, xx = np.mgrid[0:H, 0:W]
    return sample_texture_map(bg_spec, xx.astype(np.float32), yy.astype(np.float32))

def textured_polygon_layer_rgba(H: int, W: int, pts, texture_spec: Dict[str, Any]) -> np.ndarray:
    mask = polygon_mask_uint8(pts, (H, W)) > 0
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return rgba
    rgb = sample_texture_map(texture_spec, xs.astype(np.float32), ys.astype(np.float32))
    rgba[ys, xs, :3] = rgb
    rgba[ys, xs, 3] = 255
    return rgba

def warp_textured_polygon_rgba(
    H: int,
    W: int,
    pts,
    texture_spec: Dict[str, Any],
    Hmat: np.ndarray,
    out_hw: Tuple[int, int],
) -> np.ndarray:
    out_h, out_w = out_hw
    mask0 = polygon_mask_uint8(pts, (H, W))
    warped_mask = warp_mask(mask0, Hmat, (out_h, out_w)) > 0
    rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    ys, xs = np.nonzero(warped_mask)
    if ys.size == 0:
        return rgba

    Hinv = np.linalg.inv(Hmat.astype(np.float64))
    pts_view = np.stack(
        [xs.astype(np.float64), ys.astype(np.float64), np.ones_like(xs, dtype=np.float64)],
        axis=0,
    )
    pts_src = Hinv @ pts_view
    src_x = (pts_src[0] / (pts_src[2] + 1e-12)).astype(np.float32)
    src_y = (pts_src[1] / (pts_src[2] + 1e-12)).astype(np.float32)

    rgb = sample_texture_map(texture_spec, src_x, src_y)
    rgba[ys, xs, :3] = rgb
    rgba[ys, xs, 3] = 255
    return rgba

def pick_two_colors(enforce_gap=True, min_gap=60, same_color=False):
    def rand_color():
        return np.array([random.randint(30, 220) for _ in range(3)], dtype=np.int32)

    c1 = rand_color()
    if same_color:
        c2 = c1.copy()
    else:
        c2 = rand_color()
        if enforce_gap:
            tries = 0
            while np.linalg.norm(c1.astype(np.float32) - c2.astype(np.float32)) < min_gap and tries < 50:
                c2 = rand_color()
                tries += 1

    return tuple(int(x) for x in c1), tuple(int(x) for x in c2)

def triangle_area(pts):
    (x1, y1), (x2, y2), (x3, y3) = pts
    return abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2

def random_triangle(image_height, image_width):
    H, W = image_height, image_width
    scale = min(H, W) / REF_IMAGE_SIZE
    min_area = MIN_TRIANGLE_AREA * scale * scale
    while True:
        pts = [(random.uniform(5, W-5), random.uniform(5, H-5)) for _ in range(3)]
        if triangle_area(pts) >= min_area:
            return pts

def random_square(image_height, image_width):
    H, W = image_height, image_width
    scale = min(H, W) / REF_IMAGE_SIZE
    side = random.uniform(20, 60) * scale
    max_x = W - side - 5
    max_y = H - side - 5
    if max_x <= 5 or max_y <= 5:
        side = min(W, H) * 0.3
        max_x = W - side - 5
        max_y = H - side - 5
    x0 = random.uniform(5, max_x)
    y0 = random.uniform(5, max_y)
    x1 = x0 + side
    y1 = y0 + side
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

def random_regular_ngon(n, image_height, image_width, min_r=None, max_r=None):
    H, W = image_height, image_width
    scale = min(H, W) / REF_IMAGE_SIZE
    if min_r is None:
        min_r = 20 * scale
    if max_r is None:
        max_r = 50 * scale
    margin = max_r + 5
    cx = random.uniform(margin, W - margin)
    cy = random.uniform(margin, H - margin)
    r  = random.uniform(min_r,  max_r)
    phi0 = random.uniform(0, 2*math.pi)
    return [(cx + r*math.cos(phi0 + k*2*math.pi/n),
             cy + r*math.sin(phi0 + k*2*math.pi/n)) for k in range(n)]

ALL_SHAPES = ["triangle", "square", "pentagon", "hexagon", "heptagon", "octagon", "circle"]

def random_shape(H, W, shapes=None):
    if shapes is None:
        shapes = ALL_SHAPES
    pick = random.choice(shapes)
    if pick == "triangle":
        return random_triangle(H, W)
    elif pick == "square":
        return random_square(H, W)
    elif pick == "circle":
        return random_circle_pts(H, W)
    else:
        n = {"pentagon": 5, "hexagon": 6, "heptagon": 7, "octagon": 8}[pick]
        return random_regular_ngon(n, H, W)

def make_folded_paper_geometry(
    H: int,
    W: int,
    *,
    bar_w_frac: float = 0.30,
    bar_h_frac: float = 0.72,
    jitter_frac: float = 0.04,
) -> Dict[str, Any]:
    """
    Rectangle (the central near panel) for the 'protruding folded paper'
    (Von Szily 1921; Ehrenstein & Gillam 1998) stimulus.

    The figure is a solid black bar. Triangular corner notches are removed
    per-eye by the renderer: cutting the top-right & bottom-right corners in the
    LEFT view and the top-left & bottom-left corners in the RIGHT view makes
    those corner triangles monocular (half-occluded), which the visual system
    reads as receding flaps so the central panel protrudes.

    bar_w_frac : rectangle width  as a fraction of W.
    bar_h_frac : rectangle height as a fraction of H.
    jitter_frac: random +/- jitter on centre and size, for variety across demo
                 samples. Set 0 for a perfectly centred bar.

    Returns the rectangle bounds (x0, x1, y0, y1) and centre (xc, yc).
    """
    jx = random.uniform(-jitter_frac, jitter_frac) * W
    jy = random.uniform(-jitter_frac, jitter_frac) * H
    js = random.uniform(1.0 - jitter_frac, 1.0 + jitter_frac)

    bar_w = bar_w_frac * W * js
    bar_h = bar_h_frac * H * js
    xc = 0.5 * W + jx
    yc = 0.5 * H + jy
    x0, x1 = xc - 0.5 * bar_w, xc + 0.5 * bar_w
    y0, y1 = yc - 0.5 * bar_h, yc + 0.5 * bar_h
    return dict(x0=x0, x1=x1, y0=y0, y1=y1, xc=xc, yc=yc)


def folded_paper_silhouette(
    x0: float, x1: float, y0: float, y1: float,
    cx: float, cy: float, cut: str = "right",
) -> List[Tuple[float, float]]:
    """
    'Protruding folded paper' per-eye silhouette: a rectangle whose main body
    edge is recessed, with TWO SEPARATE triangular protrusions (flaps) near the
    top and bottom of one side. The two triangles are disconnected -- between
    them the body edge is a plain straight vertical line.

      cut="right" -> flaps on the RIGHT side  (LEFT-eye view)
      cut="left"  -> flaps on the LEFT  side  (RIGHT-eye view)
      cut="both"  -> plain rectangle           (cyclopean / amodal panel)

    cx = flap protrusion (px); cy = each flap's vertical extent (px). Each flap
    is a triangle whose tip reaches the bounding edge (x1 or x0) at the inner
    end and slants back to the recessed body edge toward the corner.
    """
    if cut == "right":
        xb = x1 - cx
        return [(x0, y0), (xb, y0), (x1, y0 + cy), (xb, y0 + cy),
                (xb, y1 - cy), (x1, y1 - cy), (xb, y1), (x0, y1)]
    if cut == "left":
        xa = x0 + cx
        return [(x1, y0), (xa, y0), (x0, y0 + cy), (xa, y0 + cy),
                (xa, y1 - cy), (x0, y1 - cy), (xa, y1), (x1, y1)]
    # "both" (cyclopean / amodal): body recessed by cx on each side (so its
    # width matches the per-eye rectangle), with all four flaps reaching the
    # side edges within the [x0,x1] footprint.
    xa = x0 + cx
    xb = x1 - cx
    return [(xa, y0), (xb, y0), (x1, y0 + cy), (xb, y0 + cy),
            (xb, y1 - cy), (x1, y1 - cy), (xb, y1), (xa, y1),
            (x0, y1 - cy), (xa, y1 - cy), (xa, y0 + cy), (x0, y0 + cy)]

def polygon_mask_uint8(pts: List[Tuple[float,float]], image_size: Tuple[int,int]) -> np.ndarray:
    """Return 0/255 uint8 mask (H,W)."""
    H, W = image_size
    mask_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon(pts, outline=1, fill=255)
    return np.array(mask_img, dtype=np.uint8)

# -----------------------------------------------------------------------------
# Segmentation & UDF
# -----------------------------------------------------------------------------
def build_and_save_udf_from_segmap(segmap: np.ndarray, save_path: str, d_max: Optional[float] = None):
    seg = segmap.astype(np.int32)

    b = np.zeros_like(seg, dtype=np.uint8)
    b[1:,  :] |= (seg[1:,  :] != seg[:-1, :])
    b[:-1, :] |= (seg[:-1, :] != seg[1:,  :])
    b[:,  1:] |= (seg[:,  1:] != seg[:,  :-1])
    b[:, :-1] |= (seg[:, :-1] != seg[:,   1:])

    non_edge = (b == 0).astype(np.uint8) * 255
    D = cv2.distanceTransform(non_edge, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)

    if d_max is not None and d_max > 0:
        np.minimum(D, float(d_max), out=D)

    np.save(save_path, D[..., None])

def build_and_save_udf_from_mask(mask: np.ndarray, save_path: str, d_max=None):
    """Compute UDF from a single binary amodal object mask (full shape, ignoring occlusion)."""
    seg = mask.astype(np.int32)
    b = np.zeros_like(seg, dtype=np.uint8)
    b[1:,  :] |= (seg[1:,  :] != seg[:-1, :])
    b[:-1, :] |= (seg[:-1, :] != seg[1:,  :])
    b[:,  1:] |= (seg[:,  1:] != seg[:,  :-1])
    b[:, :-1] |= (seg[:, :-1] != seg[:,   1:])
    non_edge = (b == 0).astype(np.uint8) * 255
    D = cv2.distanceTransform(non_edge, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)
    if d_max is not None and d_max > 0:
        np.minimum(D, float(d_max), out=D)
    np.save(save_path, D[..., None])


def save_udf_visualization(img_pil: Image.Image, D: np.ndarray, out_path: str,
                           d_max_for_viz: Optional[float] = UDF_VIZ_DMAX,
                           alpha: float = UDF_VIZ_ALPHA):
    img = np.array(img_pil)

    if d_max_for_viz is not None and d_max_for_viz > 0:
        Dn = np.clip(D / d_max_for_viz, 0.0, 1.0)
    else:
        m = float(np.max(D)) or 1.0
        Dn = (D / m)

    D8 = (Dn * 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(D8, cv2.COLORMAP_TURBO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heat_rgb + (1 - alpha) * img).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)

# -----------------------------------------------------------------------------
# Stereo helpers
# -----------------------------------------------------------------------------
def sample_depth_to_disparity(focal_px: float, baseline: float, z_min: float, z_max: float) -> float:
    Z = random.uniform(max(1e-3, z_min), max(z_min + 1e-3, z_max))
    return float(focal_px * baseline / Z)

def sample_ordered_depths(z_min, z_max, gap=0.1):
    z_front = random.uniform(z_min, max(z_min, z_max - gap))
    z_back_min = min(z_max, z_front + gap)
    z_back = random.uniform(z_back_min, z_max)
    return z_front, z_back

# -----------------------------------------------------------------------------
# Slanted-plane (homography) helpers
# -----------------------------------------------------------------------------
def solid_polygon_layer_rgba(H: int, W: int, pts, color_rgb: Tuple[int,int,int]) -> np.ndarray:
    """Return RGBA uint8 array (H,W,4) with polygon filled and alpha=255, else alpha=0."""
    layer = Image.new("RGBA", (W, H), (0,0,0,0))
    d = ImageDraw.Draw(layer)
    d.polygon(pts, fill=(color_rgb[0], color_rgb[1], color_rgb[2], 255))
    return np.array(layer, dtype=np.uint8)

def pil_rgba_to_np(layer_pil: Image.Image) -> np.ndarray:
    return np.array(layer_pil.convert("RGBA"), dtype=np.uint8)

def warp_rgba(layer_rgba: np.ndarray, Hmat: np.ndarray, out_hw: Tuple[int,int],
              interp=cv2.INTER_LINEAR) -> np.ndarray:
    H, W = out_hw
    return cv2.warpPerspective(
        layer_rgba, Hmat, (W, H),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,0,0,0),
    )

def warp_mask(mask_u8_255: np.ndarray, Hmat: np.ndarray, out_hw: Tuple[int,int]) -> np.ndarray:
    H, W = out_hw
    warped = cv2.warpPerspective(
        mask_u8_255, Hmat, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return warped

def alpha_comp_over_white(rgba: np.ndarray) -> np.ndarray:
    """
    Composite a single RGBA image over white background -> RGB uint8.
    RGB is treated as already premultiplied by alpha -- which is what
    warp_rgba's cv2.warpPerspective(INTER_LINEAR) naturally produces, since it
    linearly interpolates from a (0,0,0,0) background straight to the
    (color,255) interior. Multiplying by alpha a second time here (the old
    behavior) double-darkens every partial-alpha boundary pixel, showing up as
    a gray contour hugging object edges.
    """
    premult_rgb = rgba[..., :3].astype(np.float32)
    a = (rgba[..., 3:4].astype(np.float32) / 255.0)
    white = 255.0
    out = premult_rgb + white * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)

def alpha_comp_over_bg(rgba: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    """
    Composite a single premultiplied-alpha RGBA image over an arbitrary RGB
    backdrop (e.g. a textured background from render_background_rgb) -> RGB
    uint8. Generalizes alpha_comp_over_white to a non-constant background --
    same premultiplied-over math, just with bg_rgb(H,W,3) in place of the
    constant white(H,W,3).
    """
    premult_rgb = rgba[..., :3].astype(np.float32)
    a = (rgba[..., 3:4].astype(np.float32) / 255.0)
    out = premult_rgb + bg_rgb.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)

def alpha_compose(rgba_bg: np.ndarray, rgba_fg: np.ndarray) -> np.ndarray:
    """
    Alpha composite fg over bg (both uint8 RGBA, RGB channels already
    premultiplied by alpha -- see alpha_comp_over_white). The premultiplied
    'over' operator needs no re-multiplication or re-normalization, and the
    output stays premultiplied so it can be composited again or passed
    straight to alpha_comp_over_white.
    """
    bg = rgba_bg.astype(np.float32)
    fg = rgba_fg.astype(np.float32)
    af = fg[..., 3:4] / 255.0

    out_rgb = fg[..., :3] + bg[..., :3] * (1.0 - af)
    out_a = fg[..., 3:4] + bg[..., 3:4] * (1.0 - af)

    out = np.concatenate([out_rgb, out_a], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# Slant-angle sampling config. Set once from CLI args in main() (see
# --slant-dist / --slant-magnitude-max / --slant-gamma-k / --slant-gamma-theta);
# read by sample_slant_magnitude() below. Kept as module globals (like
# OUTPUT_DIR) rather than threaded through every render_*/gen_* signature.
# ---------------------------------------------------------------------------
SLANT_DIST = "independent"     # "independent" | "uniform" | "gamma"
SLANT_MAGNITUDE_MAX = 1.0      # cap on the shared magnitude, in [0, 1]
SLANT_GAMMA_K = 2.0            # gamma shape: k>1 -> peak away from 0, then decays
SLANT_GAMMA_THETA = 0.25       # gamma scale: controls where the peak/tail sit
P_FRONTAL = 0.0                # probability an object is forced exactly front-parallel (m=0)

# ---------------------------------------------------------------------------
# Boundary-pixel disparity assignment. Set once from CLI (--boundary-mode /
# --area-supersample) in main(); read by render_slanted_one_object() and
# render_slanted_two_objects(). Only affects the slanted stereo-mode's
# disp_c/disp_l/seg_c on the cyclopean and left grids -- see
# layered_boundary_disparity() below for the three rules:
#   "hard"   (default) -- original behavior: rasterize each object's mask with
#            PIL/cv2 (floor-snapped, winner-take-all, no sub-pixel awareness)
#            and hard-copy whichever object's mask claims the pixel.
#   "center" -- exact point-in-polygon test at each pixel's true center
#            (x+0.5, y+0.5); still winner-take-all (no blending) but the
#            winner is decided by an exact geometric test instead of a
#            grid-snap rasterizer.
#   "area"   -- supersample each pixel into an AREA_SUPERSAMPLE x
#            AREA_SUPERSAMPLE grid of sample points, resolve occlusion at
#            that fine resolution, then set the pixel's disparity to the
#            coverage-fraction-weighted average of the (possibly two)
#            objects' local disparity planes.
# ---------------------------------------------------------------------------
BOUNDARY_MODE = "hard"          # "hard" | "center" | "area"
AREA_SUPERSAMPLE = 8            # samples per axis per pixel, used by "area"


def sample_slant_magnitude() -> float:
    """
    Draw the shared 'how slanted is this object' magnitude in [0, SLANT_MAGNITUDE_MAX].
    This single draw scales all five tilt params (s1,s2,p1,p2,ty) together, so an
    object that's very slanted has consistently large shear/perspective/vertical
    tilt rather than an uncorrelated mix.

    With probability P_FRONTAL, returns exactly 0.0 regardless of SLANT_DIST --
    magnitude 0 zeroes out s1,s2,p1,p2,ty in sample_slanted_homographies, giving a
    genuine front-parallel (unslanted) object. None of the continuous
    distributions below ever land on exactly 0 on their own, so without this,
    the dataset would contain no true front-parallel case.

    Otherwise:
    "independent": returns 1.0 -- s1,s2,p1,p2,ty are then each sampled independently
        at full base range in sample_slanted_homographies (the original behavior).
    "uniform": magnitude ~ Uniform(0, SLANT_MAGNITUDE_MAX).
    "gamma": magnitude ~ Gamma(SLANT_GAMMA_K, SLANT_GAMMA_THETA), truncated to
        SLANT_MAGNITUDE_MAX by rejection (resampling draws above the cap, rather
        than clamping them, so the cap doesn't create an artificial spike of
        maximally-slanted objects). With k=2 this is 0 at the origin, rises to a
        peak at (k-1)*theta, then decays with a long right tail -- most objects
        end up only slightly slanted, a few end up steeply slanted.
    """
    if random.random() < P_FRONTAL:
        return 0.0
    if SLANT_DIST == "independent":
        return 1.0
    elif SLANT_DIST == "uniform":
        return random.uniform(0.0, SLANT_MAGNITUDE_MAX)
    elif SLANT_DIST == "gamma":
        while True:
            mag = np.random.gamma(SLANT_GAMMA_K, SLANT_GAMMA_THETA)
            if mag <= SLANT_MAGNITUDE_MAX:
                return float(mag)
    else:
        raise ValueError(f"Unknown SLANT_DIST: {SLANT_DIST!r}")


def sample_slanted_homographies(W: int, H: int, d0: float,
                                max_shear_base: float = 0.25,
                                max_persp_base: float = 2e-3,
                                max_ty: float = 3.0,
                                slant_params: Optional[Tuple] = None,
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (H_C, H_L, H_R) for one object.
    If slant_params=(s1,s2,p1,p2,ty) is provided, use those instead of sampling —
    allows two objects to share identical slant geometry (same-depth case).
    """
    # p1/p2 multiply raw pixel x,y in the homogeneous denominator, and d0 (disparity, px)
    # scales with image size by convention (focal_px/clip_dmax doubled alongside W/H) --
    # both must be normalized by res_scale (per the REF_IMAGE_SIZE convention used
    # elsewhere for shape size) or the same nominal slant magnitude warps objects
    # increasingly far off-canvas as the canvas grows (this caused objects to render
    # fully off-frame at 512x512 that stayed on-frame at 256x256).
    res_scale = min(H, W) / REF_IMAGE_SIZE
    scale = min(1.0, max(0.25, abs(d0) / (40.0 * res_scale)))

    if slant_params is not None:
        s1, s2, p1, p2, ty = slant_params
    else:
        m = sample_slant_magnitude()
        max_shear = max_shear_base * scale * m
        max_persp = (max_persp_base / res_scale) * scale * m
        max_ty_eff = max_ty * m
        s1 = random.uniform(-max_shear, max_shear)
        s2 = random.uniform(-max_shear, max_shear)
        p1 = random.uniform(-max_persp, max_persp)
        p2 = random.uniform(-max_persp, max_persp)
        ty = random.uniform(-max_ty_eff, max_ty_eff)

    def makeH(tx: float) -> np.ndarray:
        return np.array([
            [1.0, s1, tx],
            [s2,  1.0, ty],
            [p1,  p2, 1.0]
        ], dtype=np.float32)

    half = 0.5 * float(d0)
    Hc = makeH(0.0)
    Hl = makeH(+half)
    Hr = makeH(-half)
    return Hc, Hl, Hr, (s1, s2, p1, p2, ty)

def disparity_from_composed_homographies_on_cyclopean(Hl: np.ndarray, Hr: np.ndarray, Hc: np.ndarray,
                                                     H: int, W: int) -> np.ndarray:
    """
    Compute disparity on the *tilted cyclopean grid*:
      x_cyc -> xL = (Hl @ inv(Hc)) x_cyc
      x_cyc -> xR = (Hr @ inv(Hc)) x_cyc
      d = xL - xR
    Returns (H,W) float32.
    """
    Hc_inv = np.linalg.inv(Hc.astype(np.float64))
    Hlc = (Hl.astype(np.float64) @ Hc_inv)
    Hrc = (Hr.astype(np.float64) @ Hc_inv)

    # grid in cyclopean coords
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    ones = np.ones_like(xs)
    P = np.stack([xs, ys, ones], axis=0).reshape(3, -1)  # (3, HW)

    PL = Hlc @ P
    PR = Hrc @ P

    xL = PL[0] / (PL[2] + 1e-12)
    xR = PR[0] / (PR[2] + 1e-12)

    d = (xL - xR).reshape(H, W).astype(np.float32)
    return d


def disparity_from_composed_homographies_on_left(Hl: np.ndarray, Hr: np.ndarray,
                                                 H: int, W: int) -> np.ndarray:
    """
    Compute disparity on the *left image grid* (what IGEV++ expects):
      For each pixel x_L in the left image:
        x_R = (Hr @ inv(Hl)) @ x_L
        d = x_L.x - x_R.x
    Returns (H,W) float32.
    """
    Hl_inv = np.linalg.inv(Hl.astype(np.float64))
    Hrl = Hr.astype(np.float64) @ Hl_inv  # maps left-image coords -> right-image coords

    xs, ys = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    ones = np.ones_like(xs)
    P = np.stack([xs, ys, ones], axis=0).reshape(3, -1)  # (3, HW)

    PR = Hrl @ P
    xR = PR[0] / (PR[2] + 1e-12)

    d = (xs.ravel() - xR).reshape(H, W).astype(np.float32)
    return d

# -----------------------------------------------------------------------------
# Boundary-pixel disparity assignment: exact point-in-polygon ("center" mode)
# and supersampled area-coverage ("area" mode), as alternatives to the
# floor-snapped PIL/cv2 rasterization used by BOUNDARY_MODE == "hard".
# -----------------------------------------------------------------------------
def transform_points_h(pts: List[Tuple[float, float]], Hmat: np.ndarray) -> List[Tuple[float, float]]:
    """Project polygon vertices through a 3x3 homography (canonical -> target grid).
    A homography maps straight edges to straight edges, so the transformed
    vertex list is exactly the warped polygon -- no raster mask needed."""
    P = np.array([[x, y, 1.0] for x, y in pts], dtype=np.float64).T  # (3, n)
    Pw = Hmat.astype(np.float64) @ P
    xs = Pw[0] / (Pw[2] + 1e-12)
    ys = Pw[1] / (Pw[2] + 1e-12)
    return list(zip(xs.tolist(), ys.tolist()))

def polygon_point_test_mask(pts: List[Tuple[float, float]], H: int, W: int,
                             offset: float = 0.5) -> np.ndarray:
    """
    Exact point-in-polygon test (even-odd rule), vectorized over the full grid.
    offset=0.5 tests each cell's true center (x+0.5, y+0.5) -- unlike
    PIL's ImageDraw.polygon, which snaps edges to the integer grid regardless
    of where within a pixel the true boundary falls.
    """
    yy, xx = np.mgrid[0:H, 0:W]
    cx = xx.astype(np.float64) + offset
    cy = yy.astype(np.float64) + offset
    n = len(pts)
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    inside = np.zeros((H, W), dtype=bool)
    xj, yj = xs[-1], ys[-1]
    for i in range(n):
        xi, yi = xs[i], ys[i]
        denom = yj - yi
        if denom != 0.0:
            cond = (yi > cy) != (yj > cy)
            x_int = (xj - xi) * (cy - yi) / denom + xi
            inside ^= cond & (cx < x_int)
        xj, yj = xi, yi
    return inside

def layered_boundary_disparity(
    H: int, W: int,
    polys_back_to_front: List[List[Tuple[float, float]]],
    disp_planes_back_to_front: List[np.ndarray],
    supersample: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Composite a stack of already-warped polygons (back to front; later ones
    occlude earlier ones, same convention as the existing hard-mask code) and
    return a boundary-aware disparity map plus an integer label map.

    supersample=1   -> "center" mode: one sample per pixel, at its exact
                       center. Whichever polygon (topmost wins under
                       occlusion) contains that point owns the whole pixel --
                       still winner-take-all, no blending.
    supersample=N>1 -> "area" mode: each pixel is split into an N x N grid of
                       sample points, each resolved to the topmost polygon
                       containing it (same occlusion rule), then the pixel's
                       disparity is the coverage-fraction-weighted average of
                       each object's *local* disp plane value (background
                       samples contribute 0).

    disp_planes_back_to_front[k] must be a full (H,W) array giving that
    object's disparity plane evaluated at every pixel (as already computed by
    disparity_from_composed_homographies_on_*), not just inside its silhouette.
    """
    S = max(1, int(supersample))
    fine_h, fine_w = H * S, W * S
    label_fine = np.zeros((fine_h, fine_w), dtype=np.uint8)

    for k, pts in enumerate(polys_back_to_front, start=1):
        pts_fine = [(x * S, y * S) for (x, y) in pts]
        mask = polygon_point_test_mask(pts_fine, fine_h, fine_w, offset=0.5)
        label_fine[mask] = k  # later (nearer) objects overwrite -> occlusion

    n_layers = len(polys_back_to_front)
    label_blocks = label_fine.reshape(H, S, W, S)
    frac = np.empty((H, W, n_layers + 1), dtype=np.float32)
    for k in range(n_layers + 1):
        frac[..., k] = (label_blocks == k).sum(axis=(1, 3))
    frac /= float(S * S)

    disp_map = np.zeros((H, W), dtype=np.float32)
    for k, disp_plane in enumerate(disp_planes_back_to_front, start=1):
        disp_map += frac[..., k] * disp_plane

    label_map = np.argmax(frac, axis=-1).astype(np.uint8)
    return disp_map, label_map

# -----------------------------------------------------------------------------
# Shift-only renderer (your original)
# -----------------------------------------------------------------------------
def shift_polygon_x(pts: List[Tuple[float,float]], delta_x: float) -> List[Tuple[float,float]]:
    return [(x + delta_x, y) for (x, y) in pts]

def polygon_mask_bool(pts: List[Tuple[float,float]], image_size: Tuple[int,int]) -> np.ndarray:
    H, W = image_size
    mask_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon(pts, outline=1, fill=1)
    return np.array(mask_img, dtype=np.uint8).astype(bool)

def render_left_right_from_polygons_shift(
    H: int, W: int,
    poly_list: List[List[Tuple[float, float]]],
    color_list: List[Tuple[int,int,int]],
    disp_list: List[float],
    layer_list: Optional[List[Optional[Image.Image]]] = None
) -> Tuple[Image.Image, Image.Image, np.ndarray]:
    left_img  = Image.new("RGB", (W, H), color=(255, 255, 255))
    right_img = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw_L = ImageDraw.Draw(left_img)
    draw_R = ImageDraw.Draw(right_img)

    disp_map = np.zeros((H, W), dtype=np.float32)

    if layer_list is None:
        layer_list = [None] * len(poly_list)

    for pts, color, d, layer in zip(poly_list, color_list, disp_list, layer_list):
        mask = polygon_mask_bool(pts, (H, W))
        disp_map[mask] = d

        half = 0.5 * d

        if layer is None:
            pts_left  = shift_polygon_x(pts, +half)
            pts_right = shift_polygon_x(pts, -half)
            draw_L.polygon(pts_left,  fill=color)
            draw_R.polygon(pts_right, fill=color)
        else:
            layer = layer.convert("RGBA")
            layer_L = layer.transform((W, H), Image.AFFINE, (1, 0, -half, 0, 1, 0),
                                      resample=Image.NEAREST, fillcolor=(0,0,0,0))
            layer_R = layer.transform((W, H), Image.AFFINE, (1, 0, +half, 0, 1, 0),
                                      resample=Image.NEAREST, fillcolor=(0,0,0,0))
            left_img  = Image.alpha_composite(left_img.convert("RGBA"),  layer_L).convert("RGB")
            right_img = Image.alpha_composite(right_img.convert("RGBA"), layer_R).convert("RGB")

    return left_img, right_img, disp_map

# -----------------------------------------------------------------------------
# Slanted renderer (NEW): tilts cyclopean + left + right, and computes GT in tilted cyclopean coords
# -----------------------------------------------------------------------------
def render_slanted_one_object(
    H: int, W: int,
    pts,
    color: Tuple[int,int,int],
    d0: float,
    layer_pil: Optional[Image.Image] = None,
    texture_spec: Optional[Dict[str, Any]] = None,
    bg_rgb: Optional[np.ndarray] = None,
) -> Tuple[Image.Image, Image.Image, Image.Image, np.ndarray, np.ndarray]:
    """
    Returns:
      img_cyc_tilted (PIL RGB),
      img_left (PIL RGB),
      img_right (PIL RGB),
      seg_cyc (H,W uint8 in {0,1}),
      disp_cyc (H,W float32 disparity on tilted cyclopean grid),
      disp_left (H,W float32 disparity on left image grid),
      seg_left (H,W uint8 in {0,1}, segmentation on the left image grid)

    bg_rgb: optional (H,W,3) uint8 backdrop (see render_background_rgb) used
    in place of solid white. Treated as an infinitely-far/zero-disparity
    backdrop, so the SAME array is composited behind all three views
    (cyclopean, left, right) with no shift -- physically correct for a
    background at infinity, and far simpler than warping a third layer.
    """
    if texture_spec is not None:
        rgba0 = textured_polygon_layer_rgba(H, W, pts, texture_spec)
    elif layer_pil is None:
        rgba0 = solid_polygon_layer_rgba(H, W, pts, color)
    else:
        rgba0 = pil_rgba_to_np(layer_pil)

    mask0 = polygon_mask_uint8(pts, (H, W))  # 0/255 uint8

    Hc, Hl, Hr, _ = sample_slanted_homographies(W, H, d0)

    if texture_spec is not None:
        rgba_c = warp_textured_polygon_rgba(H, W, pts, texture_spec, Hc, (H, W))
        rgba_l = warp_textured_polygon_rgba(H, W, pts, texture_spec, Hl, (H, W))
        rgba_r = warp_textured_polygon_rgba(H, W, pts, texture_spec, Hr, (H, W))
    else:
        rgba_c = warp_rgba(rgba0, Hc, (H, W), interp=cv2.INTER_LINEAR)
        rgba_l = warp_rgba(rgba0, Hl, (H, W), interp=cv2.INTER_LINEAR)
        rgba_r = warp_rgba(rgba0, Hr, (H, W), interp=cv2.INTER_LINEAR)

    # cyclopean disparity plane (evaluated everywhere, masked below)
    disp_plane = disparity_from_composed_homographies_on_cyclopean(Hl, Hr, Hc, H, W)
    disp_plane_left = disparity_from_composed_homographies_on_left(Hl, Hr, H, W)

    if BOUNDARY_MODE == "hard":
        # cyclopean seg (warp mask with Hc)
        mask_c = warp_mask(mask0, Hc, (H, W))
        seg_c = (mask_c > 0).astype(np.uint8)  # {0,1}
        disp_c = np.zeros((H, W), dtype=np.float32)
        disp_c[seg_c == 1] = disp_plane[seg_c == 1]

        # left-image disparity (for IGEV++ training)
        mask_l = warp_mask(mask0, Hl, (H, W)) > 0
        disp_l = np.zeros((H, W), dtype=np.float32)
        disp_l[mask_l] = disp_plane_left[mask_l]
        seg_l = mask_l.astype(np.uint8)  # {0,1}
    else:
        S = 1 if BOUNDARY_MODE == "center" else AREA_SUPERSAMPLE
        pts_c = transform_points_h(pts, Hc)
        pts_l = transform_points_h(pts, Hl)
        disp_c, seg_c = layered_boundary_disparity(H, W, [pts_c], [disp_plane], supersample=S)
        disp_l, seg_l = layered_boundary_disparity(H, W, [pts_l], [disp_plane_left], supersample=S)

    if bg_rgb is not None:
        img_cyc = Image.fromarray(alpha_comp_over_bg(rgba_c, bg_rgb), mode="RGB")
        img_l   = Image.fromarray(alpha_comp_over_bg(rgba_l, bg_rgb), mode="RGB")
        img_r   = Image.fromarray(alpha_comp_over_bg(rgba_r, bg_rgb), mode="RGB")
    else:
        img_cyc = Image.fromarray(alpha_comp_over_white(rgba_c), mode="RGB")
        img_l   = Image.fromarray(alpha_comp_over_white(rgba_l), mode="RGB")
        img_r   = Image.fromarray(alpha_comp_over_white(rgba_r), mode="RGB")
    return img_cyc, img_l, img_r, seg_c, disp_c, disp_l, seg_l

# Left-view segmentation with the OPPOSITE occlusion order (obj1 in front instead of obj2), computed from
# the SAME homographies as the main render. Set by render_slanted_two_objects(); read by gen_two_objects()
# to write fields_left/{stem}_field_B.npy (the second figure-ground reading in the left view).
_LAST_SEG_LEFT_ALT = None


def render_slanted_two_objects(
    H: int, W: int,
    pts_back, pts_front,
    color_back: Tuple[int,int,int],
    color_front: Tuple[int,int,int],
    d_back: float,
    d_front: float,
    layer_back_pil: Optional[Image.Image] = None,
    layer_front_pil: Optional[Image.Image] = None,
    force_camouflage_union: bool = False,
    texture_back_spec: Optional[Dict[str, Any]] = None,
    texture_front_spec: Optional[Dict[str, Any]] = None,
    bg_rgb: Optional[np.ndarray] = None,
    slant_params: Optional[Tuple] = None,
    share_slant: bool = False,
) -> Tuple[Image.Image, Image.Image, Image.Image, np.ndarray, np.ndarray, bool, bool, np.ndarray, np.ndarray]:
    """
    Returns:
      img_cyc_tilted, img_left, img_right, seg_cyc {0,1,2}, disp_cyc float32,
      disp_left float32, seg_left {0,1,2} (segmentation on the left image grid,
      occlusion-composited: front overwrites back, same convention as seg_cyc),
      back_visible_left, back_visible_right,
      back_mc (bool H×W), front_mc (bool H×W),
      back_vis_frac_left (float in [0,1]), back_vis_frac_right (float in [0,1])

    bg_rgb: optional (H,W,3) uint8 backdrop (see render_background_rgb), used
    identically behind all three views (zero-disparity/infinite-depth
    backdrop convention -- see render_slanted_one_object).
    Each object gets its own slant (piecewise planar).
    Occlusion in cyclopean is enforced by compositing back then front for BOTH RGB and seg.
    """
    # canonical rgba layers
    if texture_back_spec is not None:
        rgba_back0 = textured_polygon_layer_rgba(H, W, pts_back, texture_back_spec)
    elif layer_back_pil is None:
        rgba_back0 = solid_polygon_layer_rgba(H, W, pts_back, color_back)
    else:
        rgba_back0 = pil_rgba_to_np(layer_back_pil)
    if texture_front_spec is not None:
        rgba_front0 = textured_polygon_layer_rgba(H, W, pts_front, texture_front_spec)
    elif layer_front_pil is None:
        rgba_front0 = solid_polygon_layer_rgba(H, W, pts_front, color_front)
    else:
        rgba_front0 = pil_rgba_to_np(layer_front_pil)

    mask_back0  = polygon_mask_uint8(pts_back,  (H, W))
    mask_front0 = polygon_mask_uint8(pts_front, (H, W))

    # homographies per object: (Hc, Hl, Hr)
    # When both objects share the same disparity, also share slant params so
    # their per-pixel disparity maps are identical (same surface orientation).
    # slant_params: reuse a given slant (so a sweep re-renders the SAME scene geometry at different disparities);
    # share_slant: give both objects the same slant even when their disparities differ (a pure disparity offset).
    Hc_b, Hl_b, Hr_b, shared_params = sample_slanted_homographies(W, H, d_back, slant_params=slant_params)
    same_disp = abs(d_front - d_back) < 1e-6
    Hc_f, Hl_f, Hr_f, _ = sample_slanted_homographies(
        W, H, d_front, slant_params=shared_params if (same_disp or share_slant) else None
    )

    # Warp RGB layers
    if texture_back_spec is not None:
        back_c = warp_textured_polygon_rgba(H, W, pts_back, texture_back_spec, Hc_b, (H, W))
        back_l = warp_textured_polygon_rgba(H, W, pts_back, texture_back_spec, Hl_b, (H, W))
        back_r = warp_textured_polygon_rgba(H, W, pts_back, texture_back_spec, Hr_b, (H, W))
    else:
        back_c = warp_rgba(rgba_back0,  Hc_b, (H, W), interp=cv2.INTER_LINEAR)
        back_l = warp_rgba(rgba_back0,  Hl_b, (H, W), interp=cv2.INTER_LINEAR)
        back_r = warp_rgba(rgba_back0,  Hr_b, (H, W), interp=cv2.INTER_LINEAR)

    if texture_front_spec is not None:
        front_c = warp_textured_polygon_rgba(H, W, pts_front, texture_front_spec, Hc_f, (H, W))
        front_l = warp_textured_polygon_rgba(H, W, pts_front, texture_front_spec, Hl_f, (H, W))
        front_r = warp_textured_polygon_rgba(H, W, pts_front, texture_front_spec, Hr_f, (H, W))
    else:
        front_c = warp_rgba(rgba_front0, Hc_f, (H, W), interp=cv2.INTER_LINEAR)
        front_l = warp_rgba(rgba_front0, Hl_f, (H, W), interp=cv2.INTER_LINEAR)
        front_r = warp_rgba(rgba_front0, Hr_f, (H, W), interp=cv2.INTER_LINEAR)

    camouflage_has_texture = (texture_back_spec is not None) and (texture_front_spec is not None)

    # Composite RGBA (back then front), or union for same-color camouflage
    if force_camouflage_union and not camouflage_has_texture:
        # Remove any internal seam by treating both layers as a single same-color silhouette.
        cyc_a = np.maximum(back_c[..., 3], front_c[..., 3])[..., None]
        left_a = np.maximum(back_l[..., 3], front_l[..., 3])[..., None]
        right_a = np.maximum(back_r[..., 3], front_r[..., 3])[..., None]

        cyc_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        left_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        right_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        cyc_rgb[...] = np.array(color_front, dtype=np.uint8)
        left_rgb[...] = np.array(color_front, dtype=np.uint8)
        right_rgb[...] = np.array(color_front, dtype=np.uint8)

        cyc_rgba = np.concatenate([cyc_rgb, cyc_a], axis=-1).astype(np.uint8)
        left_rgba = np.concatenate([left_rgb, left_a], axis=-1).astype(np.uint8)
        right_rgba = np.concatenate([right_rgb, right_a], axis=-1).astype(np.uint8)
    else:
        cyc_rgba  = alpha_compose(back_c,  front_c)
        left_rgba = alpha_compose(back_l,  front_l)
        right_rgba= alpha_compose(back_r,  front_r)

    # Warp masks into cyclopean and compose seg with occlusion: front overwrites back
    back_mc  = warp_mask(mask_back0,  Hc_b, (H, W)) > 0
    front_mc = warp_mask(mask_front0, Hc_f, (H, W)) > 0
    back_ml  = warp_mask(mask_back0,  Hl_b, (H, W)) > 0
    front_ml = warp_mask(mask_front0, Hl_f, (H, W)) > 0
    back_mr  = warp_mask(mask_back0,  Hr_b, (H, W)) > 0
    front_mr = warp_mask(mask_front0, Hr_f, (H, W)) > 0

    seg_c = np.zeros((H, W), dtype=np.uint8)
    seg_c[back_mc] = 1
    seg_c[front_mc] = 2  # overwrite

    back_visible_left = bool(np.any(back_ml & (~front_ml)))
    back_visible_right = bool(np.any(back_mr & (~front_mr)))

    # Fraction of the back object that is unoccluded (observable) in each eye,
    # relative to the back object's in-frame warped area in that eye. Used to
    # control the "unpaired" case: near-0 in one eye (fully occluded), large in
    # the other (substantial observable portion). See gen_two_objects().
    back_area_left  = int(np.count_nonzero(back_ml))
    back_area_right = int(np.count_nonzero(back_mr))
    back_vis_frac_left  = (np.count_nonzero(back_ml & (~front_ml)) / back_area_left) if back_area_left > 0 else 0.0
    back_vis_frac_right = (np.count_nonzero(back_mr & (~front_mr)) / back_area_right) if back_area_right > 0 else 0.0

    # Disparity planes on tilted cyclopean grid (per object)
    disp_back_plane  = disparity_from_composed_homographies_on_cyclopean(Hl_b, Hr_b, Hc_b, H, W)
    disp_front_plane = disparity_from_composed_homographies_on_cyclopean(Hl_f, Hr_f, Hc_f, H, W)
    disp_back_left  = disparity_from_composed_homographies_on_left(Hl_b, Hr_b, H, W)
    disp_front_left = disparity_from_composed_homographies_on_left(Hl_f, Hr_f, H, W)

    if BOUNDARY_MODE == "hard":
        disp_c = np.zeros((H, W), dtype=np.float32)
        disp_c[seg_c == 1] = disp_back_plane[seg_c == 1]
        disp_c[seg_c == 2] = disp_front_plane[seg_c == 2]

        # Left-image disparity (for IGEV++ training): front occludes back in left view
        disp_l = np.zeros((H, W), dtype=np.float32)
        disp_l[back_ml]  = disp_back_left[back_ml]
        disp_l[front_ml] = disp_front_left[front_ml]  # front overwrites back

        seg_l = np.zeros((H, W), dtype=np.uint8)
        seg_l[back_ml] = 1
        seg_l[front_ml] = 2  # overwrite, same occlusion convention as seg_c
        seg_l_alt = np.zeros((H, W), dtype=np.uint8)
        seg_l_alt[front_ml] = 2
        seg_l_alt[back_ml] = 1  # opposite order: back overwrites front
    else:
        S = 1 if BOUNDARY_MODE == "center" else AREA_SUPERSAMPLE
        pts_back_c  = transform_points_h(pts_back,  Hc_b)
        pts_front_c = transform_points_h(pts_front, Hc_f)
        disp_c, seg_c = layered_boundary_disparity(
            H, W, [pts_back_c, pts_front_c], [disp_back_plane, disp_front_plane], supersample=S
        )

        pts_back_l  = transform_points_h(pts_back,  Hl_b)
        pts_front_l = transform_points_h(pts_front, Hl_f)
        disp_l, seg_l = layered_boundary_disparity(
            H, W, [pts_back_l, pts_front_l], [disp_back_left, disp_front_left], supersample=S
        )
        _, seg_l_alt = layered_boundary_disparity(
            H, W, [pts_front_l, pts_back_l], [disp_front_left, disp_back_left], supersample=S
        )

    if bg_rgb is not None:
        img_cyc = Image.fromarray(alpha_comp_over_bg(cyc_rgba, bg_rgb), mode="RGB")
        img_l   = Image.fromarray(alpha_comp_over_bg(left_rgba, bg_rgb), mode="RGB")
        img_r   = Image.fromarray(alpha_comp_over_bg(right_rgba, bg_rgb), mode="RGB")
    else:
        img_cyc = Image.fromarray(alpha_comp_over_white(cyc_rgba), mode="RGB")
        img_l   = Image.fromarray(alpha_comp_over_white(left_rgba), mode="RGB")
        img_r   = Image.fromarray(alpha_comp_over_white(right_rgba), mode="RGB")
    global _LAST_SEG_LEFT_ALT
    _LAST_SEG_LEFT_ALT = seg_l_alt
    return (img_cyc, img_l, img_r, seg_c, disp_c, disp_l, seg_l,
            back_visible_left, back_visible_right, back_mc, front_mc,
            back_vis_frac_left, back_vis_frac_right)

# -----------------------------------------------------------------------------
# Generation routines
# -----------------------------------------------------------------------------
def gen_one_object(i, H, W, out_root, dmax_clip, save_viz=True,
                   focal_px=300.0, baseline=0.1, z_min=1.0, z_max=3.0, p_zero_disp=0.0,
                   stereo_mode="shift", mode_texture='mix', p_texture=0.3,
                   tex_paths: Optional[List[str]] = None,
                   shapes: Optional[List[str]] = None,
                   bg_texture_mode: str = "none", p_bg_texture: float = 1.0,
                   bg_tex_paths: Optional[List[str]] = None):
    if os.path.exists(os.path.join(out_root, LEFT_DIR, f"one_object_{i}_left.png")):
        return  # already generated, skip
    pts = random_shape(H, W, shapes=shapes)
    color = tuple(random.randint(50, 200) for _ in range(3))

    use_texture = _use_texture_for_object(mode_texture, p_texture)
    layer_pil = None
    texture_spec = None
    if use_texture:
        tex_array = load_random_texture(tex_paths) if tex_paths else None
        texture_spec = make_texture_spec(pts, color, image_hw=(H, W), tex_array=tex_array)
        layer_pil = Image.fromarray(textured_polygon_layer_rgba(H, W, pts, texture_spec), mode="RGBA")

    # optional textured backdrop, guaranteed distinct from the object's own texture/color
    bg_rgb = None
    if stereo_mode == "slanted" and _use_texture_for_object(bg_texture_mode, p_bg_texture):
        bg_pool = bg_tex_paths if bg_tex_paths is not None else tex_paths
        exclude_arrays = [texture_spec["tex_rgb"]] if (texture_spec is not None and bg_pool is tex_paths) else None
        bg_spec = make_background_texture_spec(
            H, W, tex_paths=bg_pool, exclude_arrays=exclude_arrays, exclude_colors=[color]
        )
        bg_rgb = render_background_rgb(H, W, bg_spec)

    # disparity for this object
    d0 = sample_depth_to_disparity(focal_px, baseline, z_min, z_max)
    if random.random() < p_zero_disp:
        d0 = 0.0

    if stereo_mode == "slanted":
        img_cyc, img_l, img_r, seg_c, disp_c, disp_l, seg_l = render_slanted_one_object(
            H, W, pts, color, d0, layer_pil=layer_pil, texture_spec=texture_spec, bg_rgb=bg_rgb
        )
        segmap = seg_c
        disp_map = disp_c
        segmap_left = seg_l
    else:
        # original cyclopean
        img_cyc = Image.new("RGB", (W, H), (255, 255, 255))
        if layer_pil is None:
            d = ImageDraw.Draw(img_cyc)
            d.polygon(pts, fill=color)
            layer_list = [None]
        else:
            img_cyc = Image.alpha_composite(img_cyc.convert("RGBA"), layer_pil).convert("RGB")
            layer_list = [layer_pil]

        # original seg (unwarped)
        segmap = (polygon_mask_uint8(pts, (H, W)) > 0).astype(np.uint8)

        # original stereo shift-only
        img_l, img_r, disp_map = render_left_right_from_polygons_shift(
            H, W, [pts], [color], [d0], layer_list=layer_list
        )

        # left-image disparity for shift mode: same value d0, mask at shifted polygon
        disp_l = np.zeros((H, W), dtype=np.float32)
        mask_left = polygon_mask_bool(shift_polygon_x(pts, +0.5 * d0), (H, W))
        disp_l[mask_left] = d0
        segmap_left = mask_left.astype(np.uint8)

    # paths
    stem       = f"one_object_{i}"
    img_path   = os.path.join(out_root, IMG_DIR,   f"{stem}.png")
    seg_path   = os.path.join(out_root, SEG_DIR,   f"{stem}_seg.npy")
    field_path = os.path.join(out_root, FIELD_DIR, f"{stem}_field.npy")
    field_left_path = os.path.join(out_root, FIELD_LEFT_DIR, f"{stem}_field.npy")
    vis_path   = os.path.join(out_root, VIS_DIR,   f"{stem}_udf.png")

    img_cyc.save(img_path)
    np.save(seg_path, segmap.astype(np.uint8))
    build_and_save_udf_from_segmap(segmap, field_path, d_max=dmax_clip)
    build_and_save_udf_from_segmap(segmap_left, field_left_path, d_max=dmax_clip)

    if save_viz:
        D = np.load(field_path).squeeze(-1)
        save_udf_visualization(img_cyc, D, vis_path, d_max_for_viz=UDF_VIZ_DMAX, alpha=UDF_VIZ_ALPHA)

    img_l.save(os.path.join(out_root, LEFT_DIR,  f"{stem}_left.png"))
    img_r.save(os.path.join(out_root, RIGHT_DIR, f"{stem}_right.png"))
    np.save(os.path.join(out_root, DISP_DIR,      f"{stem}_disp.npy"), disp_map.astype(np.float32))
    np.save(os.path.join(out_root, DISP_LEFT_DIR, f"{stem}_disp.npy"), disp_l.astype(np.float32))

    # ── per-object amodal outputs ──────────────────────────────────────
    # obj1 = the single object; obj2 = zeros (no second object)
    mask_obj1 = (segmap > 0)
    zeros_mask = np.zeros((H, W), dtype=np.uint8)
    zeros_disp = np.zeros((H, W), dtype=np.float32)
    zeros_udf  = np.zeros((H, W, 1), dtype=np.float32)

    np.save(os.path.join(out_root, MASK_OBJ1_DIR, f"{stem}_mask.npy"), mask_obj1.astype(np.uint8))
    np.save(os.path.join(out_root, MASK_OBJ2_DIR, f"{stem}_mask.npy"), zeros_mask)

    np.save(os.path.join(out_root, DISP_OBJ1_DIR, f"{stem}_disp.npy"),
            (mask_obj1.astype(np.float32) * d0))
    np.save(os.path.join(out_root, DISP_OBJ2_DIR, f"{stem}_disp.npy"), zeros_disp)

    build_and_save_udf_from_mask(mask_obj1,
                                  os.path.join(out_root, UDF_OBJ1_DIR, f"{stem}_udf.npy"),
                                  d_max=dmax_clip)
    np.save(os.path.join(out_root, UDF_OBJ2_DIR, f"{stem}_udf.npy"), zeros_udf)

def gen_multi_objects(
    idx: int,
    n_obj: int,
    H: int,
    W: int,
    out_root: str,
    dmax_clip,
    save_viz: bool = True,
    focal_px: float = 300.0,
    baseline: float = 0.1,
    z_min: float = 0.5,
    z_max: float = 3.0,
    p_zero_disp: float = 0.0,
    stereo_mode: str = "slanted",
    mode_texture: str = "mix",
    p_texture: float = 0.3,
    tex_paths: Optional[List[str]] = None,
    shapes: Optional[List[str]] = None,
    same_color: bool = False,
    enforce_color_gap: bool = True,
    min_gap: float = 60.0,
):
    """
    Generate one N-object sample.

    Outputs:
      imgs/multi_{n_obj}obj_{idx}.png
      left/multi_{n_obj}obj_{idx}_left.png
      right/multi_{n_obj}obj_{idx}_right.png
      disp/multi_{n_obj}obj_{idx}_disp.npy          # cyclopean/grid disparity
      disp_left/multi_{n_obj}obj_{idx}_disp.npy     # left-view disparity
      SEG/multi_{n_obj}obj_{idx}_seg.npy
      fields/multi_{n_obj}obj_{idx}_field.npy

    Seg labels:
      0 = background
      1..n_obj = object labels.
    """
    assert n_obj >= 1, "n_obj must be >= 1"

    stem = f"multi_{n_obj}obj_{idx}"
    if os.path.exists(os.path.join(out_root, LEFT_DIR, f"{stem}_left.png")):
        return

    # ------------------------------------------------------------------
    # 1. Sample objects
    # ------------------------------------------------------------------
    poly_list = []
    color_list = []
    layer_list = []
    texture_specs = []

    if same_color:
        shared_color = tuple(random.randint(50, 200) for _ in range(3))

    for _ in range(n_obj):
        pts = random_shape(H, W, shapes=shapes)
        poly_list.append(pts)

        if same_color:
            color = shared_color
        else:
            color = tuple(random.randint(50, 200) for _ in range(3))
        color_list.append(color)

        use_texture = _use_texture_for_object(mode_texture, p_texture)
        texture_spec = None
        layer_pil = None

        if use_texture:
            tex_array = load_random_texture(tex_paths) if tex_paths else None
            texture_spec = make_texture_spec(
                pts,
                color,
                image_hw=(H, W),
                tex_array=tex_array,
            )
            layer_pil = Image.fromarray(
                textured_polygon_layer_rgba(H, W, pts, texture_spec),
                mode="RGBA",
            )

        texture_specs.append(texture_spec)
        layer_list.append(layer_pil)

    # ------------------------------------------------------------------
    # 2. Sample depths/disparities
    # ------------------------------------------------------------------
    # Larger Z = farther = smaller disparity.
    # Render order should be far -> near, so objects with larger Z are drawn first.
    z_vals = np.random.uniform(
        low=max(1e-3, z_min),
        high=max(z_min + 1e-3, z_max),
        size=n_obj,
    ).astype(np.float32)

    disp_vals = (float(focal_px * baseline) / z_vals).astype(np.float32)

    if random.random() < p_zero_disp:
        disp_vals[:] = 0.0

    # far-to-near order: large z first, small z last
    order = list(np.argsort(-z_vals))

    # Reorder everything consistently
    poly_list = [poly_list[i] for i in order]
    color_list = [color_list[i] for i in order]
    layer_list = [layer_list[i] for i in order]
    texture_specs = [texture_specs[i] for i in order]
    disp_vals = [float(disp_vals[i]) for i in order]

    # ------------------------------------------------------------------
    # 3. Render and build GT
    # ------------------------------------------------------------------
    if stereo_mode == "slanted":
        # Start with transparent layers, then alpha-compose far -> near.
        cyc_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        left_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        right_rgba = np.zeros((H, W, 4), dtype=np.uint8)

        segmap = np.zeros((H, W), dtype=np.uint8)
        disp_map = np.zeros((H, W), dtype=np.float32)
        disp_l = np.zeros((H, W), dtype=np.float32)

        for obj_rank, (pts, color, d0, layer_pil, texture_spec) in enumerate(
            zip(poly_list, color_list, disp_vals, layer_list, texture_specs),
            start=1,
        ):
            if texture_spec is not None:
                rgba0 = textured_polygon_layer_rgba(H, W, pts, texture_spec)
            elif layer_pil is None:
                rgba0 = solid_polygon_layer_rgba(H, W, pts, color)
            else:
                rgba0 = pil_rgba_to_np(layer_pil)

            mask0 = polygon_mask_uint8(pts, (H, W))

            Hc, Hl, Hr, _ = sample_slanted_homographies(W, H, d0)

            if texture_spec is not None:
                rgba_c = warp_textured_polygon_rgba(H, W, pts, texture_spec, Hc, (H, W))
                rgba_l = warp_textured_polygon_rgba(H, W, pts, texture_spec, Hl, (H, W))
                rgba_r = warp_textured_polygon_rgba(H, W, pts, texture_spec, Hr, (H, W))
            else:
                rgba_c = warp_rgba(rgba0, Hc, (H, W), interp=cv2.INTER_LINEAR)
                rgba_l = warp_rgba(rgba0, Hl, (H, W), interp=cv2.INTER_LINEAR)
                rgba_r = warp_rgba(rgba0, Hr, (H, W), interp=cv2.INTER_LINEAR)

            # Compose RGB far -> near.
            cyc_rgba = alpha_compose(cyc_rgba, rgba_c)
            left_rgba = alpha_compose(left_rgba, rgba_l)
            right_rgba = alpha_compose(right_rgba, rgba_r)

            # Masks in cyclopean and left grids.
            mask_c = warp_mask(mask0, Hc, (H, W)) > 0
            mask_l = warp_mask(mask0, Hl, (H, W)) > 0

            # Because we iterate far -> near, later/near objects overwrite.
            segmap[mask_c] = obj_rank

            disp_plane_c = disparity_from_composed_homographies_on_cyclopean(
                Hl, Hr, Hc, H, W
            )
            disp_map[mask_c] = disp_plane_c[mask_c]

            disp_plane_l = disparity_from_composed_homographies_on_left(
                Hl, Hr, H, W
            )
            disp_l[mask_l] = disp_plane_l[mask_l]

        img_cyc = Image.fromarray(alpha_comp_over_white(cyc_rgba), mode="RGB")
        img_l = Image.fromarray(alpha_comp_over_white(left_rgba), mode="RGB")
        img_r = Image.fromarray(alpha_comp_over_white(right_rgba), mode="RGB")

    else:
        # Shift-only mode: draw polygons far -> near.
        img_cyc = Image.new("RGB", (W, H), (255, 255, 255))
        img_l = Image.new("RGB", (W, H), (255, 255, 255))
        img_r = Image.new("RGB", (W, H), (255, 255, 255))

        segmap = np.zeros((H, W), dtype=np.uint8)
        disp_map = np.zeros((H, W), dtype=np.float32)
        disp_l = np.zeros((H, W), dtype=np.float32)

        for obj_rank, (pts, color, d0, layer_pil) in enumerate(
            zip(poly_list, color_list, disp_vals, layer_list),
            start=1,
        ):
            mask_c = polygon_mask_uint8(pts, (H, W)) > 0
            segmap[mask_c] = obj_rank
            disp_map[mask_c] = d0

            half = 0.5 * d0
            pts_l = shift_polygon_x(pts, +half)
            pts_r = shift_polygon_x(pts, -half)

            mask_l = polygon_mask_uint8(pts_l, (H, W)) > 0
            disp_l[mask_l] = d0

            if layer_pil is None:
                draw_c = ImageDraw.Draw(img_cyc)
                draw_l = ImageDraw.Draw(img_l)
                draw_r = ImageDraw.Draw(img_r)

                draw_c.polygon(pts, fill=color)
                draw_l.polygon(pts_l, fill=color)
                draw_r.polygon(pts_r, fill=color)
            else:
                layer = layer_pil.convert("RGBA")

                img_cyc = Image.alpha_composite(
                    img_cyc.convert("RGBA"),
                    layer,
                ).convert("RGB")

                layer_L = layer.transform(
                    (W, H),
                    Image.AFFINE,
                    (1, 0, -half, 0, 1, 0),
                    resample=Image.NEAREST,
                    fillcolor=(0, 0, 0, 0),
                )
                layer_R = layer.transform(
                    (W, H),
                    Image.AFFINE,
                    (1, 0, +half, 0, 1, 0),
                    resample=Image.NEAREST,
                    fillcolor=(0, 0, 0, 0),
                )

                img_l = Image.alpha_composite(
                    img_l.convert("RGBA"),
                    layer_L,
                ).convert("RGB")
                img_r = Image.alpha_composite(
                    img_r.convert("RGBA"),
                    layer_R,
                ).convert("RGB")

    # ------------------------------------------------------------------
    # 4. Save outputs
    # ------------------------------------------------------------------
    img_cyc.save(os.path.join(out_root, IMG_DIR, f"{stem}.png"))
    img_l.save(os.path.join(out_root, LEFT_DIR, f"{stem}_left.png"))
    img_r.save(os.path.join(out_root, RIGHT_DIR, f"{stem}_right.png"))

    np.save(
        os.path.join(out_root, DISP_DIR, f"{stem}_disp.npy"),
        disp_map.astype(np.float32),
    )
    np.save(
        os.path.join(out_root, DISP_LEFT_DIR, f"{stem}_disp.npy"),
        disp_l.astype(np.float32),
    )

    seg_path = os.path.join(out_root, SEG_DIR, f"{stem}_seg.npy")
    field_path = os.path.join(out_root, FIELD_DIR, f"{stem}_field.npy")
    vis_path = os.path.join(out_root, VIS_DIR, f"{stem}_udf.png")

    np.save(seg_path, segmap.astype(np.uint8))
    build_and_save_udf_from_segmap(segmap, field_path, d_max=dmax_clip)

    if save_viz:
        D = np.load(field_path).squeeze(-1)
        save_udf_visualization(
            img_cyc,
            D,
            vis_path,
            d_max_for_viz=UDF_VIZ_DMAX,
            alpha=UDF_VIZ_ALPHA,
        )

def _save_two_object_result(
    out_dir: str, stem: str,
    img_cyc, img_l, img_r,
    segmap: np.ndarray, disp_map: np.ndarray, disp_l: np.ndarray,
    mask_obj1_cyc: np.ndarray, mask_obj2_cyc: np.ndarray,
    d_front: float, d_back: float,
    same_color: bool, same_disp_thresh: float,
    save_viz: bool, dmax_clip,
    has_texture: bool = False,
    segmap_left: Optional[np.ndarray] = None,
    allow_textured_bimodal: bool = False,
    segmap_left_alt: Optional[np.ndarray] = None,
):
    """Write all outputs for one two-object render into out_dir.

    segmap_left_alt: optional left-view segmentation with the opposite occlusion order; when given
    (same-colour/same-disparity camouflage scenes) it is saved as fields_left/{stem}_field_B.npy, the
    left-view counterpart of the cyclopean _field_B (fields_left/{stem}_field.npy stays reading A).

    allow_textured_bimodal: when True, a same-color/same-disp pair saves the
    bimodal _field_A/_field_B split even if has_texture -- used by
    gen_two_objects, where textured camouflage is no longer tie-broken to a
    single UDF (see gen_two_objects for the corresponding removal of the
    forced sub-pixel disparity epsilon). gen_comparison_pair leaves this at
    the default False: its CAMO_EQ_DIR condition still relies on its own
    epsilon (see gen_comparison_pair) to guarantee exactly one UDF under
    texture, per its docstring.

    segmap_left: optional (H,W) uint8 {0,1,2} segmentation on the LEFT image
    grid (occlusion-composited, front overwrites back -- see
    render_slanted_two_objects). When given, saves fields_left/{stem}_field.npy
    -- always single-valued even when the cyclopean side is bimodal (_field_A/
    _field_B), since the same-disp/same-color boundary-ownership ambiguity
    that motivates the A/B split is a cyclopean-grid construct (both objects
    contribute the shared boundary in the midpoint view); on the left grid the
    front object deterministically occludes the back one, so there is only
    one physically correct answer. None (default) skips fields_left entirely,
    for callers that don't have a left segmentation (e.g. gen_comparison_pair,
    which doesn't currently need this output).
    """
    img_cyc.save(os.path.join(out_dir, IMG_DIR,      f"{stem}.png"))
    img_l.save(  os.path.join(out_dir, LEFT_DIR,     f"{stem}_left.png"))
    img_r.save(  os.path.join(out_dir, RIGHT_DIR,    f"{stem}_right.png"))
    np.save(os.path.join(out_dir, DISP_DIR,      f"{stem}_disp.npy"), disp_map.astype(np.float32))
    np.save(os.path.join(out_dir, DISP_LEFT_DIR, f"{stem}_disp.npy"), disp_l.astype(np.float32))

    # ── per-object amodal outputs ──────────────────────────────────────
    # obj1 = back object, obj2 = front object (full polygon masks, occlusion-free)
    np.save(os.path.join(out_dir, MASK_OBJ1_DIR, f"{stem}_mask.npy"),
            mask_obj1_cyc.astype(np.uint8))
    np.save(os.path.join(out_dir, MASK_OBJ2_DIR, f"{stem}_mask.npy"),
            mask_obj2_cyc.astype(np.uint8))

    disp_obj1 = mask_obj1_cyc.astype(np.float32) * d_back
    disp_obj2 = mask_obj2_cyc.astype(np.float32) * d_front
    np.save(os.path.join(out_dir, DISP_OBJ1_DIR, f"{stem}_disp.npy"), disp_obj1)
    np.save(os.path.join(out_dir, DISP_OBJ2_DIR, f"{stem}_disp.npy"), disp_obj2)

    build_and_save_udf_from_mask(mask_obj1_cyc,
                                  os.path.join(out_dir, UDF_OBJ1_DIR, f"{stem}_udf.npy"),
                                  d_max=dmax_clip)
    build_and_save_udf_from_mask(mask_obj2_cyc,
                                  os.path.join(out_dir, UDF_OBJ2_DIR, f"{stem}_udf.npy"),
                                  d_max=dmax_clip)

    is_same_disp = ((abs(d_front - d_back) <= same_disp_thresh) and same_color
                     and (not has_texture or allow_textured_bimodal))
    if is_same_disp:
        seg_A = segmap
        seg_B = np.zeros_like(segmap)
        seg_B[mask_obj2_cyc] = 2
        seg_B[mask_obj1_cyc] = 1  # back overwrites overlap
        for suffix, seg in [("A", seg_A), ("B", seg_B)]:
            seg_path   = os.path.join(out_dir, SEG_DIR,   f"{stem}_seg_{suffix}.npy")
            field_path = os.path.join(out_dir, FIELD_DIR, f"{stem}_field_{suffix}.npy")
            vis_path   = os.path.join(out_dir, VIS_DIR,   f"{stem}_udf_{suffix}.png")
            np.save(seg_path, seg.astype(np.uint8))
            build_and_save_udf_from_segmap(seg, field_path, d_max=dmax_clip)
            if save_viz:
                D = np.load(field_path).squeeze(-1)
                save_udf_visualization(img_cyc, D, vis_path, d_max_for_viz=UDF_VIZ_DMAX, alpha=UDF_VIZ_ALPHA)
    else:
        seg_path   = os.path.join(out_dir, SEG_DIR,   f"{stem}_seg.npy")
        field_path = os.path.join(out_dir, FIELD_DIR, f"{stem}_field.npy")
        vis_path   = os.path.join(out_dir, VIS_DIR,   f"{stem}_udf.png")
        np.save(seg_path, segmap.astype(np.uint8))
        build_and_save_udf_from_segmap(segmap, field_path, d_max=dmax_clip)
        if save_viz:
            D = np.load(field_path).squeeze(-1)
            save_udf_visualization(img_cyc, D, vis_path, d_max_for_viz=UDF_VIZ_DMAX, alpha=UDF_VIZ_ALPHA)

    if segmap_left is not None:
        field_left_path = os.path.join(out_dir, FIELD_LEFT_DIR, f"{stem}_field.npy")
        build_and_save_udf_from_segmap(segmap_left, field_left_path, d_max=dmax_clip)
        if segmap_left_alt is not None and is_same_disp:
            build_and_save_udf_from_segmap(
                segmap_left_alt, os.path.join(out_dir, FIELD_LEFT_DIR, f"{stem}_field_B.npy"), d_max=dmax_clip)


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.count_nonzero(mask_a | mask_b)
    return float(np.count_nonzero(mask_a & mask_b) / union) if union else 0.0


def place_pair_for_target_iou(pts_back, pts_front, H, W, iou_lo, iou_hi):
    """Return a new pts_front whose amodal-mask IoU with pts_back lands in [iou_lo, iou_hi]
    (rasterised IoU, intersection/union), or None if this shape pair cannot reach it.

    Low targets (<0.5) keep the independently drawn front shape and slide it along a random
    direction (bisection on the offset). High targets derive the front shape from the back
    shape (scaled/rotated by an amount that shrinks as the target -> 1) so that a large IoU is
    reachable at all; target >= 0.999 returns an identical copy. The result must stay inside
    the image."""
    t = random.uniform(iou_lo, iou_hi)
    mask_b = polygon_mask_uint8(pts_back, (H, W)) > 0
    pts_b = np.asarray(pts_back, dtype=np.float64)

    if t >= 0.999:
        return [tuple(p) for p in pts_b]
    if t >= 0.5:
        cx, cy = pts_b.mean(0)
        s_ = 1.0 + (1.0 - t) * random.uniform(-1.0, 1.0) * 0.6
        a = (1.0 - t) * random.uniform(-1.0, 1.0) * 0.6
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        base = (pts_b - [cx, cy]) @ R.T * s_ + [cx, cy]
    else:
        base = np.asarray(pts_front, dtype=np.float64)
    c_b = pts_b.mean(0)
    c_f = base.mean(0)

    phi = random.uniform(0.0, 2.0 * np.pi)
    u = np.array([np.cos(phi), np.sin(phi)])

    def moved(r):
        # r = 0 puts the front centroid on the back centroid; r grows along direction u
        return base - c_f + c_b + r * u

    def iou_at(r):
        return _mask_iou(mask_b, polygon_mask_uint8([tuple(p) for p in moved(r)], (H, W)) > 0)

    if iou_at(0.0) < t:           # even perfectly co-centred, this pair cannot reach the target
        return None
    lo_r, hi_r = 0.0, float(max(H, W))
    for _ in range(30):
        mid = 0.5 * (lo_r + hi_r)
        if iou_at(mid) > t:
            lo_r = mid
        else:
            hi_r = mid
    cand = moved(0.5 * (lo_r + hi_r))
    if cand.min() < 0 or cand[:, 0].max() >= W or cand[:, 1].max() >= H:
        return None
    iou = iou_at(0.5 * (lo_r + hi_r))
    if not (iou_lo - 1e-6 <= iou <= iou_hi + 1e-6):
        return None
    return [tuple(p) for p in cand]


def gen_two_objects(
    idx, H, W, out_root, dmax_clip, save_viz=True,
    same_color=False, enforce_color_gap=True, min_gap=60,
    focal_px=300.0, baseline=0.1, z_min=1.0, z_max=3.0, p_zero_disp=0.0,
    stereo_mode="shift", mode_texture='mix', p_texture=0.3,
    tex_paths: Optional[List[str]] = None,
    same_disp_thresh: float = 1.0,
    shapes: Optional[List[str]] = None,
    p_same_depth: float = 0.0,
    require_unpaired: bool = False,
    min_overlap_ratio: float = 0.0,
    p_large_overlap: float = 0.0,
    unpaired_occluded_max: float = 0.05,
    unpaired_visible_min: float = 0.30,
    bg_texture_mode: str = "none",
    p_bg_texture: float = 1.0,
    bg_tex_paths: Optional[List[str]] = None,
    target_iou_range: Optional[Tuple[float, float]] = None,
):
    if os.path.exists(os.path.join(out_root, LEFT_DIR, f"two_object_{idx}_left.png")):
        return  # already generated, skip
    # Unpaired ("XOR-occlusion") sampling is a narrow geometric target, so give it
    # a much larger attempt budget than the ordinary two-object case.
    max_attempts = 4000 if require_unpaired else 200
    require_large_overlap = (p_large_overlap > 0.0 and min_overlap_ratio > 0.0
                             and random.random() < p_large_overlap)
    for _ in range(max_attempts):
        pts_back  = random_shape(H, W, shapes=shapes)
        pts_front = random_shape(H, W, shapes=shapes)

        # For the unpaired case the front object must be able to FULLY occlude the
        # back in one eye, which is only possible if it is at least as large. Bias
        # toward that by swapping so the larger silhouette is the front object.
        if require_unpaired:
            if np.count_nonzero(polygon_mask_uint8(pts_front, (H, W))) < \
               np.count_nonzero(polygon_mask_uint8(pts_back, (H, W))):
                pts_back, pts_front = pts_front, pts_back

        if target_iou_range is not None:
            pts_front = place_pair_for_target_iou(pts_back, pts_front, H, W, *target_iou_range)
            if pts_front is None:
                continue

        if require_large_overlap:
            mask_b = polygon_mask_uint8(pts_back,  (H, W)) > 0
            mask_f = polygon_mask_uint8(pts_front, (H, W)) > 0
            intersection  = np.count_nonzero(mask_b & mask_f)
            smaller_area  = min(np.count_nonzero(mask_b), np.count_nonzero(mask_f))
            if smaller_area == 0 or intersection / smaller_area < min_overlap_ratio:
                continue

        color_back, color_front = pick_two_colors(
            enforce_gap=enforce_color_gap and (not same_color),
            min_gap=min_gap,
            same_color=same_color
        )

        # optional textures
        layers_pil = [None, None]
        texture_specs = [None, None]
        if same_color:
            use_camouflage_texture = _use_texture_for_object(mode_texture, p_texture)
            use_texture_back = use_camouflage_texture
            use_texture_front = use_camouflage_texture
        else:
            use_texture_back  = _use_texture_for_object(mode_texture, p_texture)
            use_texture_front = _use_texture_for_object(mode_texture, p_texture)

        # For same-color camouflage with external textures, both objects share one
        # randomly chosen image so the pattern flows continuously across the boundary.
        shared_tex = None
        if tex_paths and same_color and (use_texture_back or use_texture_front):
            shared_tex = load_random_texture(tex_paths)

        camouflage_pattern_kind = None
        if same_color and (use_texture_back or use_texture_front) and not tex_paths:
            camouflage_pattern_kind = random.choice(["checker", "stripes", "dots"])

        if use_texture_back:
            tex_b = shared_tex if shared_tex is not None else (load_random_texture(tex_paths) if tex_paths else None)
            texture_specs[0] = make_texture_spec(
                pts_back,
                color_back,
                anchor_mode="global" if same_color else "object",
                image_hw=(H, W),
                pattern_kind=camouflage_pattern_kind,
                tex_array=tex_b,
            )
            layers_pil[0] = Image.fromarray(
                textured_polygon_layer_rgba(H, W, pts_back, texture_specs[0]),
                mode="RGBA",
            )

        if use_texture_front:
            # For camouflage: share all UV params from back spec so the texture
            # is continuous at the object boundary (same theta, repeat, phase, anchor).
            texture_specs[1] = make_texture_spec(
                pts_front,
                color_front,
                same_texture_field=texture_specs[0] if same_color else None,
                anchor_mode="global" if same_color else "object",
                image_hw=(H, W),
                pattern_kind=camouflage_pattern_kind,
                tex_array=None if same_color else (shared_tex if shared_tex is not None else (load_random_texture(tex_paths) if tex_paths else None)),
            )
            layers_pil[1] = Image.fromarray(
                textured_polygon_layer_rgba(H, W, pts_front, texture_specs[1]),
                mode="RGBA",
            )

        has_texture = texture_specs[0] is not None or texture_specs[1] is not None

        # depths/disparities
        if random.random() < p_same_depth:
            z = random.uniform(max(1e-3, z_min), max(z_min + 1e-3, z_max))
            d_front = float(focal_px * baseline / z)
            d_back  = d_front
        else:
            depth_gap = 1.5 if require_unpaired else 0.5
            z_front, z_back = sample_ordered_depths(z_min, z_max, gap=depth_gap)
            d_front = float(focal_px * baseline / z_front)
            d_back  = float(focal_px * baseline / z_back)

        if random.random() < p_zero_disp:
            d_back = 0.0
            d_front = 0.0

        if stereo_mode == "slanted":
            bg_rgb = None
            if _use_texture_for_object(bg_texture_mode, p_bg_texture):
                bg_pool = bg_tex_paths if bg_tex_paths is not None else tex_paths
                exclude_arrays = (
                    [t["tex_rgb"] for t in texture_specs if t is not None]
                    if bg_pool is tex_paths else None
                ) or None
                bg_spec = make_background_texture_spec(
                    H, W, tex_paths=bg_pool, exclude_arrays=exclude_arrays,
                    exclude_colors=[color_back, color_front],
                )
                bg_rgb = render_background_rgb(H, W, bg_spec)

            (img_cyc, img_l, img_r, segmap, disp_map, disp_l, segmap_left,
             back_visible_left, back_visible_right, mask_obj1_cyc, mask_obj2_cyc,
             back_vis_frac_left, back_vis_frac_right) = render_slanted_two_objects(
                H, W,
                pts_back, pts_front,
                color_back, color_front,
                d_back, d_front,
                layer_back_pil=layers_pil[0],
                layer_front_pil=layers_pil[1],
                force_camouflage_union=same_color,
                texture_back_spec=texture_specs[0],
                texture_front_spec=texture_specs[1],
                bg_rgb=bg_rgb,
            )
        else:
            # original cyclopean preview (unwarped) with occlusion order back->front
            img = Image.new("RGB", (W, H), color=(255, 255, 255))
            if layers_pil[0] is None:
                d0 = ImageDraw.Draw(img)
                d0.polygon(pts_back, fill=color_back)
            else:
                img = Image.alpha_composite(img.convert("RGBA"), layers_pil[0]).convert("RGB")

            if layers_pil[1] is None:
                d1 = ImageDraw.Draw(img)
                d1.polygon(pts_front, fill=color_front)
            else:
                img = Image.alpha_composite(img.convert("RGBA"), layers_pil[1]).convert("RGB")

            img_cyc = img

            # seg {0,1,2} in unwarped cyclopean
            segmap = np.zeros((H, W), dtype=np.uint8)
            mb = (polygon_mask_uint8(pts_back, (H, W)) > 0)
            mf = (polygon_mask_uint8(pts_front, (H, W)) > 0)
            segmap[mb] = 1
            segmap[mf] = 2
            mask_obj1_cyc, mask_obj2_cyc = mb, mf

            # original shift-only stereo + disparity on unwarped grid
            img_l, img_r, disp_map = render_left_right_from_polygons_shift(
                H, W,
                [pts_back, pts_front],
                [color_back, color_front],
                [d_back, d_front],
                layer_list=layers_pil
            )

            mb_l = polygon_mask_uint8(shift_polygon_x(pts_back, +0.5 * d_back), (H, W)) > 0
            mf_l = polygon_mask_uint8(shift_polygon_x(pts_front, +0.5 * d_front), (H, W)) > 0
            mb_r = polygon_mask_uint8(shift_polygon_x(pts_back, -0.5 * d_back), (H, W)) > 0
            mf_r = polygon_mask_uint8(shift_polygon_x(pts_front, -0.5 * d_front), (H, W)) > 0
            back_visible_left = bool(np.any(mb_l & (~mf_l)))
            back_visible_right = bool(np.any(mb_r & (~mf_r)))

            # Observable fraction of the back object per eye (see slanted branch).
            back_area_left  = int(np.count_nonzero(mb_l))
            back_area_right = int(np.count_nonzero(mb_r))
            back_vis_frac_left  = (np.count_nonzero(mb_l & (~mf_l)) / back_area_left) if back_area_left > 0 else 0.0
            back_vis_frac_right = (np.count_nonzero(mb_r & (~mf_r)) / back_area_right) if back_area_right > 0 else 0.0

            # left-image disparity for shift mode: front occludes back
            disp_l = np.zeros((H, W), dtype=np.float32)
            disp_l[mb_l] = d_back
            disp_l[mf_l] = d_front  # front overwrites back

            segmap_left = np.zeros((H, W), dtype=np.uint8)
            segmap_left[mb_l] = 1
            segmap_left[mf_l] = 2  # overwrite, same occlusion convention as segmap

        # Visibility filter
        if require_unpaired:
            # Controlled unpaired case: the back object must be (near-)fully
            # occluded in one eye and substantially observable in the other.
            #   occluded eye : back_vis_frac <= unpaired_occluded_max  (~0)
            #   visible  eye : back_vis_frac >= unpaired_visible_min   (large)
            lo = min(back_vis_frac_left, back_vis_frac_right)
            hi = max(back_vis_frac_left, back_vis_frac_right)
            if not (lo <= unpaired_occluded_max and hi >= unpaired_visible_min):
                continue
        elif target_iou_range is None:
            # (skipped in target-IoU mode: objects share colour+disparity there, and IoU=1 means
            # the back object is fully hidden by construction)
            if not back_visible_left and not back_visible_right:
                continue
        break
    else:
        raise RuntimeError("Failed to sample a valid two-object pair after many attempts.")

    _save_two_object_result(
        out_root, f"two_object_{idx}",
        img_cyc, img_l, img_r,
        segmap, disp_map, disp_l,
        mask_obj1_cyc, mask_obj2_cyc,
        d_front, d_back, same_color, same_disp_thresh, save_viz, dmax_clip,
        has_texture=has_texture,
        segmap_left=segmap_left,
        allow_textured_bimodal=True,
        segmap_left_alt=(_LAST_SEG_LEFT_ALT if stereo_mode == "slanted" else None),
    )

def gen_delta_d_sweep(base_id, n_total, deltas, H, W, out_root, dmax_clip, iou_range,
                      shapes=None, bg_texture_mode="all", p_bg_texture=1.0, bg_tex_paths=None,
                      d_range=(10.0, 50.0)):
    """One BASE scene (two same-colour untextured objects, controlled IoU, textured background) rendered at
    every disparity offset in `deltas`: d_front = d_back + delta, nearer object = obj2 (the renderer's front object,
    so reading A is the physically correct figure-ground for delta > 0). Shapes, colour, background and slant are
    IDENTICAL across levels (paired design), only the horizontal disparity offset changes. Both A/B UDF readings are
    always saved. File index = level_index * n_total + base_id (so a level is a contiguous index range)."""
    idx_of = lambda li: li * n_total + base_id
    if all(os.path.exists(os.path.join(out_root, LEFT_DIR, f"two_object_{idx_of(li)}_left.png")) for li in range(len(deltas))):
        return
    lo, hi = iou_range
    for _ in range(2000):
        pts_back = random_shape(H, W, shapes=shapes)
        pts_front = random_shape(H, W, shapes=shapes)
        pts_front = place_pair_for_target_iou(pts_back, pts_front, H, W, lo, hi)
        if pts_front is not None:
            break
    else:
        raise RuntimeError("could not place a shape pair in the requested IoU range")
    color_back, color_front = pick_two_colors(enforce_gap=False, same_color=True)
    d_back = random.uniform(*d_range)
    bg_rgb = None
    if _use_texture_for_object(bg_texture_mode, p_bg_texture):
        bg_spec = make_background_texture_spec(H, W, tex_paths=bg_tex_paths, exclude_arrays=None,
                                               exclude_colors=[color_back, color_front])
        bg_rgb = render_background_rgb(H, W, bg_spec)
    slant = sample_slanted_homographies(W, H, d_back)[3]
    rows = []
    for li, delta in enumerate(deltas):
        d_front = d_back + float(delta)
        (img_cyc, img_l, img_r, segmap, disp_map, disp_l, segmap_left,
         _bvl, _bvr, mask_obj1_cyc, mask_obj2_cyc, _vfl, _vfr) = render_slanted_two_objects(
            H, W, pts_back, pts_front, color_back, color_front, d_back, d_front,
            layer_back_pil=None, layer_front_pil=None, force_camouflage_union=True,
            texture_back_spec=None, texture_front_spec=None, bg_rgb=bg_rgb,
            slant_params=slant, share_slant=True)
        idx = idx_of(li)
        _save_two_object_result(
            out_root, f"two_object_{idx}", img_cyc, img_l, img_r, segmap, disp_map, disp_l,
            mask_obj1_cyc, mask_obj2_cyc, d_front, d_back, True, 1e9, False, dmax_clip,
            has_texture=False, segmap_left=segmap_left, allow_textured_bimodal=True,
            segmap_left_alt=_LAST_SEG_LEFT_ALT)
        union = np.count_nonzero(mask_obj1_cyc | mask_obj2_cyc)
        iou = float(np.count_nonzero(mask_obj1_cyc & mask_obj2_cyc) / union) if union else 0.0
        rows.append((idx, base_id, li, float(delta), d_back, d_front, iou))
    os.makedirs(os.path.join(out_root, "manifest"), exist_ok=True)
    with open(os.path.join(out_root, "manifest", f"base_{base_id}.csv"), "w") as f:
        f.write("idx,base_id,level_index,delta_d,d_back,d_front,iou\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


# -----------------------------------------------------------------------------
# Comparison-pair generation
# -----------------------------------------------------------------------------
def gen_comparison_pair(
    idx: int, H: int, W: int, out_root: str, dmax_clip,
    save_viz: bool = True,
    focal_px: float = 300.0, baseline: float = 0.1,
    z_min: float = 1.0, z_max: float = 3.0,
    stereo_mode: str = "slanted",
    mode_texture: str = "all", p_texture: float = 1.0,
    tex_paths: Optional[List] = None,
    same_disp_thresh: float = 1.0,
    shapes: Optional[List[str]] = None,
    depth_gap: float = 0.8,
):
    """
    Generate one paired sample across all four comparison conditions:
      camouflage_equal_disp / camouflage_nonequal_disp
      noncamo_equal_disp    / noncamo_nonequal_disp

    Within each pair (camo or noncamo) the shapes, colors, textures and slant
    homographies are IDENTICAL — only the disparity values differ.
    No back-visibility filter is applied.

    camouflage + equal disp produces two UDF versions (A/B) because the depth
    order is visually ambiguous; all other conditions produce one.
    """
    stem = f"sample_{idx:05d}"
    if os.path.exists(os.path.join(out_root, CAMO_EQ_DIR, LEFT_DIR, f"{stem}_left.png")):
        return

    # ── inner helpers ─────────────────────────────────────────────────────────

    def _build_scene(same_color: bool, pts_back, pts_front) -> dict:
        """Sample textures/colors for pre-sampled shapes; return dict with post-setup RNG state."""
        color_back, color_front = pick_two_colors(
            enforce_gap=not same_color, min_gap=60, same_color=same_color,
        )

        layers_pil    = [None, None]
        texture_specs = [None, None]

        if same_color:
            use_tex = _use_texture_for_object(mode_texture, p_texture)
            use_tex_back = use_tex_front = use_tex
        else:
            use_tex_back  = _use_texture_for_object(mode_texture, p_texture)
            use_tex_front = _use_texture_for_object(mode_texture, p_texture)

        shared_tex = None
        if tex_paths and same_color and (use_tex_back or use_tex_front):
            shared_tex = load_random_texture(tex_paths)

        camo_pattern_kind = None
        if same_color and (use_tex_back or use_tex_front) and not tex_paths:
            camo_pattern_kind = random.choice(["checker", "stripes", "dots"])

        if use_tex_back:
            tex_b = shared_tex if shared_tex is not None else (
                load_random_texture(tex_paths) if tex_paths else None)
            texture_specs[0] = make_texture_spec(
                pts_back, color_back,
                anchor_mode="global" if same_color else "object",
                image_hw=(H, W), pattern_kind=camo_pattern_kind, tex_array=tex_b,
            )
            layers_pil[0] = Image.fromarray(
                textured_polygon_layer_rgba(H, W, pts_back, texture_specs[0]), mode="RGBA",
            )

        if use_tex_front:
            texture_specs[1] = make_texture_spec(
                pts_front, color_front,
                same_texture_field=texture_specs[0] if same_color else None,
                anchor_mode="global" if same_color else "object",
                image_hw=(H, W), pattern_kind=camo_pattern_kind,
                tex_array=None if same_color else (
                    shared_tex if shared_tex is not None else (
                        load_random_texture(tex_paths) if tex_paths else None)),
            )
            layers_pil[1] = Image.fromarray(
                textured_polygon_layer_rgba(H, W, pts_front, texture_specs[1]), mode="RGBA",
            )

        return dict(
            pts_back=pts_back, pts_front=pts_front,
            color_back=color_back, color_front=color_front,
            layers_pil=layers_pil, texture_specs=texture_specs,
            rng_state=random.getstate(), np_state=np.random.get_state(),
            same_color=same_color,
        )

    def _render(scene: dict, d_back: float, d_front: float):
        """Restore RNG to post-scene-setup state and render."""
        random.setstate(scene["rng_state"])
        np.random.set_state(scene["np_state"])

        if stereo_mode == "slanted":
            img_cyc, img_l, img_r, segmap, disp_map, disp_l, _, _, _, mask1, mask2, _, _ = \
                render_slanted_two_objects(
                    H, W,
                    scene["pts_back"], scene["pts_front"],
                    scene["color_back"], scene["color_front"],
                    d_back, d_front,
                    layer_back_pil=scene["layers_pil"][0],
                    layer_front_pil=scene["layers_pil"][1],
                    force_camouflage_union=scene["same_color"],
                    texture_back_spec=scene["texture_specs"][0],
                    texture_front_spec=scene["texture_specs"][1],
                )
        else:  # shift
            img = Image.new("RGB", (W, H), color=(255, 255, 255))
            if scene["layers_pil"][0] is None:
                ImageDraw.Draw(img).polygon(scene["pts_back"], fill=scene["color_back"])
            else:
                img = Image.alpha_composite(img.convert("RGBA"), scene["layers_pil"][0]).convert("RGB")
            if scene["layers_pil"][1] is None:
                ImageDraw.Draw(img).polygon(scene["pts_front"], fill=scene["color_front"])
            else:
                img = Image.alpha_composite(img.convert("RGBA"), scene["layers_pil"][1]).convert("RGB")
            img_cyc = img

            segmap = np.zeros((H, W), dtype=np.uint8)
            mb = polygon_mask_uint8(scene["pts_back"],  (H, W)) > 0
            mf = polygon_mask_uint8(scene["pts_front"], (H, W)) > 0
            segmap[mb] = 1
            segmap[mf] = 2
            mask1, mask2 = mb, mf

            img_l, img_r, disp_map = render_left_right_from_polygons_shift(
                H, W,
                [scene["pts_back"],  scene["pts_front"]],
                [scene["color_back"], scene["color_front"]],
                [d_back, d_front],
                layer_list=scene["layers_pil"],
            )
            mb_l = polygon_mask_uint8(shift_polygon_x(scene["pts_back"],  +0.5*d_back),  (H, W)) > 0
            mf_l = polygon_mask_uint8(shift_polygon_x(scene["pts_front"], +0.5*d_front), (H, W)) > 0
            disp_l = np.zeros((H, W), dtype=np.float32)
            disp_l[mb_l] = d_back
            disp_l[mf_l] = d_front

        return img_cyc, img_l, img_r, segmap, disp_map, disp_l, mask1, mask2

    # ── shared geometry + depth values ────────────────────────────────────────
    pts_back  = random_shape(H, W, shapes=shapes)
    pts_front = random_shape(H, W, shapes=shapes)

    z_eq = random.uniform(max(1e-3, z_min), max(z_min + 1e-3, z_max))
    d_eq = float(focal_px * baseline / z_eq)
    z_f, z_b = sample_ordered_depths(z_min, z_max, gap=depth_gap)
    d_f = float(focal_px * baseline / z_f)
    d_b = float(focal_px * baseline / z_b)

    # ── camouflage pair ───────────────────────────────────────────────────────
    camo = _build_scene(same_color=True, pts_back=pts_back, pts_front=pts_front)

    camo_has_texture = (camo["texture_specs"][0] is not None or camo["texture_specs"][1] is not None)
    # Textured camouflage can't be at equal disp: give the front object a sub-pixel
    # epsilon so there is a canonical depth ordering and only one UDF is saved.
    d_eq_front = d_eq + random.uniform(0.1, 0.5) if camo_has_texture else d_eq

    _save_two_object_result(
        os.path.join(out_root, CAMO_EQ_DIR), stem,
        *_render(camo, d_eq, d_eq_front),
        d_eq_front, d_eq, True, same_disp_thresh, save_viz, dmax_clip,
        has_texture=camo_has_texture,
    )
    _save_two_object_result(
        os.path.join(out_root, CAMO_NEQ_DIR), stem,
        *_render(camo, d_b, d_f),
        d_f, d_b, True, same_disp_thresh, save_viz, dmax_clip,
        has_texture=camo_has_texture,
    )

    # ── non-camouflage pair ───────────────────────────────────────────────────
    noncamo = _build_scene(same_color=False, pts_back=pts_back, pts_front=pts_front)

    # Always add a small epsilon so the two objects are never at exactly the same
    # disparity — exact equality is unrealistic and can't be rendered at sub-pixel
    # precision in pixel-rasterised images.
    _eq_eps = random.uniform(0.05, min(0.3, same_disp_thresh * 0.5))
    _d_eq_f = max(0.0, d_eq + random.choice([-1, 1]) * _eq_eps)
    _save_two_object_result(
        os.path.join(out_root, NONCAMO_EQ_DIR), stem,
        *_render(noncamo, d_eq, _d_eq_f),
        _d_eq_f, d_eq, False, same_disp_thresh, save_viz, dmax_clip,
    )
    _save_two_object_result(
        os.path.join(out_root, NONCAMO_NEQ_DIR), stem,
        *_render(noncamo, d_b, d_f),
        d_f, d_b, False, same_disp_thresh, save_viz, dmax_clip,
    )


# -----------------------------------------------------------------------------
# Folded-paper stimulus
# -----------------------------------------------------------------------------
def gen_folded_paper(
    idx: int,
    H: int,
    W: int,
    out_root: str,
    dmax_clip,
    save_viz: bool = True,
    focal_px: float = 300.0,
    baseline: float = 0.1,
    z_min: float = 0.5,
    z_max: float = 3.0,
    stereo_mode: str = "slanted",          # accepted for CLI compat; ignored.
    mode_texture: str = "none",
    p_texture: float = 0.0,
    tex_paths: Optional[List] = None,
    disp_polarity: str = "center_near",    # accepted for CLI compat; ignored.
    depth_gap: float = 0.5,
    fold_color: Tuple[int, int, int] = (0, 0, 0),
    bar_w_frac: float = 0.30,
    bar_h_frac: float = 0.72,
    jitter_frac: float = 0.04,
    notch_w_frac: float = 0.33,
    notch_h_frac: float = 0.18,
    fold_angle=(0.4, 0.95),
    body_disp: Optional[float] = None,
    d_near_override: Optional[float] = None,
    d_apex_override: Optional[float] = None,
):
    """
    Generate one 'protruding folded paper' stereo stimulus (Von Szily 1921;
    Ehrenstein & Gillam 1998).

    Three rectangular panels: a central front-parallel rectangle at constant
    d_near, plus a top and a bottom rectangle that share its short edges and
    fold back (recede) behind it. The whole scene is rendered as a real stereo
    pair by forward-warping each panel by its disparity with a painter's
    z-buffer, so the near middle occludes the receding flaps. As a result each
    folded panel is seen only as a triangular protrusion poking past the middle
    -- on the LEFT in the left view, the RIGHT in the right view.

    The middle carries a real binocular shift (it moves +d_near/2 / -d_near/2).
    Each flap's disparity ramps from d_near at the shared short edge (the fold)
    to d_far at the far edge of a band of height notch_h_frac * bar_height; the
    triangle's width follows the half-occlusion (d_near - d_far)/2. d_near /
    d_far come from --fold-d-near / --fold-d-apex, else sampled from the depth
    range. (notch_w_frac and body_disp are no longer used.)

    Seg/UDF cover the cyclopean silhouette (the middle rectangle; the flaps are
    occluded at the midpoint). Per-region amodal: obj1 = middle panel,
    obj2 = top flap, obj3 = bottom flap (flaps carry the ramped disparity).
    """
    stem = f"folded_paper_{idx}"
    if os.path.exists(os.path.join(out_root, LEFT_DIR, f"{stem}_left.png")):
        return

    color = tuple(int(c) for c in fold_color)

    # Geometry fractions may be a scalar (fixed) or a [min, max] range that is
    # sampled per sample, to vary the rectangle width/height and flap size.
    def _samp(v):
        if isinstance(v, (list, tuple)):
            return float(v[0]) if len(v) == 1 else random.uniform(float(v[0]), float(v[1]))
        return float(v)
    bar_w_frac   = _samp(bar_w_frac)
    bar_h_frac   = _samp(bar_h_frac)
    notch_w_frac = _samp(notch_w_frac)
    notch_h_frac = _samp(notch_h_frac)
    geo = make_folded_paper_geometry(
        H, W, bar_w_frac=bar_w_frac, bar_h_frac=bar_h_frac, jitter_frac=jitter_frac)
    x0, x1, y0, y1 = geo["x0"], geo["x1"], geo["y0"], geo["y1"]

    # ── Depth + fold angle ─────────────────────────────────────────────────────
    # d_near = the near (middle-panel) disparity. fold_frac in [0,1] is the fold
    # angle: 0 = flat (no fold), 1 = fully folded. It sets how far the panel
    # recedes (d_far) AND how tall the visible band is, so a larger fold gives a
    # bigger triangle in BOTH width and height.
    if d_near_override is not None:
        d_near = float(d_near_override)
    else:
        z_near = random.uniform(max(1e-3, z_min), max(z_min + 1e-3, z_max))
        d_near = float(focal_px * baseline / z_near)
    if d_apex_override is not None:               # explicit far-disparity override
        d_far = float(d_apex_override)
        fold_frac = max(0.0, min(1.0, 1.0 - d_far / max(d_near, 1e-6)))
    else:
        fold_frac = max(0.0, min(1.0, _samp(fold_angle)))
        d_far = d_near * (1.0 - fold_frac)

    # ── Three-panel folded-paper scene ────────────────────────────────────────
    # Middle = front rectangle [x0,x1] x [y0,y1] at constant d_near. The top and
    # bottom flaps share the short edges (y0, y1) and fold back BEHIND the middle;
    # within a band of height fh their disparity ramps from d_near at the fold to
    # d_far at the far edge. The near middle occludes them, so in each eye the
    # receding flap pokes out past one side of the middle as a triangle -- on the
    # LEFT in the left view and the RIGHT in the right view.
    # Band height scales with the fold angle too (steeper fold = taller band).
    fh = max(2.0, notch_h_frac * (y1 - y0) * fold_frac)

    yyr   = np.arange(H, dtype=np.float32)
    d_top = d_near + (d_far - d_near) * (yyr - y0) / fh   # d_near @ y0 -> d_far @ y0+fh
    d_bot = d_near + (d_far - d_near) * (y1 - yyr) / fh   # d_near @ y1 -> d_far @ y1-fh

    def _rect_mask(xa, xb, ya, yb):
        m = np.zeros((H, W), dtype=bool)
        ia, ib = int(round(max(0.0, xa))), int(round(min(float(W), xb)))
        ja, jb = int(round(max(0.0, ya))), int(round(min(float(H), yb)))
        if ib > ia and jb > ja:
            m[ja:jb, ia:ib] = True
        return m

    mid_mask = _rect_mask(x0, x1, y0, y1)
    top_mask = _rect_mask(x0, x1, y0, y0 + fh)
    bot_mask = _rect_mask(x0, x1, y1 - fh, y1)
    d_mid  = np.full((H, W), d_near, dtype=np.float32)
    d_topf = np.broadcast_to(d_top.reshape(H, 1), (H, W))
    d_botf = np.broadcast_to(d_bot.reshape(H, 1), (H, W))
    panels = [(mid_mask, d_mid), (top_mask, d_topf), (bot_mask, d_botf)]

    def _render(sign):
        # Forward-warp every panel pixel by sign * d/2 with a painter's z-buffer
        # (far painted first, near last) -> occlusion + disparity.
        xs_all, ys_all, d_all = [], [], []
        for mask, dfield in panels:
            ys, xs = np.nonzero(mask)
            dd = dfield[ys, xs]
            xt = np.round(xs + sign * 0.5 * dd).astype(np.int64)
            v = (xt >= 0) & (xt < W)
            xs_all.append(xt[v]); ys_all.append(ys[v]); d_all.append(dd[v])
        X = np.concatenate(xs_all); Y = np.concatenate(ys_all); D = np.concatenate(d_all)
        order = np.argsort(D, kind="stable")     # ascending: nearer (larger d) last
        X, Y, D = X[order], Y[order], D[order]
        obj  = np.zeros((H, W), dtype=bool)
        disp = np.zeros((H, W), dtype=np.float32)
        obj[Y, X]  = True
        disp[Y, X] = D
        return obj, disp

    left_obj,  disp_l = _render(+1)
    right_obj, _      = _render(-1)

    # Cyclopean (midpoint) view: the middle, plus each folded flap peeking out as
    # a small triangular sliver on BOTH long sides (half the eye-view protrusion,
    # which is itself (d_near - d(y))/2). Slivers carry the flap's ramped disp.
    cyc_obj  = mid_mask.copy()
    disp_map = np.where(mid_mask, d_near, 0.0).astype(np.float32)
    # Labelled segmap so the UDF treats the rectangle and the folded slivers as
    # SEPARATE regions: 1 = middle rectangle, 2 = top flap, 3 = bottom flap. The
    # label change at the rectangle's long edge makes that edge a udf=0 contour,
    # so the rectangle boundary does not merge with the folded parts.
    seg_lab  = mid_mask.astype(np.uint8)
    ix0, ix1 = int(round(x0)), int(round(x1))
    for (ya, yb), dband, lab in [((y0, y0 + fh), d_top, 2), ((y1 - fh, y1), d_bot, 3)]:
        for yrow in range(max(0, int(round(ya))), min(H, int(round(yb)))):
            wl = int(round(0.25 * (d_near - float(dband[yrow]))))
            if wl <= 0:
                continue
            xl = max(0, ix0 - wl)
            cyc_obj[yrow, xl:ix0] = True; disp_map[yrow, xl:ix0] = dband[yrow]; seg_lab[yrow, xl:ix0] = lab
            xr = min(W, ix1 + wl)
            cyc_obj[yrow, ix1:xr] = True; disp_map[yrow, ix1:xr] = dband[yrow]; seg_lab[yrow, ix1:xr] = lab

    def _img(obj):
        im = np.full((H, W, 3), 255, dtype=np.uint8)
        im[obj] = np.array(color, dtype=np.uint8)
        return Image.fromarray(im, mode="RGB")

    img_cyc = _img(cyc_obj)
    img_l   = _img(left_obj)
    img_r   = _img(right_obj)
    segmap  = seg_lab   # 1 = rectangle, 2 = top flap, 3 = bottom flap

    # ── Save outputs ──────────────────────────────────────────────────────────
    img_cyc.save(os.path.join(out_root, IMG_DIR,   f"{stem}.png"))
    img_l.save(  os.path.join(out_root, LEFT_DIR,  f"{stem}_left.png"))
    img_r.save(  os.path.join(out_root, RIGHT_DIR, f"{stem}_right.png"))
    np.save(os.path.join(out_root, DISP_DIR,      f"{stem}_disp.npy"), disp_map)
    np.save(os.path.join(out_root, DISP_LEFT_DIR, f"{stem}_disp.npy"), disp_l)

    seg_path   = os.path.join(out_root, SEG_DIR,   f"{stem}_seg.npy")
    field_path = os.path.join(out_root, FIELD_DIR, f"{stem}_field.npy")
    vis_path   = os.path.join(out_root, VIS_DIR,   f"{stem}_udf.png")
    np.save(seg_path, segmap)
    build_and_save_udf_from_segmap(segmap, field_path, d_max=dmax_clip)
    if save_viz:
        D = np.load(field_path).squeeze(-1)
        save_udf_visualization(img_cyc, D, vis_path,
                               d_max_for_viz=UDF_VIZ_DMAX, alpha=UDF_VIZ_ALPHA)

    # ── Per-region amodal: 1 = middle panel, 2 = top flap, 3 = bottom flap ─────
    # Middle is flat d_near; flaps carry the vertical ramp over their band.
    for m, ramp, mask_dir, disp_dir, udf_dir in [
        (mid_mask, None,   MASK_OBJ1_DIR, DISP_OBJ1_DIR, UDF_OBJ1_DIR),
        (top_mask, d_topf, MASK_OBJ2_DIR, DISP_OBJ2_DIR, UDF_OBJ2_DIR),
        (bot_mask, d_botf, MASK_OBJ3_DIR, DISP_OBJ3_DIR, UDF_OBJ3_DIR),
    ]:
        np.save(os.path.join(out_root, mask_dir, f"{stem}_mask.npy"), m.astype(np.uint8))
        region_disp = (m.astype(np.float32) * d_near) if ramp is None \
            else np.where(m, ramp, 0.0).astype(np.float32)
        np.save(os.path.join(out_root, disp_dir, f"{stem}_disp.npy"), region_disp)
        build_and_save_udf_from_mask(
            m, os.path.join(out_root, udf_dir, f"{stem}_udf.npy"), d_max=dmax_clip)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic UDF data (cyclopean) and stereo pairs."
    )
    parser.add_argument(
        "--mode",
        choices=["one", "two", "both", "multi", "comparison", "folded_paper", "delta_sweep"],
        default="both",
    )
    parser.add_argument("--n-one", type=int, default=NUM_ONE_OBJECT)
    parser.add_argument("--n-two-diff", type=int, default=NUM_TWO_OBJECT_DIFF)
    parser.add_argument("--n-two-same", type=int, default=NUM_TWO_OBJECT_SAME)
    parser.add_argument("--n-two-unpaired", type=int, default=0)
    parser.add_argument("--n-comparison", type=int, default=1000,
                        help="Number of paired samples for --mode comparison")
    parser.add_argument("--n-multi", type=int, default=10000,
                        help="Number of samples per object count for --mode multi.")

    parser.add_argument(
        "--multi-objects",
        type=int,
        nargs="+",
        default=[3, 4, 5],
        help="Object counts to generate in --mode multi. Example: --multi-objects 3 4 5",
    )
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, nargs=2, default=IMAGE_SIZE, metavar=("H","W"))
    parser.add_argument("--clip-dmax", type=float, default=None)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--shapes", nargs="+", default=None,
                        choices=ALL_SHAPES, metavar="SHAPE",
                        help=f"Shapes to sample from. Default: all. Choices: {ALL_SHAPES}")

    # Stereo/camera params
    parser.add_argument("--focal-px", type=float, default=300.0)
    parser.add_argument("--baseline", type=float, default=0.10)
    parser.add_argument("--z-min", type=float, default=0.5)
    parser.add_argument("--z-max", type=float, default=3.0)
    parser.add_argument("--p-zero-disp", type=float, default=0.0)

    parser.add_argument("--stereo-mode", choices=["shift", "slanted"], default="slanted",
                        help="shift: simple +/- d/2 translation. slanted: per-object homography "
                             "tilt applied to cyclopean/left/right, GT on the tilted cyclopean grid.")
    parser.add_argument("--slant-dist", choices=["independent", "uniform", "gamma"], default="independent",
                        help="How --stereo-mode slanted samples each object's tilt magnitude. "
                             "independent (default): s1,s2,p1,p2,ty each sampled independently at "
                             "full range -- original behavior. uniform: one shared magnitude ~ "
                             "Uniform(0, --slant-magnitude-max) scales all five together. gamma: "
                             "shared magnitude ~ Gamma(--slant-gamma-k, --slant-gamma-theta) clipped "
                             "to --slant-magnitude-max -- peaks at a small angle then decays with a "
                             "long tail, so most objects are only slightly slanted and a few are steep.")
    parser.add_argument("--slant-magnitude-max", type=float, default=1.0,
                        help="Cap on the shared slant magnitude in [0,1] (1.0 = same max tilt as "
                             "the original fixed shear/perspective/ty ranges). Used by "
                             "--slant-dist uniform/gamma.")
    parser.add_argument("--slant-gamma-k", type=float, default=2.0,
                        help="Gamma shape param for --slant-dist gamma. k>1 gives a peak away from "
                             "0 followed by decay; k=2 is the recommended default.")
    parser.add_argument("--slant-gamma-theta", type=float, default=0.25,
                        help="Gamma scale param for --slant-dist gamma; peak sits at "
                             "(k-1)*theta, tail length scales with theta.")
    parser.add_argument("--p-frontal", type=float, default=0.0,
                        help="Probability [0,1] that an object's tilt magnitude is forced to "
                             "exactly 0 (true front-parallel, no shear/perspective/ty at all), "
                             "independent of --slant-dist. None of the slant distributions ever "
                             "land on exactly 0 on their own, so this is the only way to get "
                             "genuine unslanted objects in the dataset.")
    parser.add_argument("--boundary-mode", choices=["hard", "center", "area"], default="hard",
                        help="How disp_c/disp_l/seg_c are assigned at object-boundary pixels "
                             "(--mode one/two, --stereo-mode slanted only). hard (default): "
                             "original PIL/cv2 rasterization -- floor-snapped, winner-take-all, "
                             "not sub-pixel accurate. center: exact point-in-polygon test at each "
                             "pixel's true center, still winner-take-all but geometrically exact. "
                             "area: supersample each pixel (--area-supersample) and set its "
                             "disparity to the coverage-fraction-weighted average of the object(s) "
                             "overlapping it.")
    parser.add_argument("--area-supersample", type=int, default=8,
                        help="Sub-samples per axis per pixel used by --boundary-mode area "
                             "(e.g. 8 -> 64 samples/pixel). Ignored otherwise.")
    parser.add_argument("--bg-texture", choices=["none", "mix", "all"], default="none",
                        help="Texture the backdrop behind the object(s), instead of solid white "
                             "(--mode one/two, --stereo-mode slanted only). none (default): "
                             "solid white, original behavior. mix: textured with probability "
                             "--p-bg-texture. all: always textured. The backdrop is treated as "
                             "an infinitely-far (zero-disparity) surface, so it is identical in "
                             "the left/right/cyclopean views. Sampled from --bg-texture-dir if "
                             "given, else falls back to --texture-dir (excluding whichever image "
                             "the object(s) in that sample used); with neither, falls back to a "
                             "procedural pattern with a distinct base color.")
    parser.add_argument("--p-bg-texture", type=float, default=1.0,
                        help="Probability of a textured backdrop when --bg-texture mix.")
    parser.add_argument("--bg-texture-dir", type=str, default=None,
                        help="Directory of texture images for the BACKGROUND only (e.g. a "
                             "different material class than --texture-dir, so backdrops never "
                             "share a source pool with object textures). Falls back to "
                             "--texture-dir if not given.")
    parser.add_argument("--mode-texture", choices=["none", "mix", "all"], default="all")
    parser.add_argument("--p-texture", type=float, default=1.0)
    parser.add_argument("--p-same-depth", type=float, default=0.0,
                        help="Probability that a two-object sample gets both objects at the same depth "
                             "(triggers bimodal A/B UDF for same-color pairs).")
    parser.add_argument("--idx-offset", type=int, default=0,
                        help="Add this value to every sample index (--mode two, --mode multi). "
                             "Allows multiple runs to share the same output directory without overwriting.")
    parser.add_argument("--same-disp-thresh", type=float, default=1.0,
                        help="Disparity difference (px) below which two objects are treated as same-depth "
                             "and two UDF versions (A/B) are saved instead of one.")
    parser.add_argument("--depth-gap", type=float, default=0.8,
                        help="Minimum z separation (metres) between front/back objects "
                             "in the non-equal-disp conditions of --mode comparison.")
    parser.add_argument("--texture-dir", type=str, default=None,
                        help="Directory of texture images (e.g., DTD images/). "
                             "Searched recursively. Falls back to procedural textures if not given.")
    parser.add_argument("--min-overlap-ratio", type=float, default=0.0,
                        help="Minimum intersection/smaller-object-area ratio required when "
                             "--p-large-overlap triggers. 0=disabled. E.g. 0.5 means smaller "
                             "object must be at least 50%% covered by the larger.")
    parser.add_argument("--target-iou-min", type=float, default=None,
                        help="With --target-iou-max: place the two objects so the IoU of their amodal "
                             "masks falls in [min, max] (two-object modes). Use 1.0 1.0 for identical "
                             "coincident shapes.")
    parser.add_argument("--target-iou-max", type=float, default=None)
    parser.add_argument("--delta-d-list", type=float, nargs="+", default=None,
                        help="--mode delta_sweep: disparity offsets (px) of the nearer object; one base scene is rendered "
                             "at every offset (same shapes/colour/background/slant). Uses --target-iou-min/max.")
    parser.add_argument("--n-delta-bases", type=int, default=1, help="delta_sweep: base scenes for THIS run")
    parser.add_argument("--delta-base-offset", type=int, default=0, help="delta_sweep: first base id of this run")
    parser.add_argument("--n-delta-total", type=int, default=1, help="delta_sweep: total base scenes over all runs (index stride)")
    parser.add_argument("--delta-d-range", type=float, nargs=2, default=[10.0, 50.0], help="delta_sweep: base disparity range (px)")
    parser.add_argument("--p-large-overlap", type=float, default=0.0,
                        help="Fraction of two-object samples that must satisfy --min-overlap-ratio. "
                             "E.g. 0.5 means half the samples will have large intersection.")
    parser.add_argument("--unpaired-occluded-max", type=float, default=0.05,
                        help="For --n-two-unpaired: in the eye where the back object is occluded, "
                             "at most this fraction of the back's silhouette may remain visible "
                             "(0.0 = must be 100%% occluded; 0.05 = near-complete occlusion).")
    parser.add_argument("--unpaired-visible-min", type=float, default=0.30,
                        help="For --n-two-unpaired: in the OTHER eye, at least this fraction of the "
                             "back object's silhouette must be observable (unoccluded). Raise it "
                             "(e.g. 0.5) to force the back to have a larger observable portion.")

    # Folded-paper stimulus args
    parser.add_argument("--n-folded", type=int, default=1000,
                        help="Number of folded-paper samples for --mode folded_paper.")
    parser.add_argument("--disp-polarity",
                        choices=["center_near", "center_far", "both"],
                        default="center_near",
                        help="Which strip protrudes: center_near (center closer), "
                             "center_far (sides closer), or both (save both assignments "
                             "from the same geometry under separate stems).")
    parser.add_argument("--fold-depth-gap", type=float, default=0.5,
                        help="Minimum z-separation (m) between the near (rectangle) and far "
                             "(triangle apex) depths in --mode folded_paper.")
    parser.add_argument("--fold-color", type=int, nargs=3, default=[0, 0, 0],
                        metavar=("R", "G", "B"),
                        help="Solid fill colour of the folded-paper silhouette (default black).")
    parser.add_argument("--fold-bar-w-frac", type=float, nargs="+", default=[0.20, 0.45],
                        metavar="FRAC",
                        help="Rectangle width / image width. One value = fixed; two = "
                             "[min max] sampled per sample (folded_paper).")
    parser.add_argument("--fold-bar-h-frac", type=float, nargs="+", default=[0.55, 0.85],
                        metavar="FRAC",
                        help="Rectangle height / image height. One value = fixed; two = "
                             "[min max] sampled per sample.")
    parser.add_argument("--fold-notch-w-frac", type=float, nargs="+", default=[0.25, 0.45],
                        metavar="FRAC",
                        help="Flap protrusion / bar width. One value = fixed; two = "
                             "[min max] sampled per sample.")
    parser.add_argument("--fold-notch-h-frac", type=float, nargs="+", default=[0.12, 0.25],
                        metavar="FRAC",
                        help="Flap vertical extent / bar height. One value = fixed; two = "
                             "[min max] sampled per sample.")
    parser.add_argument("--fold-body-disp", type=float, default=None,
                        help="Binocular shift of the near rectangle (px). Default = d_near "
                             "(rectangle carries a real shift); pass 0 for no shift.")
    parser.add_argument("--fold-jitter-frac", type=float, default=0.04,
                        help="Random jitter on bar centre/size for variety across samples "
                             "(0 = perfectly centred).")
    parser.add_argument("--fold-angle", type=float, nargs="+", default=[0.4, 0.95],
                        metavar="FRAC",
                        help="Fold amount in [0,1] (0 = flat, 1 = fully folded). Sets how far "
                             "each flap recedes AND its band height, so a larger fold gives a "
                             "bigger triangle. One value = fixed; two = [min max] sampled per "
                             "sample. Ignored if --fold-d-apex is given.")
    parser.add_argument("--fold-angle-sweep", action="store_true",
                        help="Sweep the fold angle linearly from min to max across the run "
                             "(sample 0 = smallest fold, last sample = largest) instead of "
                             "sampling it at random. Uses the [min max] from --fold-angle. The "
                             "rectangle shape/position is held FIXED across the sweep so only "
                             "the folded part changes.")
    parser.add_argument("--fold-sweep-seed", type=int, default=0,
                        help="Seed that fixes the (otherwise sampled) rectangle shape during "
                             "--fold-angle-sweep. Change it for a different fixed shape.")
    parser.add_argument("--fold-d-near", type=float, default=None,
                        help="GT disparity (px) of the near central panel. Else sampled "
                             "from the depth range.")
    parser.add_argument("--fold-d-apex", type=float, default=None,
                        help="Explicit far-flap disparity (px) override; bypasses --fold-angle "
                             "(the fold amount is then derived as 1 - d_apex/d_near).")

    args = parser.parse_args()

    global SLANT_DIST, SLANT_MAGNITUDE_MAX, SLANT_GAMMA_K, SLANT_GAMMA_THETA, P_FRONTAL
    global BOUNDARY_MODE, AREA_SUPERSAMPLE
    SLANT_DIST = args.slant_dist
    SLANT_MAGNITUDE_MAX = args.slant_magnitude_max
    SLANT_GAMMA_K = args.slant_gamma_k
    SLANT_GAMMA_THETA = args.slant_gamma_theta
    P_FRONTAL = args.p_frontal
    BOUNDARY_MODE = args.boundary_mode
    AREA_SUPERSAMPLE = args.area_supersample

    H, W = tuple(args.image_size)
    out_root = args.output_dir

    tex_paths = None
    if args.texture_dir is not None:
        tex_paths = preload_textures(get_texture_paths(args.texture_dir))
        print(f"Loaded {len(tex_paths)} texture images from '{args.texture_dir}' into memory")

    bg_tex_paths = tex_paths
    if args.bg_texture_dir is not None:
        bg_tex_paths = preload_textures(get_texture_paths(args.bg_texture_dir))
        print(f"Loaded {len(bg_tex_paths)} background texture images from '{args.bg_texture_dir}' into memory")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, VIS_DIR),   exist_ok=True)
    os.makedirs(os.path.join(out_root, FIELD_DIR), exist_ok=True)
    os.makedirs(os.path.join(out_root, FIELD_LEFT_DIR), exist_ok=True)
    os.makedirs(os.path.join(out_root, SEG_DIR),   exist_ok=True)
    os.makedirs(os.path.join(out_root, IMG_DIR),   exist_ok=True)
    os.makedirs(os.path.join(out_root, LEFT_DIR),      exist_ok=True)
    os.makedirs(os.path.join(out_root, RIGHT_DIR),     exist_ok=True)
    os.makedirs(os.path.join(out_root, DISP_DIR),      exist_ok=True)
    os.makedirs(os.path.join(out_root, DISP_LEFT_DIR), exist_ok=True)
    os.makedirs(os.path.join(out_root, MASK_OBJ1_DIR), exist_ok=True)
    os.makedirs(os.path.join(out_root, MASK_OBJ2_DIR), exist_ok=True)
    os.makedirs(os.path.join(out_root, DISP_OBJ1_DIR), exist_ok=True)
    os.makedirs(os.path.join(out_root, DISP_OBJ2_DIR), exist_ok=True)
    os.makedirs(os.path.join(out_root, UDF_OBJ1_DIR),  exist_ok=True)
    os.makedirs(os.path.join(out_root, UDF_OBJ2_DIR),  exist_ok=True)
    os.makedirs(os.path.join(out_root, MASK_OBJ3_DIR), exist_ok=True)
    os.makedirs(os.path.join(out_root, DISP_OBJ3_DIR), exist_ok=True)
    os.makedirs(os.path.join(out_root, UDF_OBJ3_DIR),  exist_ok=True)

    if args.mode == "comparison":
        for cond in COMPARISON_SUBDIRS:
            for sub in (VIS_DIR, FIELD_DIR, SEG_DIR, IMG_DIR,
                        LEFT_DIR, RIGHT_DIR, DISP_DIR, DISP_LEFT_DIR):
                os.makedirs(os.path.join(out_root, cond, sub), exist_ok=True)

    if args.mode == "delta_sweep":
        assert args.delta_d_list and args.target_iou_min is not None and args.target_iou_max is not None
        for b in range(args.n_delta_bases):
            gen_delta_d_sweep(args.delta_base_offset + b, args.n_delta_total, args.delta_d_list, H, W, out_root,
                              args.clip_dmax, (args.target_iou_min, args.target_iou_max), shapes=args.shapes,
                              bg_texture_mode=args.bg_texture, p_bg_texture=args.p_bg_texture,
                              bg_tex_paths=bg_tex_paths, d_range=tuple(args.delta_d_range))
        print(f"[delta_sweep] done: bases {args.delta_base_offset}..{args.delta_base_offset + args.n_delta_bases - 1}")

    if args.mode in ("one", "both"):
        for i in range(args.n_one):
            idx = args.idx_offset + i
            gen_one_object(
                idx, H, W, out_root,
                dmax_clip=args.clip_dmax,
                save_viz=(not args.no_viz),
                focal_px=args.focal_px,
                baseline=args.baseline,
                z_min=args.z_min,
                z_max=args.z_max,
                p_zero_disp=args.p_zero_disp,
                stereo_mode=args.stereo_mode,
                mode_texture=args.mode_texture,
                p_texture=args.p_texture,
                tex_paths=tex_paths,
                shapes=args.shapes,
                bg_texture_mode=args.bg_texture,
                p_bg_texture=args.p_bg_texture,
                bg_tex_paths=bg_tex_paths,
            )
            if (i+1) % 1000 == 0:
                print(f"[one] {i+1}/{args.n_one}")

    if args.mode in ("two", "both"):
        for j in range(args.n_two_diff):
            idx = args.idx_offset + j
            gen_two_objects(
                idx, H, W, out_root,
                dmax_clip=args.clip_dmax,
                save_viz=(not args.no_viz),
                same_color=False,
                enforce_color_gap=True,
                min_gap=60,
                focal_px=args.focal_px,
                baseline=args.baseline,
                z_min=args.z_min,
                z_max=args.z_max,
                p_zero_disp=args.p_zero_disp,
                stereo_mode=args.stereo_mode,
                mode_texture=args.mode_texture,
                p_texture=args.p_texture,
                tex_paths=tex_paths,
                same_disp_thresh=args.same_disp_thresh,
                shapes=args.shapes,
                p_same_depth=args.p_same_depth,
                min_overlap_ratio=args.min_overlap_ratio,
                p_large_overlap=args.p_large_overlap,
                target_iou_range=(None if args.target_iou_min is None else (args.target_iou_min, args.target_iou_max)),
                bg_texture_mode=args.bg_texture,
                p_bg_texture=args.p_bg_texture,
                bg_tex_paths=bg_tex_paths,
            )
            if (j+1) % 1000 == 0:
                print(f"[two-diff] {j+1}/{args.n_two_diff}")

        for k in range(args.n_two_same):
            idx = args.idx_offset + args.n_two_diff + k
            gen_two_objects(
                idx, H, W, out_root,
                dmax_clip=args.clip_dmax,
                save_viz=(not args.no_viz),
                same_color=True,
                enforce_color_gap=False,
                focal_px=args.focal_px,
                baseline=args.baseline,
                z_min=args.z_min,
                z_max=args.z_max,
                p_zero_disp=args.p_zero_disp,
                stereo_mode=args.stereo_mode,
                mode_texture=args.mode_texture,
                p_texture=args.p_texture,
                tex_paths=tex_paths,
                same_disp_thresh=args.same_disp_thresh,
                shapes=args.shapes,
                p_same_depth=args.p_same_depth,
                min_overlap_ratio=args.min_overlap_ratio,
                p_large_overlap=args.p_large_overlap,
                target_iou_range=(None if args.target_iou_min is None else (args.target_iou_min, args.target_iou_max)),
                bg_texture_mode=args.bg_texture,
                p_bg_texture=args.p_bg_texture,
                bg_tex_paths=bg_tex_paths,
            )
            if (k+1) % 1000 == 0:
                print(f"[two-same] {k+1}/{args.n_two_same}")

        for u in range(args.n_two_unpaired):
            idx = args.idx_offset + args.n_two_diff + args.n_two_same + u
            gen_two_objects(
                idx, H, W, out_root,
                dmax_clip=args.clip_dmax,
                save_viz=(not args.no_viz),
                same_color=False,
                enforce_color_gap=True,
                min_gap=60,
                focal_px=args.focal_px,
                baseline=args.baseline,
                z_min=args.z_min,
                z_max=args.z_max,
                p_zero_disp=args.p_zero_disp,
                stereo_mode=args.stereo_mode,
                mode_texture=args.mode_texture,
                p_texture=args.p_texture,
                tex_paths=tex_paths,
                same_disp_thresh=args.same_disp_thresh,
                shapes=args.shapes,
                p_same_depth=0.0,
                require_unpaired=True,
                min_overlap_ratio=args.min_overlap_ratio,
                p_large_overlap=args.p_large_overlap,
                target_iou_range=(None if args.target_iou_min is None else (args.target_iou_min, args.target_iou_max)),
                unpaired_occluded_max=args.unpaired_occluded_max,
                unpaired_visible_min=args.unpaired_visible_min,
                bg_texture_mode=args.bg_texture,
                p_bg_texture=args.p_bg_texture,
                bg_tex_paths=bg_tex_paths,
            )
            if (u+1) % 1000 == 0:
                print(f"[two-unpaired] {u+1}/{args.n_two_unpaired}")

    if args.mode == "multi":
        for n_obj in args.multi_objects:
            for k in range(args.n_multi):
                gen_multi_objects(
                    idx=args.idx_offset + k,
                    n_obj=n_obj,
                    H=H,
                    W=W,
                    out_root=out_root,
                    dmax_clip=args.clip_dmax,
                    save_viz=(not args.no_viz),
                    focal_px=args.focal_px,
                    baseline=args.baseline,
                    z_min=args.z_min,
                    z_max=args.z_max,
                    p_zero_disp=args.p_zero_disp,
                    stereo_mode=args.stereo_mode,
                    mode_texture=args.mode_texture,
                    p_texture=args.p_texture,
                    tex_paths=tex_paths,
                    shapes=args.shapes,
                    same_color=False,
                )

                if (k + 1) % 1000 == 0:
                    print(f"[multi-{n_obj}obj] {k+1}/{args.n_multi}")

    if args.mode == "comparison":
        for c in range(args.n_comparison):
            gen_comparison_pair(
                c, H, W, out_root,
                dmax_clip=args.clip_dmax,
                save_viz=(not args.no_viz),
                focal_px=args.focal_px,
                baseline=args.baseline,
                z_min=args.z_min,
                z_max=args.z_max,
                stereo_mode=args.stereo_mode,
                mode_texture=args.mode_texture,
                p_texture=args.p_texture,
                tex_paths=tex_paths,
                same_disp_thresh=args.same_disp_thresh,
                shapes=args.shapes,
                depth_gap=args.depth_gap,
            )
            if (c + 1) % 100 == 0:
                print(f"[comparison] {c+1}/{args.n_comparison}")

    if args.mode == "folded_paper":
        for fp in range(args.n_folded):
            # Optionally sweep the fold angle linearly from min->max across the
            # run (sample 0 = smallest fold, last sample = largest) instead of
            # sampling it at random per sample. In sweep mode the RNG is reseeded
            # to a fixed value every sample, so the rectangle shape/position and
            # all other random draws stay identical -- only the fold changes.
            fold_angle = args.fold_angle
            if args.fold_angle_sweep and len(fold_angle) >= 2:
                random.seed(args.fold_sweep_seed)
                lo, hi = float(fold_angle[0]), float(fold_angle[-1])
                t = fp / max(1, args.n_folded - 1)
                fold_angle = [lo + (hi - lo) * t]   # scalar -> fixed for this sample
            gen_folded_paper(
                fp, H, W, out_root,
                dmax_clip=args.clip_dmax,
                save_viz=(not args.no_viz),
                focal_px=args.focal_px,
                baseline=args.baseline,
                z_min=args.z_min,
                z_max=args.z_max,
                stereo_mode=args.stereo_mode,
                mode_texture=args.mode_texture,
                p_texture=args.p_texture,
                tex_paths=tex_paths,
                disp_polarity=args.disp_polarity,
                depth_gap=args.fold_depth_gap,
                fold_color=tuple(args.fold_color),
                bar_w_frac=args.fold_bar_w_frac,
                bar_h_frac=args.fold_bar_h_frac,
                notch_w_frac=args.fold_notch_w_frac,
                notch_h_frac=args.fold_notch_h_frac,
                fold_angle=fold_angle,
                body_disp=args.fold_body_disp,
                jitter_frac=args.fold_jitter_frac,
                d_near_override=args.fold_d_near,
                d_apex_override=args.fold_d_apex,
            )
            if (fp + 1) % 500 == 0:
                print(f"[folded_paper] {fp+1}/{args.n_folded}")

    if args.mode == "comparison":
        print(f"\nComparison dataset saved under '{out_root}':")
        for cond in COMPARISON_SUBDIRS:
            n = len(os.listdir(os.path.join(out_root, cond, LEFT_DIR)))
            print(f"  {cond:<35s}  {n} samples")
    else:
        print(f"Saved cyclopean images in    '{os.path.join(out_root, IMG_DIR)}'")
        print(f"Saved left images in         '{os.path.join(out_root, LEFT_DIR)}'")
        print(f"Saved right images in        '{os.path.join(out_root, RIGHT_DIR)}'")
        print(f"Saved GT disparity in        '{os.path.join(out_root, DISP_DIR)}' (float32 .npy, cyclopean grid)")
        print(f"Saved UDF fields in          '{os.path.join(out_root, FIELD_DIR)}'")
        print(f"Saved segment maps in        '{os.path.join(out_root, SEG_DIR)}'")
        if not args.no_viz:
            print(f"Saved UDF visualizations in  '{os.path.join(out_root, VIS_DIR)}'")

if __name__ == "__main__":
    main()
