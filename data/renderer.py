"""Deterministic synthetic scene renderer for the tiny UMM dataset.

A scene contains n objects with the same shape and color.  The source image and
its edit target share identical object centers/sizes, background parameters and
noise field; only the global shape and/or color changes.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .schema import palette


# ---------------------------------------------------------------------------
# shape / layout primitives
# ---------------------------------------------------------------------------
def _pixel_centers(image_size: int) -> Tuple[np.ndarray, np.ndarray]:
    c = np.arange(image_size, dtype=np.float32) + 0.5
    return np.meshgrid(c, c, indexing="xy")  # xs, ys


def _inside_shape(shape: int, xs: np.ndarray, ys: np.ndarray,
                  cx: float, cy: float, half: float, angle: float = 0.0) -> np.ndarray:
    """Boolean inside test for square(0), circle(1), upward triangle(2).

    Boundary points count as inside.  Works for arbitrary coordinate arrays so
    the same geometry drives both the binary mask and supersampled coverage.
    The shape is rotated about its center and kept inside the same
    axis-aligned square object bbox of side ``2 * half``; for squares and
    triangles that means the rotated silhouette is scaled to fit the box.
    """
    if shape == 1:
        return (xs - cx) ** 2 + (ys - cy) ** 2 <= half**2

    dx = xs - cx
    dy = ys - cy
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    if shape == 0:
        # Inverse-rotate into the square's local frame.  The box-extent of a
        # rotated square grows by |cos|+|sin|, so rescale to stay inscribed.
        xr = c * dx + s * dy
        yr = -s * dx + c * dy
        he = half / max(1e-6, abs(c) + abs(s))
        return (np.abs(xr) <= he) & (np.abs(yr) <= he)

    # Upward triangle with its apex at local (0,-half).  Rotate the three
    # vertices, then scale uniformly so both bbox dimensions fit in 2*half.
    verts = [(0.0, -half), (-half, half), (half, half)]
    rot = [(c * x - s * y, s * x + c * y) for x, y in verts]
    vx = [p[0] for p in rot]
    vy = [p[1] for p in rot]
    # Fit about the rotation center. Using only max-min spans clips an
    # asymmetric rotated triangle whose vertices extend past one side.
    radius = max(max(map(abs, vx)), max(map(abs, vy)), 1e-6)
    scale = half / radius
    x1, y1 = rot[0][0] * scale, rot[0][1] * scale
    x2, y2 = rot[1][0] * scale, rot[1][1] * scale
    x3, y3 = rot[2][0] * scale, rot[2][1] * scale

    def edge(a_x, a_y, b_x, b_y):
        return (dx - a_x) * (b_y - a_y) - (dy - a_y) * (b_x - a_x)

    d1 = edge(x1, y1, x2, y2)
    d2 = edge(x2, y2, x3, y3)
    d3 = edge(x3, y3, x1, y1)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


def shape_mask(shape: int, cx: float, cy: float, size: float, image_size: int,
               angle: float = 0.0) -> np.ndarray:
    """Binary mask for square(0), circle(1), or upward triangle(2).

    `size` is the side of the common axis-aligned object bbox and `angle` is in
    radians.  This keeps hard-edge semantics because evaluation uses it as a
    shape template; use :func:`shape_coverage` for anti-aliased rendering.
    """
    xs, ys = _pixel_centers(image_size)
    return _inside_shape(shape, xs, ys, float(cx), float(cy), float(size) / 2.0,
                         float(angle))


def shape_coverage(
    shape: int,
    cx: float,
    cy: float,
    size: float,
    image_size: int,
    supersample: int = 4,
    angle: float = 0.0,
) -> np.ndarray:
    """Anti-aliased coverage in [0, 1] via vectorized supersampling.

    Each pixel is sampled on an ``supersample x supersample`` grid of subpixel
    centers; the returned value is the fraction of samples inside the shape.
    Only pixels intersecting the square object bbox are evaluated, so the cost
    stays proportional to the shape area rather than the full image.
    """
    ss = max(1, int(supersample))
    image_size = int(image_size)
    cx, cy = float(cx), float(cy)
    half = float(size) / 2.0
    coverage = np.zeros((image_size, image_size), dtype=np.float32)

    # Pixels whose unit square intersects the continuous object bbox.
    xa = max(0, int(math.floor(cx - half)))
    xb = min(image_size, int(math.ceil(cx + half)))
    ya = max(0, int(math.floor(cy - half)))
    yb = min(image_size, int(math.ceil(cy + half)))
    if xb <= xa or yb <= ya:
        return coverage

    # Subpixel sample centers, ordered pixel-major so a reshape separates the
    # pixel index from the supersample index.
    sx = xa + (np.arange((xb - xa) * ss, dtype=np.float32) + 0.5) / ss
    sy = ya + (np.arange((yb - ya) * ss, dtype=np.float32) + 0.5) / ss
    gx, gy = np.meshgrid(sx, sy, indexing="xy")
    inside = _inside_shape(shape, gx, gy, cx, cy, half, float(angle))
    coverage[ya:yb, xa:xb] = inside.reshape(yb - ya, ss, xb - xa, ss).mean(
        axis=(1, 3), dtype=np.float32
    )
    return coverage


def object_bbox(cx: float, cy: float, size: float) -> Tuple[float, float, float, float]:
    h = float(size) / 2.0
    return cx - h, cy - h, cx + h, cy + h


def bbox_mask(centers: np.ndarray, sizes: np.ndarray, count: int, image_size: int, pad: int = 0) -> np.ndarray:
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    for i in range(count):
        cx, cy = float(centers[i, 0]), float(centers[i, 1])
        s = float(sizes[i])
        x0, y0, x1, y1 = object_bbox(cx, cy, s)
        xa = max(0, int(math.floor(x0)) - pad)
        ya = max(0, int(math.floor(y0)) - pad)
        xb = min(image_size, int(math.ceil(x1)) + pad + 1)
        yb = min(image_size, int(math.ceil(y1)) + pad + 1)
        mask[ya:yb, xa:xb] = 1
    return mask


def _bbox_inside(cx: float, cy: float, size: float, image_size: int, margin: float) -> bool:
    h = size / 2.0
    return (cx - h >= margin and cy - h >= margin and
            cx + h <= image_size - margin and cy + h <= image_size - margin)


def _boxes_separated(centers: np.ndarray, sizes: np.ndarray, count: int, gap: float) -> bool:
    for i in range(count):
        for j in range(i + 1, count):
            dx = abs(float(centers[i, 0] - centers[j, 0]))
            dy = abs(float(centers[i, 1] - centers[j, 1]))
            sx = (float(sizes[i]) + float(sizes[j])) / 2.0 + gap
            sy = sx
            # Intersection in both axes means the inflated boxes collide.
            if dx < sx and dy < sy:
                return False
    return True


def _choose_grid(count: int, image_size: int = 28) -> Tuple[int, int]:
    best = (1, count)
    best_score = -1e9
    for rows in range(1, count + 1):
        cols = int(math.ceil(count / rows))
        cell = min(float(image_size) / cols, float(image_size) / rows)
        waste = rows * cols - count
        score = cell - 1.5 * waste
        if score > best_score:
            best_score = score
            best = (rows, cols)
    return best


def _sample_layout(
    rng: np.random.Generator,
    count: int,
    arrangement: str,
    size_ranges: Dict[int, list],
    image_size: int,
    margin: float,
    gap: float,
    max_attempts: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    lo, hi = [int(v) for v in size_ranges[count]]
    best = None
    for _ in range(max_attempts):
        sizes = rng.integers(lo, hi + 1, size=count).astype(np.float32)
        centers = np.zeros((count, 2), dtype=np.float32)
        if arrangement == "grid":
            rows, cols = _choose_grid(count, image_size)
            cell_w, cell_h = image_size / cols, image_size / rows
            cells = rng.choice(rows * cols, size=count, replace=False)
            ok = True
            for k, cell in enumerate(cells):
                r, c = divmod(int(cell), cols)
                cell_x0, cell_y0 = c * cell_w, r * cell_h
                hi_fit = min(float(sizes[k]), min(cell_w, cell_h) - 2.0 * margin)
                if hi_fit < max(3.0, lo):
                    ok = False
                    break
                sizes[k] = float(rng.integers(max(3, int(math.floor(lo))), int(math.floor(hi_fit)) + 1))
                half = sizes[k] / 2.0
                xmin, xmax = cell_x0 + margin + half, cell_x0 + cell_w - margin - half
                ymin, ymax = cell_y0 + margin + half, cell_y0 + cell_h - margin - half
                if xmax < xmin or ymax < ymin:
                    ok = False
                    break
                centers[k, 0] = rng.uniform(xmin, xmax)
                centers[k, 1] = rng.uniform(ymin, ymax)
            if not ok:
                continue
        else:
            for k in range(count):
                if arrangement == "edge" and k == 0:
                    half = sizes[k] / 2.0
                    side = int(rng.integers(0, 4))
                    if side == 0:
                        centers[k, 0] = rng.uniform(margin + half, margin + half + 2.0)
                        centers[k, 1] = rng.uniform(margin + half, image_size - margin - half)
                    elif side == 1:
                        centers[k, 0] = rng.uniform(image_size - margin - half - 2.0, image_size - margin - half)
                        centers[k, 1] = rng.uniform(margin + half, image_size - margin - half)
                    elif side == 2:
                        centers[k, 0] = rng.uniform(margin + half, image_size - margin - half)
                        centers[k, 1] = rng.uniform(margin + half, margin + half + 2.0)
                    else:
                        centers[k, 0] = rng.uniform(margin + half, image_size - margin - half)
                        centers[k, 1] = rng.uniform(image_size - margin - half - 2.0, image_size - margin - half)
                else:
                    half = sizes[k] / 2.0
                    centers[k, 0] = rng.uniform(margin + half, image_size - margin - half)
                    centers[k, 1] = rng.uniform(margin + half, image_size - margin - half)
        if not all(_bbox_inside(centers[k, 0], centers[k, 1], sizes[k], image_size, margin)
                   for k in range(count)):
            continue
        if not _boxes_separated(centers, sizes, count, gap):
            continue
        # Grid placements are constructed safely; random/edge need this check.
        best = (centers, sizes)
        break
    return best


# ---------------------------------------------------------------------------
# background / noise
# ---------------------------------------------------------------------------
def make_background(
    image_size: int,
    base_rgb: np.ndarray,
    bg_type: int,
    bg_dir: int,
    f0: float,
    f1: float,
    checker_amp: float,
    checker_cell: int,
    checker_phase: int,
) -> np.ndarray:
    ys, xs = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    if bg_type == 0:
        return np.broadcast_to(base_rgb[None, None, :], (image_size, image_size, 3)).copy()
    if bg_type == 1:
        if bg_dir == 0:
            t = xs / max(1, image_size - 1)
        elif bg_dir == 1:
            t = ys / max(1, image_size - 1)
        else:
            t = (xs + ys) / max(1, 2 * (image_size - 1))
        factor = f0 + (f1 - f0) * t
        return np.clip(base_rgb[None, None, :] * factor[..., None], 0.0, 1.0)
    # checker
    cell = max(1, int(checker_cell))
    alt = (((xs // cell) + (ys // cell) + int(checker_phase)) % 2).astype(np.float32)
    factor = 1.0 + checker_amp * (2.0 * alt - 1.0)
    return np.clip(base_rgb[None, None, :] * factor[..., None], 0.0, 1.0)


def make_noise_field(
    image_size: int,
    noise_type: int,
    sigma: float,
    fraction: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    if noise_type == 0:
        return np.zeros((image_size, image_size, 3), dtype=np.float32)
    if noise_type == 1:
        return rng.normal(0.0, float(sigma), size=(image_size, image_size, 3)).astype(np.float32)
    # sparse additive impulse-like corruption
    mask = rng.random((image_size, image_size, 1)) < float(fraction)
    vals = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(image_size, image_size, 3))
    mag = rng.uniform(0.15, 0.55, size=(image_size, image_size, 3)).astype(np.float32)
    return mask.astype(np.float32) * vals * mag


# ---------------------------------------------------------------------------
# scene construction
# ---------------------------------------------------------------------------
def render_objects(
    shape: int,
    color: int,
    centers: np.ndarray,
    sizes: np.ndarray,
    count: int,
    image_size: int,
    bg_img: np.ndarray,
    noise_field: np.ndarray,
    color_rgb: np.ndarray,
    supersample: int = 4,
    angles: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Alpha-blend anti-aliased objects over the background, then add noise.

    The returned mask marks every pixel whose color was affected by any shape
    coverage (> 0), including partially covered anti-aliased edge pixels.
    `angles` is an optional per-object rotation array in radians.
    """
    canvas = bg_img.astype(np.float32, copy=True)
    union = np.zeros((image_size, image_size), dtype=np.uint8)
    rgb = np.asarray(color_rgb[color], dtype=np.float32)
    for i in range(count):
        angle = 0.0 if angles is None else float(angles[i])
        cov = shape_coverage(
            shape, float(centers[i, 0]), float(centers[i, 1]),
            float(sizes[i]), image_size, supersample=int(supersample),
            angle=angle,
        )
        union[cov > 0.0] = 1
        alpha = cov[..., None]
        canvas = canvas * (1.0 - alpha) + rgb[None, None, :] * alpha
    noise = noise_field[..., None] if np.ndim(noise_field) == 2 else noise_field
    canvas = np.clip(canvas + noise, 0.0, 1.0)
    return (canvas * 255.0 + 0.5).astype(np.uint8), union


def _supersample_from_cfg(cfg: Dict[str, Any]) -> int:
    """Supersample factor from ``data.render.supersample`` with a safe fallback."""
    value = cfg.get("data", {}).get("render", {}).get("supersample", 4)
    try:
        factor = int(value)
        return factor if not isinstance(value, bool) and factor > 0 and factor == float(value) else 4
    except (TypeError, ValueError, OverflowError):
        return 4


def _validate_masks(
    cfg: Dict[str, Any],
    shape: int,
    centers: np.ndarray,
    sizes: np.ndarray,
    count: int,
    angles: Optional[np.ndarray] = None,
) -> bool:
    image_size = int(cfg["image_size"])
    min_area = int(cfg["data"]["layout"]["min_object_area"])
    min_tri = int(cfg["data"]["layout"]["min_triangle_area"])
    for i in range(count):
        angle = 0.0 if angles is None else float(angles[i])
        m = shape_mask(shape, float(centers[i, 0]), float(centers[i, 1]),
                       float(sizes[i]), image_size, angle=angle)
        area = int(m.sum())
        if area < min_area:
            return False
        if shape == 2 and area < min_tri:
            return False
    return True


def _choice_from_probs(rng: np.random.Generator, n: int, probs) -> int:
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum()
    return int(rng.choice(n, p=p))


def _sample_angles(cfg: Dict[str, Any], rng: np.random.Generator, count: int) -> np.ndarray:
    """Sample per-object orientations from a discrete circle (radians)."""
    rot = cfg.get("data", {}).get("rotation", {})
    n_angles = int(rot.get("num_angles", 12)) if rot.get("enabled", True) else 1
    if n_angles <= 1:
        return np.zeros(count, dtype=np.float32)
    ids = rng.integers(0, n_angles, size=count)
    return (ids.astype(np.float32) * (2.0 * math.pi / n_angles)).astype(np.float32)


def sample_scene(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    count: int,
    color: int,
    shape: int,
    scene_id: int,
    split: str,
    is_comp: bool,
) -> Optional[Dict[str, Any]]:
    """Sample one valid scene and its edit target (or return None on rejection)."""
    image_size = int(cfg["image_size"])
    layout_cfg = cfg["data"]["layout"]
    pal = palette(cfg)
    min_fg_bg = float(cfg["data"]["background"]["min_fg_bg_distance"]) / 255.0
    count = int(count)
    lo, hi = [int(v) for v in layout_cfg["size_ranges"][count]]
    arrangement_name = _choice_name(rng, list(layout_cfg["arrangement_probs"].keys()), list(layout_cfg["arrangement_probs"].values()))
    layout = _sample_layout(
        rng, count, arrangement_name, layout_cfg["size_ranges"], image_size,
        float(layout_cfg["margin"]), float(layout_cfg["object_gap"]), int(layout_cfg["max_attempts"]),
    )
    if layout is None:
        return None
    centers, sizes = layout
    angles = _sample_angles(cfg, rng, count)

    if not _validate_masks(cfg, shape, centers, sizes, count, angles):
        return None

    # Background color must contrast with the object color.
    bg_candidates = [c for c in range(len(pal)) if np.linalg.norm(pal[c] - pal[color]) >= min_fg_bg]
    if not bg_candidates:
        return None
    bg_color = int(rng.choice(bg_candidates))
    bg_types = cfg["data"]["background"]["types"]
    bg_type = _choice_from_probs(rng, len(bg_types), cfg["data"]["background"]["probs"])
    bg_dir = int(rng.integers(0, 3))
    f_lo, f_hi = [float(v) for v in cfg["data"]["background"]["gradient_factor"]]
    f0, f1 = float(rng.uniform(f_lo, f_hi)), float(rng.uniform(f_lo, f_hi))
    checker_amp = float(cfg["data"]["background"]["checker_factor"])
    checker_cell = int(cfg["data"]["background"]["checker_cell"])
    checker_phase = int(rng.integers(0, 2))
    bg_img = make_background(image_size, pal[bg_color], bg_type, bg_dir, f0, f1,
                             checker_amp, checker_cell, checker_phase)
    # Verify contrast against the actual gradient/texture extrema, too.
    if np.min(np.linalg.norm(bg_img - pal[color], axis=-1)) < min_fg_bg:
        return None

    noise_types = cfg["data"]["noise"]["types"]
    noise_type = _choice_from_probs(rng, len(noise_types), cfg["data"]["noise"]["probs"])
    n_lo, n_hi = [float(v) for v in cfg["data"]["noise"]["gaussian_sigma"]]
    sigma = float(rng.uniform(n_lo, n_hi)) if noise_type == 1 else 0.0
    i_lo, i_hi = [float(v) for v in cfg["data"]["noise"]["impulse_fraction"]]
    fraction = float(rng.uniform(i_lo, i_hi)) if noise_type == 2 else 0.0
    noise_seed = int(rng.integers(0, 2**31 - 1))
    noise_field = make_noise_field(image_size, noise_type, sigma, fraction, noise_seed)
    supersample = _supersample_from_cfg(cfg)

    src_img, src_mask = render_objects(
        shape, color, centers, sizes, count, image_size, bg_img, noise_field, pal,
        supersample=supersample, angles=angles,
    )

    # Choose edit type and target attributes.
    edit_cfg = cfg["data"]["edit"]
    edit_type = _choice_from_probs(rng, len(edit_cfg["types"]), edit_cfg["probs"])
    edit_type_name = edit_cfg["types"][edit_type]
    target_shapes = [s for s in range(len(cfg["shapes"]["names"])) if s != shape]
    target_colors = [c for c in range(len(pal))
                     if np.linalg.norm(pal[c] - pal[bg_color]) >= min_fg_bg
                     and c != color]
    if edit_type_name == "color":
        target_shape = shape
        if not target_colors:
            return None
        target_color = int(rng.choice(target_colors))
    elif edit_type_name == "shape":
        target_color = color
        if not target_shapes:
            return None
        target_shape = int(rng.choice(target_shapes))
    else:
        if not target_colors or not target_shapes:
            return None
        target_color = int(rng.choice(target_colors))
        target_shape = int(rng.choice(target_shapes))

    if not _validate_masks(cfg, target_shape, centers, sizes, count, angles):
        return None
    if np.min(np.linalg.norm(bg_img - pal[target_color], axis=-1)) < min_fg_bg:
        return None

    tgt_img, tgt_mask = render_objects(
        target_shape, target_color, centers, sizes, count, image_size, bg_img, noise_field, pal,
        supersample=supersample, angles=angles,
    )
    region_mask = bbox_mask(centers, sizes, count, image_size, int(edit_cfg["bbox_pad"]))

    return {
        "image": src_img,
        "edit_target": tgt_img,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
        "region_mask": region_mask,
        "count": count,
        "color": int(color),
        "shape": int(shape),
        "edit_color": int(target_color),
        "edit_shape": int(target_shape),
        "edit_type": int(edit_type),
        "centers": centers.astype(np.float32),
        "sizes": sizes.astype(np.float32),
        "angles": angles.astype(np.float32),
        "bg_color": bg_color,
        "bg_type": bg_type,
        "bg_dir": bg_dir,
        "bg_f0": f0,
        "bg_f1": f1,
        "bg_checker_amp": checker_amp,
        "bg_checker_cell": checker_cell,
        "bg_checker_phase": checker_phase,
        "noise_type": noise_type,
        "noise_sigma": sigma,
        "noise_fraction": fraction,
        "noise_seed": noise_seed,
        "arrangement": arrangement_name,
        "scene_id": int(scene_id),
        "split": split,
        "is_comp": int(is_comp),
    }


def _choice_name(rng: np.random.Generator, names, probs) -> str:
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum()
    return str(rng.choice(names, p=p))


def rerender_from_scene(cfg: Dict[str, Any], scene: Dict[str, Any], shape: int, color: int,
                        target: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce a render from stored parameters (used by invariants/tests)."""
    image_size = int(cfg["image_size"])
    pal = palette(cfg)
    bg = make_background(
        image_size, pal[int(scene["bg_color"])], int(scene["bg_type"]), int(scene["bg_dir"]),
        float(scene["bg_f0"]), float(scene["bg_f1"]), float(scene["bg_checker_amp"]),
        int(scene["bg_checker_cell"]), int(scene["bg_checker_phase"]),
    )
    noise = make_noise_field(
        image_size, int(scene["noise_type"]), float(scene["noise_sigma"]),
        float(scene["noise_fraction"]), int(scene["noise_seed"]),
    )
    angles = scene.get("angles")
    if angles is None:
        angles = np.zeros(int(scene["count"]), dtype=np.float32)
    return render_objects(
        shape, color, np.asarray(scene["centers"], dtype=np.float32),
        np.asarray(scene["sizes"], dtype=np.float32), int(scene["count"]),
        image_size, bg, noise, pal, supersample=_supersample_from_cfg(cfg),
        angles=np.asarray(angles, dtype=np.float32),
    )


def foreground_union(shape: int, centers: np.ndarray, sizes: np.ndarray, count: int,
                     image_size: int, supersample: int = 4,
                     angles: Optional[np.ndarray] = None) -> np.ndarray:
    """Union of all anti-aliased affected pixels (any coverage > 0)."""
    m = np.zeros((image_size, image_size), dtype=np.uint8)
    for i in range(count):
        angle = 0.0 if angles is None else float(angles[i])
        cov = shape_coverage(shape, float(centers[i, 0]), float(centers[i, 1]),
                             float(sizes[i]), image_size, supersample=int(supersample),
                             angle=angle)
        m[cov > 0.0] = 1
    return m
