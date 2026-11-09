"""Rule-based image parsing and metrics; independent of the trained models."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from functools import cache, cached_property
from itertools import permutations, product

import numpy as np

from data.renderer import bbox_mask, shape_mask
from data.schema import palette

ATTRIBUTES = ("count", "color", "shape")


@dataclass(frozen=True)
class ObjectFit:
    x: float
    y: float
    size: float


@dataclass
class ParsedScene:
    count: int | None
    color: int | None
    shape: int | None
    objects: tuple[ObjectFit, ...]
    foreground: np.ndarray

    @property
    def parsed(self) -> bool:
        return self.count is not None and self.color is not None and self.shape is not None


def components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    """Return the coordinates of each 8-connected region, excluding isolated noise."""
    remaining = np.pad(mask, 1)
    groups = []
    for y, x in np.argwhere(remaining):
        y, x = int(y), int(x)
        if not remaining[y, x]:
            continue
        stack = [(y, x)]
        remaining[y, x] = False
        points = []
        while stack:
            yy, xx = stack.pop()
            points.append((yy - 1, xx - 1))
            for ny in range(yy - 1, yy + 2):
                for nx in range(xx - 1, xx + 2):
                    if remaining[ny, nx]:
                        remaining[ny, nx] = False
                        stack.append((ny, nx))
        if len(points) >= min_area:
            groups.append(np.asarray(points))
    return groups


def _bounds(points: np.ndarray) -> tuple[int, int, int, int]:
    y, x = map(int, points.min(axis=0))
    bottom, right = map(int, points.max(axis=0))
    return y, x, bottom - y + 1, right - x + 1


class SceneOracle:
    """Fit the background, segment a common palette color, and compare rotated templates."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.size = cfg["image_size"]
        self.palette = palette(cfg)
        self.settings = cfg["evaluation"]["oracle"]
        ax = np.linspace(-1.0, 1.0, self.size, dtype=np.float32)
        x, y = np.meshgrid(ax, ax)
        self.design = np.stack((np.ones_like(x), x, y, x * x, x * y, y * y), axis=-1)
        self.border = (np.abs(x) == 1) | (np.abs(y) == 1)
        self.border_design = self.design[self.border]
        self._aligned: dict[tuple[int, int], tuple] = {}

    def fit_background(self, rgb: np.ndarray) -> np.ndarray:
        """Quadratic RGB surface with iterative rejection of border impulses."""
        samples = rgb[self.border]
        coeff = np.linalg.lstsq(self.border_design, samples, rcond=None)[0]
        for _ in range(3):
            error = np.linalg.norm(self.border_design @ coeff - samples, axis=-1)
            keep = error <= max(0.025, float(np.quantile(error, 0.80)))
            coeff = np.linalg.lstsq(self.border_design[keep], samples[keep], rcond=None)[0]
        return (self.design @ coeff).clip(0, 1)

    @cached_property
    def templates(self) -> dict:
        bank = defaultdict(list)
        rotation = self.cfg["data"].get("rotation", {})
        n = max(1, rotation.get("num_angles", 12)) if rotation.get("enabled", True) else 1
        angles = np.arange(n) * (2 * math.pi / n)
        max_size = max(high for _, high in self.cfg["data"]["layout"]["size_ranges"].values())
        for shape in range(len(self.cfg["shapes"]["names"])):
            if shape == 1:  # circle
                allowed = (0.0,)
            elif shape == 0:  # square repeats every quarter turn
                allowed = sorted({round(angle % (math.pi / 2), 6) for angle in angles})
            else:
                allowed = angles
            for side, angle, dx, dy in product(range(3, min(self.size, max_size + 3)), allowed, range(4), range(4)):
                cx, cy = self.size // 2 + dx / 4, self.size // 2 + dy / 4
                mask = shape_mask(shape, cx, cy, side, self.size, angle)
                if not mask.any():
                    continue
                y, x, h, w = _bounds(np.argwhere(mask))
                bank[h, w].append((mask[y:y + h, x:x + w].copy(), shape, cx - x, cy - y, side))
        return bank

    def _aligned_templates(self, h: int, w: int) -> tuple:
        if (h, w) not in self._aligned:
            masks, ids, geometry = [], [], []
            for hh, ww in product(range(max(1, h - 1), h + 2), range(max(1, w - 1), w + 2)):
                dy, dx = h - hh, w - ww
                shifts = list(product(range(min(0, dy), max(0, dy) + 1), range(min(0, dx), max(0, dx) + 1)))
                for mask, shape, cx, cy, side in self.templates.get((hh, ww), ()):
                    for sy, sx in shifts:
                        canvas = np.zeros((h + 2, w + 2), dtype=bool)
                        canvas[1 + sy:1 + sy + hh, 1 + sx:1 + sx + ww] = mask
                        masks.append(canvas.ravel())
                        ids.append(shape)
                        geometry.append((cx + sx, cy + sy, side))
            bank = np.asarray(masks, dtype=bool).reshape(len(masks), (h + 2) * (w + 2))
            self._aligned[h, w] = bank, bank.sum(axis=1), np.asarray(ids), geometry
        return self._aligned[h, w]

    def _shape_fits(self, points: np.ndarray) -> tuple[np.ndarray, list[ObjectFit]]:
        y, x, h, w = _bounds(points)
        bank, areas, ids, geometry = self._aligned_templates(h, w)
        n_shapes = len(self.cfg["shapes"]["names"])
        scores = np.zeros(n_shapes, dtype=np.float32)
        fits = [ObjectFit(x + w / 2, y + h / 2, max(h, w))] * n_shapes
        if len(bank):
            pixels = (points[:, 0] - y + 1) * (w + 2) + points[:, 1] - x + 1
            intersections = bank[:, pixels].sum(axis=1)
            ious = intersections / (areas + len(points) - intersections)
            for shape in range(n_shapes):
                candidates = np.flatnonzero(ids == shape)
                if len(candidates):
                    best = int(candidates[ious[candidates].argmax()])
                    scores[shape] = ious[best]
                    cx, cy, side = geometry[best]
                    fits[shape] = ObjectFit(x + cx, y + cy, side)
        return scores, fits

    def parse(self, image: np.ndarray) -> ParsedScene:
        if image.shape != (self.size, self.size, 3):
            raise ValueError(f"oracle expects a {self.size}x{self.size} RGB image")
        rgb = image.astype(np.float32) / 255.0
        background = self.fit_background(rgb)
        delta = rgb - background
        residual = np.linalg.norm(delta, axis=-1)
        threshold = max(0.105, 2.8 * float(np.median(residual[self.border])) + 0.02)
        distances = np.linalg.norm(rgb[:, :, None] - self.palette, axis=-1)
        nearest = distances.argmin(axis=-1)
        nearest_dist = distances.min(axis=-1)
        core = (residual > threshold) & (nearest_dist < 0.20) & (nearest_dist + 0.06 < residual)
        votes = np.bincount(
            nearest[core], weights=residual[core] * (1 - nearest_dist[core] / 0.20),
            minlength=len(self.palette),
        )
        min_area = self.settings["min_component_area"]
        color = int(votes.argmax())
        if votes[color] <= 0.52 * votes.sum() or np.count_nonzero(core & (nearest == color)) < min_area:
            # Keep a source-only localization fallback, but do not report semantic
            # labels for a noisy image without a reliable foreground color.
            foreground = np.zeros_like(core)
            objects = []
            for points in components(residual > threshold, min_area):
                y, x, h, w = _bounds(points)
                foreground[points[:, 0], points[:, 1]] = True
                objects.append(ObjectFit(x + w / 2, y + h / 2, max(h, w)))
            return ParsedScene(None, None, None, tuple(objects), foreground)

        # Antialiased pixels lie on the RGB line from background to foreground.
        direction = self.palette[color] - background
        alpha = (delta * direction).sum(axis=-1) / np.maximum((direction ** 2).sum(axis=-1), 1e-6)
        line_error = np.linalg.norm(delta - alpha.clip(0, 1)[..., None] * direction, axis=-1)
        mask = (alpha >= 0.47) & (residual > threshold) & (line_error < threshold + 0.065)
        groups = components(mask, min_area)
        foreground = np.zeros_like(mask)
        if not groups:
            return ParsedScene(None, None, None, (), foreground)
        matches = [self._shape_fits(points) for points in groups]
        scores = np.stack([score for score, _ in matches]).mean(axis=0)
        shape = int(scores.argmax())
        confidence = float(scores[shape])
        margin = confidence - float(np.sort(scores)[-2])
        reliable = confidence >= self.settings["min_template_iou"] and margin >= 0.015
        fits = []
        for points, (score, candidates) in zip(groups, matches):
            foreground[points[:, 0], points[:, 1]] = True
            fits.append(candidates[shape if reliable else int(score.argmax())])
        return ParsedScene(len(groups), color, shape if reliable else None, tuple(fits), foreground)

    def region(self, parsed: ParsedScene) -> np.ndarray:
        centers = np.array([(obj.x, obj.y) for obj in parsed.objects], dtype=np.float32)
        sizes = np.array([obj.size for obj in parsed.objects], dtype=np.float32)
        boxes = bbox_mask(centers, sizes, len(sizes), self.size, self.settings["localization_pad"])
        return boxes.astype(bool) | parsed.foreground


def classification_metrics(targets: dict, predictions: dict, class_counts: tuple[int, ...]) -> dict:
    n = len(targets["count"])
    result = {"samples": n}
    joint = np.ones(n, dtype=bool)
    for name, classes in zip(ATTRIBUTES, class_counts):
        truth, pred = np.asarray(targets[name]), np.asarray(predictions[name])
        cm = np.zeros((classes, classes), dtype=np.int64)
        np.add.at(cm, (truth, pred), 1)
        denom = cm.sum(axis=0) + cm.sum(axis=1)
        f1 = np.divide(2 * cm.diagonal(), denom, out=np.zeros(classes), where=denom > 0)
        joint &= truth == pred
        result[name] = {
            "accuracy": float((truth == pred).mean()), "macro_f1": float(f1.mean()), "confusion_matrix": cm.tolist()
        }
    result["joint_accuracy"] = float(joint.mean())
    return result


class AttributeScores:
    def __init__(self):
        self.n = self.parsed = self.count_parsed = self.joint = 0
        self.correct = np.zeros(3, dtype=np.int64)
        self.count_error = 0

    def add(self, parsed: ParsedScene, target: tuple[int, int, int]) -> bool:
        self.n += 1
        self.parsed += parsed.parsed
        matches = [getattr(parsed, key) == value for key, value in zip(ATTRIBUTES, target)]
        self.correct += matches
        joint = all(matches)
        self.joint += joint
        if parsed.count is not None:
            self.count_parsed += 1
            self.count_error += abs(parsed.count - target[0])
        return joint

    def report(self) -> dict:
        n = self.n
        return {
            "samples": n, "parse_rate": self.parsed / n, "count_parse_rate": self.count_parsed / n,
            **{f"{key}_accuracy": int(value) / n for key, value in zip(ATTRIBUTES, self.correct)},
            "joint_accuracy": self.joint / n,
            "count_mae": self.count_error / self.count_parsed if self.count_parsed else None,
        }


class PixelScores:
    """Pixel metrics in [0,1]; abstain if the region has no pixels."""

    def __init__(self):
        self.images = self.pixels = self.exact = self.exact_images = 0
        self.absolute = self.squared = 0.0

    def add(self, image: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> None:
        error = (image[mask].astype(np.float32) - reference[mask].astype(np.float32)) / 255.0
        if len(error):
            exact = int(np.count_nonzero(np.all(error == 0, axis=1)))
            self.images += 1
            self.pixels += len(error)
            self.exact += exact
            self.exact_images += exact == len(error)
            self.absolute += np.abs(error).sum(dtype=np.float64).item()
            self.squared += np.square(error).sum(dtype=np.float64).item()

    def report(self) -> dict:
        n = self.pixels
        mse = self.squared / (3 * n) if n else None
        return {
            "pixels": n, "mae": self.absolute / (3 * n) if n else None, "mse": mse,
            "exact_match": self.exact / n if n else None,
            "image_exact_match": self.exact_images / self.images if self.images else None,
            "psnr_db": -10 * math.log10(mse) if mse else None,
        }


def localization_scores(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    overlap = int((predicted & target).sum())
    union = int((predicted | target).sum())
    return overlap / max(union, 1), overlap / max(int(target.sum()), 1), overlap / max(int(predicted.sum()), 1)


@cache
def _assignments(n: int) -> np.ndarray:
    return np.array(list(permutations(range(n))), dtype=np.intp)


class GeometryScores:
    def __init__(self):
        self.n = self.matched = self.objects = 0
        self.center_error = self.size_error = 0.0

    def add(self, parsed: ParsedScene, centers: np.ndarray, sizes: np.ndarray) -> None:
        self.n += 1
        n = len(sizes)
        if parsed.count != n or n == 0:
            return
        predicted = np.array([(obj.x, obj.y) for obj in parsed.objects])
        distance = np.linalg.norm(centers[:, None, :] - predicted[None, :, :], axis=-1)
        choices = _assignments(n)
        best = choices[np.argmin(distance[np.arange(n), choices].sum(axis=1))]
        self.center_error += distance[np.arange(n), best].sum().item()
        self.size_error += np.abs(sizes - np.array([parsed.objects[i].size for i in best])).sum().item()
        self.matched += 1
        self.objects += n

    def report(self) -> dict:
        return {
            "matched_scene_rate": self.matched / self.n,
            "center_mae_px": self.center_error / self.objects if self.objects else None,
            "size_mae_px": self.size_error / self.objects if self.objects else None,
        }
