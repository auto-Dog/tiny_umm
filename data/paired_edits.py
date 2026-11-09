"""Render a second valid source appearance for each training edit."""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from .build_dataset import heldout_set
from .dataset import load_group
from .renderer import _validate_masks, make_background, rerender_from_scene
from .schema import load_config, palette


SCENE_FIELDS = {
    "count": "counts", "centers": "centers", "sizes": "sizes", "angles": "angles",
    "bg_color": "bg_colors", "bg_type": "bg_types", "bg_dir": "bg_dirs",
    "bg_f0": "bg_f0", "bg_f1": "bg_f1", "bg_checker_amp": "bg_checker_amp",
    "bg_checker_cell": "bg_checker_cell", "bg_checker_phase": "bg_checker_phase",
    "noise_type": "noise_types", "noise_sigma": "noise_sigmas",
    "noise_fraction": "noise_fractions", "noise_seed": "noise_seeds",
}


def source_fingerprint(data: dict) -> str:
    digest = sha256()
    for name in ("scene_ids", "images", "edit_targets", "prompts_edit"):
        digest.update(memoryview(np.ascontiguousarray(data[name])).cast("B"))
    return digest.hexdigest()


def read_pairs(path: Path, data: dict) -> np.ndarray:
    with np.load(path, allow_pickle=False) as saved:
        images = saved["images"]
        if (
            saved["source_sha256"].item() != source_fingerprint(data)
            or images.shape != data["images"].shape
            or images.dtype != np.uint8
        ):
            raise ValueError(f"{path} does not match the current training scenes")
    return images


def render_pairs(cfg: dict, data: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    colors = palette(cfg)
    min_contrast = cfg["data"]["background"]["min_fg_bg_distance"] / 255
    held = heldout_set(cfg)
    images = data["images"].copy()
    attributes = np.stack((data["colors"], data["shapes"]), axis=1)
    for i, (src, tgt) in enumerate(zip(attributes.copy(), data["prompts_edit"][:, 2:])):
        scene = {key: data[name][i] for key, name in SCENE_FIELDS.items()}
        n = int(scene["count"])
        background = make_background(
            cfg["image_size"], colors[scene["bg_color"]], scene["bg_type"], scene["bg_dir"],
            scene["bg_f0"], scene["bg_f1"], scene["bg_checker_amp"], scene["bg_checker_cell"],
            scene["bg_checker_phase"],
        )
        contrast = np.linalg.norm(background[:, :, None] - colors, axis=-1).min(axis=(0, 1))
        shapes = [
            s for s in range(len(cfg["shapes"]["names"]))
            if _validate_masks(cfg, s, scene["centers"], scene["sizes"], n, scene["angles"])
        ]
        original, target = tuple(src), tuple(tgt)
        options = [
            (c, s) for c in np.flatnonzero(contrast >= min_contrast) for s in shapes
            if (c, s) not in (original, target) and (n, c, s) not in held
        ]
        # Preserve the edit type when the scene supports another appearance of that type.
        same_type = [
            (c, s) for c, s in options
            if (c == target[0], s == target[1]) == (src[0] == tgt[0], src[1] == tgt[1])
        ]
        candidates = same_type or options
        if candidates:
            c, s = candidates[rng.integers(len(candidates))]
            image, mask = rerender_from_scene(cfg, scene, int(s), int(c))
            outside = ~(mask.astype(bool) | data["src_masks"][i].astype(bool))
            image[outside] = images[i, outside]
            images[i] = image
            attributes[i] = c, s
        if (i + 1) % 5000 == 0:
            print(f"Rendered {i + 1}/{len(images)} training views", flush=True)
    return images, attributes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", default=731, type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = load_group(cfg, "train")
    images, attributes = render_pairs(cfg, data, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as output:
        np.savez(
            output, images=images, attributes=attributes, source_sha256=source_fingerprint(data), seed=args.seed,
        )
    distinct = np.any(attributes != np.stack((data["colors"], data["shapes"]), axis=1), axis=1).sum()
    print(f"Distinct second views: {distinct}/{len(images)}; saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
