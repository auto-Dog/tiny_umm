"""Build the scene-level 28x28 dataset and task views.

Storage is scene-oriented: each scene yields one understand, one generate and
one edit task.  The 3 tasks stay in the same split by construction.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .renderer import _supersample_from_cfg, sample_scene
from .schema import (
    edit_prompt,
    ensure_dir,
    generate_prompt,
    load_config,
    save_json,
    set_seed,
    understand_prompt,
)

GROUP_SPECS = {
    "train": {"split": "train", "split_code": 0, "scene_offset": 0, "is_comp": 0},
    "val": {"split": "val", "split_code": 1, "scene_offset": 1_000_000, "is_comp": 0},
    "test_random": {"split": "test", "split_code": 2, "scene_offset": 2_000_000, "is_comp": 0},
    "test_comp": {"split": "test", "split_code": 2, "scene_offset": 3_000_000, "is_comp": 1},
}


def all_combos(cfg: Dict[str, Any]) -> List[Tuple[int, int, int]]:
    return [(n, c, s)
            for n in cfg["counts"]
            for c in range(len(cfg["colors"]["names"]))
            for s in range(len(cfg["shapes"]["names"]))]


def heldout_set(cfg: Dict[str, Any]) -> set:
    return {tuple(int(v) for v in row) for row in cfg["data"]["heldout_combos"]}


def combos_for_group(cfg: Dict[str, Any], group: str) -> List[Tuple[int, int, int]]:
    held = heldout_set(cfg)
    combos = [c for c in all_combos(cfg) if c not in held]
    if group == "test_comp":
        return [tuple(int(v) for v in row) for row in cfg["data"]["heldout_combos"]]
    return combos


def balanced_sequence(combos: List[Tuple[int, int, int]], n: int, rng: np.random.Generator) -> List[Tuple[int, int, int]]:
    if n > 0 and not combos:
        raise ValueError("cannot sample scenes from an empty combination set")
    out: List[Tuple[int, int, int]] = []
    combo_arr = np.asarray(combos, dtype=np.int64)
    while len(out) < n:
        perm = rng.permutation(len(combo_arr))
        out.extend(tuple(int(v) for v in combo_arr[i]) for i in perm)
    return out[:n]


def _add_scene(acc: Dict[str, list], scene: Dict[str, Any], cfg: Dict[str, Any], group: str) -> None:
    acc["images"].append(scene["image"])
    acc["edit_targets"].append(scene["edit_target"])
    acc["counts"].append(scene["count"])
    acc["colors"].append(scene["color"])
    acc["shapes"].append(scene["shape"])
    acc["edit_colors"].append(scene["edit_color"])
    acc["edit_shapes"].append(scene["edit_shape"])
    acc["edit_types"].append(scene["edit_type"])
    acc["src_masks"].append(scene["src_mask"])
    acc["tgt_masks"].append(scene["tgt_mask"])
    acc["region_masks"].append(scene["region_mask"])
    centers = np.zeros((len(cfg["counts"]), 2), dtype=np.float32)
    sizes = np.zeros((len(cfg["counts"]),), dtype=np.float32)
    angles = np.zeros((len(cfg["counts"]),), dtype=np.float32)
    n = int(scene["count"])
    centers[:n] = scene["centers"]
    sizes[:n] = scene["sizes"]
    angles[:n] = scene.get("angles", np.zeros(n, dtype=np.float32))
    acc["centers"].append(centers)
    acc["sizes"].append(sizes)
    acc["angles"].append(angles)
    acc["bg_colors"].append(scene["bg_color"])
    acc["bg_types"].append(scene["bg_type"])
    acc["bg_dirs"].append(scene["bg_dir"])
    acc["bg_f0"].append(scene["bg_f0"])
    acc["bg_f1"].append(scene["bg_f1"])
    acc["bg_checker_amp"].append(scene["bg_checker_amp"])
    acc["bg_checker_cell"].append(scene["bg_checker_cell"])
    acc["bg_checker_phase"].append(scene["bg_checker_phase"])
    acc["noise_types"].append(scene["noise_type"])
    acc["noise_sigmas"].append(scene["noise_sigma"])
    acc["noise_fractions"].append(scene["noise_fraction"])
    acc["noise_seeds"].append(scene["noise_seed"])
    acc["scene_ids"].append(scene["scene_id"])
    acc["split_codes"].append(GROUP_SPECS[group]["split_code"])
    acc["is_comp"].append(scene["is_comp"])
    acc["prompts_und"].append(understand_prompt(cfg))
    acc["prompts_gen"].append(generate_prompt(cfg, scene["count"], scene["color"], scene["shape"]))
    acc["prompts_edit"].append(edit_prompt(cfg, scene["edit_color"], scene["edit_shape"]))
    arrangement_names = list(cfg["data"]["layout"]["arrangement_probs"].keys())
    acc["arrangements"].append(arrangement_names.index(scene["arrangement"]))


def _new_acc() -> Dict[str, list]:
    return {k: [] for k in [
        "images", "edit_targets", "counts", "colors", "shapes", "edit_colors", "edit_shapes",
        "edit_types", "src_masks", "tgt_masks", "region_masks", "centers", "sizes",
        "angles", "bg_colors", "bg_types", "bg_dirs", "bg_f0", "bg_f1", "bg_checker_amp",
        "bg_checker_cell", "bg_checker_phase", "noise_types", "noise_sigmas",
        "noise_fractions", "noise_seeds", "scene_ids", "split_codes", "is_comp",
        "prompts_und", "prompts_gen", "prompts_edit", "arrangements",
    ]}


def _stack_acc(acc: Dict[str, list]) -> Dict[str, np.ndarray]:
    return {
        "images": np.stack(acc["images"], axis=0).astype(np.uint8),
        "edit_targets": np.stack(acc["edit_targets"], axis=0).astype(np.uint8),
        "counts": np.asarray(acc["counts"], dtype=np.uint8),
        "colors": np.asarray(acc["colors"], dtype=np.uint8),
        "shapes": np.asarray(acc["shapes"], dtype=np.uint8),
        "edit_colors": np.asarray(acc["edit_colors"], dtype=np.uint8),
        "edit_shapes": np.asarray(acc["edit_shapes"], dtype=np.uint8),
        "edit_types": np.asarray(acc["edit_types"], dtype=np.uint8),
        "src_masks": np.stack(acc["src_masks"], axis=0).astype(np.uint8),
        "tgt_masks": np.stack(acc["tgt_masks"], axis=0).astype(np.uint8),
        "region_masks": np.stack(acc["region_masks"], axis=0).astype(np.uint8),
        "centers": np.stack(acc["centers"], axis=0).astype(np.float32),
        "sizes": np.stack(acc["sizes"], axis=0).astype(np.float32),
        "angles": np.stack(acc["angles"], axis=0).astype(np.float32),
        "bg_colors": np.asarray(acc["bg_colors"], dtype=np.uint8),
        "bg_types": np.asarray(acc["bg_types"], dtype=np.uint8),
        "bg_dirs": np.asarray(acc["bg_dirs"], dtype=np.uint8),
        "bg_f0": np.asarray(acc["bg_f0"], dtype=np.float32),
        "bg_f1": np.asarray(acc["bg_f1"], dtype=np.float32),
        "bg_checker_amp": np.asarray(acc["bg_checker_amp"], dtype=np.float32),
        "bg_checker_cell": np.asarray(acc["bg_checker_cell"], dtype=np.uint8),
        "bg_checker_phase": np.asarray(acc["bg_checker_phase"], dtype=np.uint8),
        "noise_types": np.asarray(acc["noise_types"], dtype=np.uint8),
        "noise_sigmas": np.asarray(acc["noise_sigmas"], dtype=np.float32),
        "noise_fractions": np.asarray(acc["noise_fractions"], dtype=np.float32),
        "noise_seeds": np.asarray(acc["noise_seeds"], dtype=np.int64),
        "scene_ids": np.asarray(acc["scene_ids"], dtype=np.int64),
        "split_codes": np.asarray(acc["split_codes"], dtype=np.uint8),
        "is_comp": np.asarray(acc["is_comp"], dtype=np.uint8),
        "prompts_und": np.stack(acc["prompts_und"], axis=0).astype(np.uint8),
        "prompts_gen": np.stack(acc["prompts_gen"], axis=0).astype(np.uint8),
        "prompts_edit": np.stack(acc["prompts_edit"], axis=0).astype(np.uint8),
        "arrangements": np.asarray(acc["arrangements"], dtype=np.uint8),
    }


def build_group(cfg: Dict[str, Any], group: str, n_scenes: int, shard_size: int,
                out_dir: Path, seed_offset: int) -> List[str]:
    spec = GROUP_SPECS[group]
    combos = combos_for_group(cfg, group)
    held = heldout_set(cfg)
    rng = np.random.default_rng(int(cfg["seed"]) + seed_offset)
    seq = balanced_sequence(combos, n_scenes, rng)
    shard_files: List[str] = []
    rejection = Counter()
    edit_types = Counter()
    t0 = time.time()
    for shard_idx, start in enumerate(range(0, n_scenes, shard_size)):
        end = min(n_scenes, start + shard_size)
        acc = _new_acc()
        for global_idx in range(start, end):
            n, color, shape = seq[global_idx]
            scene = None
            for attempt in range(80):
                cand = sample_scene(
                    cfg, rng, count=n, color=color, shape=shape,
                    scene_id=int(spec["scene_offset"] + global_idx),
                    split=spec["split"], is_comp=spec["is_comp"],
                )
                if cand is None:
                    rejection["sample"] += 1
                    continue
                # Do not leak held-out combinations through edit target attributes.
                if spec["is_comp"] == 0 and (cand["count"], cand["edit_color"], cand["edit_shape"]) in held:
                    rejection["heldout_edit_target"] += 1
                    continue
                scene = cand
                break
            if scene is None:
                raise RuntimeError(f"failed to build scene {group}:{global_idx} attrs={(n, color, shape)}")
            _add_scene(acc, scene, cfg, group)
            edit_types[cfg["data"]["edit"]["types"][scene["edit_type"]]] += 1
            if (global_idx + 1) % 1000 == 0 or global_idx + 1 == end:
                elapsed = time.time() - t0
                done = global_idx + 1
                print(f"[{group}] {done}/{n_scenes} scenes, {elapsed:.1f}s, "
                      f"reject={dict(rejection)}", flush=True)
        arrays = _stack_acc(acc)
        path = out_dir / f"{group}_shard_{shard_idx:03d}.npz"
        np.savez(path, **arrays)
        shard_files.append(path.name)
        print(f"[{group}] wrote {path.name}: {arrays['images'].shape[0]} scenes, "
              f"edit types={dict(edit_types)}", flush=True)
    return shard_files


def make_preview(cfg: Dict[str, Any], data_dir: Path, out_path: Path, n: int = 12) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(data_dir.glob("train_shard_*.npz"))
    if not files:
        return
    with np.load(files[0]) as z:
        m = min(n, z["images"].shape[0])
        rng = np.random.default_rng(123)
        idx = rng.choice(z["images"].shape[0], size=m, replace=False)
        rows = int(math.ceil(m / 2))
        fig, axes = plt.subplots(rows, 4, figsize=(8, 2.2 * rows))
        axes = np.asarray(axes).reshape(rows, 4)
        for r in range(rows):
            for col_pair in range(2):
                j = r * 2 + col_pair
                ax0, ax1 = axes[r, 2 * col_pair], axes[r, 2 * col_pair + 1]
                if j >= m:
                    ax0.axis("off"); ax1.axis("off"); continue
                i = int(idx[j])
                ax0.imshow(z["images"][i]); ax0.axis("off")
                ax1.imshow(z["edit_targets"][i]); ax1.axis("off")
                src = f"n={z['counts'][i]} {cfg['colors']['names'][z['colors'][i]]} {cfg['shapes']['names'][z['shapes'][i]]}"
                tgt = f"edit -> {cfg['colors']['names'][z['edit_colors'][i]]} {cfg['shapes']['names'][z['edit_shapes'][i]]}"
                ax0.set_title("source\n" + src, fontsize=7)
                ax1.set_title("target\n" + tgt, fontsize=7)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Save to a fresh path, then atomically replace.  Overwriting an
        # existing PNG directly has failed with EINVAL on this filesystem
        # (likely a transient scanner lock on the destination).
        tmp_path = out_path.with_name(out_path.stem + ".tmp.png")
        fig.savefig(tmp_path, dpi=160)
        plt.close(fig)
        tmp_path.replace(out_path)


def build(cfg: Dict[str, Any]) -> Dict[str, Any]:
    counts = cfg["data"]["scene_counts"]
    shard_size = int(cfg["data"]["shard_size"])
    combos = heldout_set(cfg)
    if (not combos or len(combos) != len(cfg["data"]["heldout_combos"])
            or not combos < set(all_combos(cfg))):
        raise ValueError("heldout_combos must be unique and form a nonempty proper subset")
    if shard_size <= 0 or any(int(counts[group]) <= 0 for group in GROUP_SPECS):
        raise ValueError("shard_size and all scene counts must be positive")
    data_dir = ensure_dir(cfg["paths"]["data_dir"])
    print("Building dataset:", json.dumps(counts, ensure_ascii=False), flush=True)
    all_shards = {}
    for group, off in (("train", 0), ("val", 1), ("test_random", 2), ("test_comp", 3)):
        files = build_group(cfg, group, int(counts[group]), shard_size, data_dir, off * 10_000)
        all_shards[group] = files

    total_scenes = sum(int(v) for v in counts.values())
    task_counts = {"understand": total_scenes, "generate": total_scenes, "edit": total_scenes}
    manifest = {
        "version": 1,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(cfg["seed"]),
        "image_size": int(cfg["image_size"]),
        "counts": [int(x) for x in cfg["counts"]],
        "colors": cfg["colors"]["names"],
        "shapes": cfg["shapes"]["names"],
        "prompt_schema": {
            "length": 4,
            "order": ["operation", "count", "color", "shape"],
            "ask": "understand = [UND, MASK, MASK, MASK]",
            "generate": "generate = [GEN, n, color, shape]",
            "edit": "edit = [EDIT, KEEP, target_color, target_shape]",
        },
        "heldout_combos": [list(map(int, row)) for row in cfg["data"]["heldout_combos"]],
        "render": {
            "supersample": _supersample_from_cfg(cfg),
            "rotation": {
                "enabled": bool(cfg["data"].get("rotation", {}).get("enabled", True)),
                "num_angles": int(cfg["data"].get("rotation", {}).get("num_angles", 12)),
            },
        },
        "task_counts": task_counts,
        "total_tasks": sum(task_counts.values()),
        "groups": {
            group: {
                "split": GROUP_SPECS[group]["split"],
                "is_comp": GROUP_SPECS[group]["is_comp"],
                "scenes": int(counts[group]),
                "tasks": {task: int(counts[group]) for task in ("understand", "generate", "edit")},
                "shards": files,
            }
            for group, files in all_shards.items()
        },
        "shard_size": shard_size,
        "config_path": cfg.get("_config_path", ""),
        "smoke": bool(cfg.get("_smoke", False)),
    }
    save_json(data_dir / "split_manifest.json", manifest)
    make_preview(cfg, data_dir, data_dir / "preview.png")
    print(f"Done: {total_scenes} scenes, {sum(task_counts.values())} task samples -> {data_dir}", flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    set_seed(int(cfg["seed"]))
    build(cfg)


if __name__ == "__main__":
    main()
