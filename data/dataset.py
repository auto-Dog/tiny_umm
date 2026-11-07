"""Dataset access helpers (scene shards -> task views)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import (
    TASK_EDIT,
    TASK_GENERATE,
    TASK_UNDERSTAND,
)


def manifest_path(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["paths"]["data_dir"]) / "split_manifest.json"


def load_manifest(cfg: Dict[str, Any]) -> Dict[str, Any]:
    path = manifest_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run python -m data.build_dataset with the same config and smoke flag.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _concat_npz(paths: Iterable[Path]) -> Dict[str, np.ndarray]:
    parts: Dict[str, List[np.ndarray]] = {}
    for p in paths:
        with np.load(p) as z:
            for k in z.files:
                parts.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def load_group(cfg: Dict[str, Any], group: str) -> Dict[str, np.ndarray]:
    """Load one of train/val/test_random/test_comp into memory."""
    manifest = load_manifest(cfg)
    if group not in manifest["groups"]:
        raise KeyError(f"unknown data group {group!r}; have {list(manifest['groups'])}")
    data_dir = Path(cfg["paths"]["data_dir"])
    paths = [data_dir / name for name in manifest["groups"][group]["shards"]]
    if not paths:
        raise FileNotFoundError(f"no shards for {group} in {data_dir}")
    return _concat_npz(paths)


def normalize_images(images: np.ndarray) -> np.ndarray:
    """uint8 [N,28,28,3] -> float32 [-1,1]."""
    return (images.astype(np.float32) / 127.5) - 1.0


def denormalize_images(images: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(images, torch.Tensor):
        images = images.detach().float().cpu().numpy()
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 3:
        images = images[None]
    return np.clip(np.rint((images + 1.0) * 127.5), 0, 255).astype(np.uint8)


def task_labels(data: Dict[str, np.ndarray], task: str) -> Dict[str, np.ndarray]:
    """Return task-specific label mapping (count -> 0..4)."""
    if task == TASK_UNDERSTAND:
        return {
            "count": (data["counts"].astype(np.int64) - 1),
            "color": data["colors"].astype(np.int64),
            "shape": data["shapes"].astype(np.int64),
        }
    if task == TASK_GENERATE:
        return {
            "count": (data["counts"].astype(np.int64) - 1),
            "color": data["colors"].astype(np.int64),
            "shape": data["shapes"].astype(np.int64),
        }
    if task == TASK_EDIT:
        return {
            "count": (data["counts"].astype(np.int64) - 1),
            "color": data["edit_colors"].astype(np.int64),
            "shape": data["edit_shapes"].astype(np.int64),
        }
    raise ValueError(task)


def task_prompts(data: Dict[str, np.ndarray], task: str) -> np.ndarray:
    if task == TASK_UNDERSTAND:
        return data["prompts_und"]
    if task == TASK_GENERATE:
        return data["prompts_gen"]
    if task == TASK_EDIT:
        return data["prompts_edit"]
    raise ValueError(task)


class TaskView(Dataset):
    """A lightweight PyTorch view over a loaded scene group."""

    def __init__(self, data: Dict[str, np.ndarray], task: str):
        self.data = data
        self.task = task
        self.prompts = task_prompts(data, task)

    def __len__(self) -> int:
        return int(self.data["images"].shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        d = self.data
        out: Dict[str, Any] = {
            "image": torch.from_numpy(normalize_images(d["images"][idx])),
            "prompt": torch.from_numpy(self.prompts[idx]).long(),
            "count": int(d["counts"][idx]) - 1,
            "color": int(d["colors"][idx]) if self.task != TASK_EDIT else int(d["edit_colors"][idx]),
            "shape": int(d["shapes"][idx]) if self.task != TASK_EDIT else int(d["edit_shapes"][idx]),
            "scene_id": int(d["scene_ids"][idx]),
        }
        if self.task == TASK_EDIT:
            out.update({
                "edit_target": torch.from_numpy(normalize_images(d["edit_targets"][idx])),
                "src_mask": torch.from_numpy(d["src_masks"][idx]).bool(),
                "tgt_mask": torch.from_numpy(d["tgt_masks"][idx]).bool(),
                "region_mask": torch.from_numpy(d["region_masks"][idx]).bool(),
                "src_count": int(d["counts"][idx]) - 1,
                "src_color": int(d["colors"][idx]),
                "src_shape": int(d["shapes"][idx]),
                "edit_type": int(d["edit_types"][idx]),
            })
        return out


def batch_indices(rng: np.random.Generator, n: int, batch_size: int) -> Iterator[np.ndarray]:
    perm = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield perm[start:start + batch_size]


def take_images(data: Dict[str, np.ndarray], indices: np.ndarray) -> torch.Tensor:
    arr = normalize_images(data["images"][indices])
    return torch.from_numpy(arr)


def take_edit(data: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, torch.Tensor]:
    return {
        "src": torch.from_numpy(normalize_images(data["images"][indices])),
        "tgt": torch.from_numpy(normalize_images(data["edit_targets"][indices])),
        "prompt": torch.from_numpy(data["prompts_edit"][indices]).long(),
        "src_mask": torch.from_numpy(data["src_masks"][indices]).bool(),
        "tgt_mask": torch.from_numpy(data["tgt_masks"][indices]).bool(),
        "region_mask": torch.from_numpy(data["region_masks"][indices]).bool(),
        "count": torch.from_numpy(data["counts"][indices].astype(np.int64) - 1),
        "color": torch.from_numpy(data["edit_colors"][indices].astype(np.int64)),
        "shape": torch.from_numpy(data["edit_shapes"][indices].astype(np.int64)),
        "scene_id": torch.from_numpy(data["scene_ids"][indices]),
    }
