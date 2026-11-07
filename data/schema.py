"""Shared configuration, token schema, and small utilities."""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

OP_UNDERSTAND = 0
OP_GENERATE = 1
OP_EDIT = 2
OP_NAMES = ("understand", "generate", "edit")

TASK_UNDERSTAND = "understand"
TASK_GENERATE = "generate"
TASK_EDIT = "edit"
TASKS = (TASK_UNDERSTAND, TASK_GENERATE, TASK_EDIT)


def load_config(path: str | os.PathLike = "configs/default.yaml", smoke: bool = False) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if smoke:
        cfg = deep_merge(cfg, cfg.get("smoke", {}))
    cfg["_config_path"] = str(path)
    cfg["_root"] = str(ROOT)
    cfg["_smoke"] = bool(smoke)
    # Resolve artifact paths against the project root, not the caller's cwd.
    for key in ("data_dir", "checkpoint_dir", "output_dir"):
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            p = ROOT / p
        cfg["paths"][key] = str(p)
    return cfg


def deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: Dict[str, Any] | None = None) -> torch.device:
    requested = "auto" if cfg is None else str(cfg.get("training", {}).get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def palette(cfg: Dict[str, Any]) -> np.ndarray:
    return np.asarray(cfg["colors"]["rgb"], dtype=np.float32) / 255.0


def palette_u8(cfg: Dict[str, Any]) -> np.ndarray:
    return np.asarray(cfg["colors"]["rgb"], dtype=np.uint8)


def color_name(cfg: Dict[str, Any], idx: int) -> str:
    return cfg["colors"]["names"][int(idx)]


def shape_name(cfg: Dict[str, Any], idx: int) -> str:
    return cfg["shapes"]["names"][int(idx)]


def count_token(cfg: Dict[str, Any], n: int) -> int:
    """Raw count token: 1..5 for counts, KEEP for edit, MASK for understanding."""
    if 1 <= int(n) <= len(cfg["counts"]):
        return int(n)
    raise ValueError(f"count {n} outside configured counts")


def make_prompt(cfg: Dict[str, Any], op: int, count_tok: int, color_tok: int, shape_tok: int) -> np.ndarray:
    return np.asarray([op, count_tok, color_tok, shape_tok], dtype=np.uint8)


def understand_prompt(cfg: Dict[str, Any]) -> np.ndarray:
    t = cfg["tokens"]
    return make_prompt(cfg, OP_UNDERSTAND, int(t["count_mask"]), int(t["color_mask"]), int(t["shape_mask"]))


def generate_prompt(cfg: Dict[str, Any], n: int, color: int, shape: int) -> np.ndarray:
    return make_prompt(cfg, OP_GENERATE, count_token(cfg, n), int(color), int(shape))


def edit_prompt(cfg: Dict[str, Any], target_color: int, target_shape: int) -> np.ndarray:
    t = cfg["tokens"]
    return make_prompt(cfg, OP_EDIT, int(t["count_keep"]), int(target_color), int(target_shape))


def prompt_to_text(cfg: Dict[str, Any], tokens: Iterable[int]) -> str:
    tok = [int(x) for x in tokens]
    t = cfg["tokens"]
    op = OP_NAMES[tok[0]] if 0 <= tok[0] < 3 else f"bad_op({tok[0]})"
    if tok[1] == int(t["count_mask"]):
        c = "MASK"
    elif tok[1] == int(t["count_keep"]):
        c = "KEEP"
    else:
        c = str(tok[1])
    col = "MASK" if tok[2] == int(t["color_mask"]) else color_name(cfg, tok[2])
    sha = "MASK" if tok[3] == int(t["shape_mask"]) else shape_name(cfg, tok[3])
    return f"[{op}, {c}, {col}, {sha}]"


def labels_from_counts(counts: np.ndarray) -> np.ndarray:
    """Map raw counts 1..5 to head classes 0..4."""
    return (np.asarray(counts, dtype=np.int64) - 1).astype(np.int64)


def counts_from_labels(labels: np.ndarray) -> np.ndarray:
    return (np.asarray(labels, dtype=np.int64) + 1).astype(np.int64)


def save_json(path: str | os.PathLike, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_task_counts(n_scenes: int) -> Dict[str, int]:
    return {task: n_scenes for task in TASKS}


def format_metric(value: float) -> str:
    return f"{100.0 * value:.2f}%"
