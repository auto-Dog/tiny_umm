"""Construction and checkpoint IO for the independent sequence baselines."""
from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from configs.base_models import base_recipe

from .bagel import TinyBagel
from .emu3 import TinyEmu3

BASE_TYPES = {"emu3": TinyEmu3, "bagel": TinyBagel}


def base_checkpoint_path(cfg: dict, kind: str) -> Path:
    return Path(cfg["paths"]["checkpoint_dir"]) / "base_models" / f"{kind}.pt"


def _signature(cfg: dict, kind: str) -> dict:
    return {
        **{key: cfg[key] for key in ("image_size", "counts", "colors", "shapes", "tokens")},
        "model": base_recipe(cfg, kind)["model"],
    }


def save_base_checkpoint(cfg: dict, kind: str, model: TinyEmu3 | TinyBagel, epoch: int, score: float) -> Path:
    path = base_checkpoint_path(cfg, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".pt.tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save({
            "version": 1, "kind": kind, "signature": _signature(cfg, kind),
            "epoch": epoch, "validation_score": score, "state": model.state_dict(),
        }, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_base_checkpoint(
    cfg: dict, kind: str, device: torch.device, path: str | Path | None = None,
) -> TinyEmu3 | TinyBagel:
    path = Path(path) if path is not None else base_checkpoint_path(cfg, kind)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}. Run python -m training.train_base --model {kind}.")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("version") != 1 or saved.get("kind") != kind:
        raise ValueError(f"{path} is not a compatible {kind} checkpoint")
    if saved.get("signature") != _signature(cfg, kind):
        raise ValueError(f"{path} does not match the model or token schema in this configuration")
    model = BASE_TYPES[kind](cfg, saved["signature"]["model"])
    model.load_state_dict(saved["state"])
    return model.to(device).eval()
