"""Versioned, self-contained checkpoints for Tiny Emu3 and BAGEL."""
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from .spatial_codec import SpatialCodec


def save_latent_checkpoint(path: Path, module: torch.nn.Module, **metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save({"version": 2, "state": module.state_dict(), **metadata}, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_codec(path: Path, device: torch.device) -> tuple[SpatialCodec, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("version") != 2 or saved.get("kind") != "codec":
        raise ValueError(f"incompatible spatial codec checkpoint {path}")
    codec = SpatialCodec(saved["image_size"], **saved["settings"])
    codec.load_state_dict(saved["state"])
    return codec.to(device).eval(), saved


def load_latent_model(path: Path, device: torch.device):
    from .emu3_bagel import TinyBagel, TinyEmu3

    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("version") != 2 or saved.get("kind") not in ("emu3", "bagel"):
        raise ValueError(f"incompatible Tiny Emu3/BAGEL checkpoint {path}")
    cfg, recipe = saved["config"], saved["recipe"]
    codec = SpatialCodec(cfg["image_size"], **recipe["codec"])
    model_type = TinyEmu3 if saved["kind"] == "emu3" else TinyBagel
    model = model_type(cfg, recipe, codec)
    model.load_state_dict(saved["state"])
    return model.to(device).eval(), saved
