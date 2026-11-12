"""Checkpoint IO for the TinyUMM stages and the sequence baselines."""
from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from configs.base_models import base_recipe

from .bagel import TinyBagel as BagelBase
from .bottleneck import ConditionalVisualBottleneck, SharedQueryBottleneck
from .emu3 import TinyEmu3 as Emu3Base
from .spatial_codec import SpatialCodec
from .tiny_umm import TinyUMM
from .understanding import UnderstandingMLP

CHECKPOINTS = {
    "understanding": "understanding.pt",
    "flow": "flow.pt",
    "connector": "connected.pt",
    "joint": "connected_joint.pt",
    "discrete": "discrete.pt",
    "adapter": "adapted.pt",
    "posttrain": "posttrained.pt",
    "shared": "shared.pt",
}
FLOW_MODULES = ("flow", "instruction", "source_encoder")
FLOW_KINDS = ("flow", "connector", "joint", "discrete", "adapter", "posttrain", "shared")


def _atomic_save(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=f"{path.suffix}.tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


# Stage checkpoints for the shared TinyUMM pipeline.


def checkpoint_path(cfg: dict, kind: str) -> Path:
    return Path(cfg["paths"]["checkpoint_dir"]) / CHECKPOINTS[kind]


def _signature(cfg: dict) -> dict:
    return {key: cfg[key] for key in ("image_size", "counts", "colors", "shapes", "tokens", "model")}


def save_checkpoint(cfg: dict, kind: str, state: dict, epoch: int, score: float, **metadata) -> Path:
    return _atomic_save(checkpoint_path(cfg, kind), {
        "version": 1, "kind": kind, "signature": _signature(cfg),
        "epoch": epoch, "validation_score": score, "state": state, **metadata,
    })


def read_checkpoint(cfg: dict, path: str | Path, kinds: tuple[str, ...]) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}. Run the corresponding training stage with the same config.")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("version") != 1 or saved.get("kind") not in kinds:
        raise ValueError(f"{path} is not a compatible {'/'.join(kinds)} checkpoint")
    if saved["signature"] != _signature(cfg):
        raise ValueError(f"{path} does not match the model or token schema in this configuration")
    return saved


def load_understanding(
    cfg: dict, device: torch.device, path: str | Path | None = None,
) -> UnderstandingMLP | TinyUMM:
    path = path or checkpoint_path(cfg, "understanding")
    saved = read_checkpoint(cfg, path, ("understanding", "shared"))
    if saved["kind"] == "shared":
        return _restore_flow(cfg, path, device, saved)
    model = UnderstandingMLP(cfg)
    model.load_state_dict(saved["state"])
    return model.to(device).eval()


def flow_state(model: TinyUMM, connected: bool = False) -> dict:
    names = FLOW_MODULES + ("understanding", "bridge") if connected else FLOW_MODULES
    if connected and model.bottleneck is not None:
        names += ("bottleneck",)
    if connected and model.text_aux is not None:
        names += ("text_aux",)
    return {name: getattr(model, name).state_dict() for name in names}


def apply_flow_state(model: TinyUMM, state: dict) -> None:
    for name, weights in state.items():
        getattr(model, name).load_state_dict(weights)


def load_flow(cfg: dict, path: str | Path, device: torch.device) -> tuple[TinyUMM, dict]:
    saved = read_checkpoint(cfg, path, FLOW_KINDS)
    return _restore_flow(cfg, path, device, saved), saved


def _restore_flow(cfg: dict, path: str | Path, device: torch.device, saved: dict) -> TinyUMM:
    kind = saved["kind"]
    bridge = None if kind == "flow" else ("discrete" if kind == "discrete" else "connector")
    bottleneck_kind = saved.get("bottleneck_kind")
    has_bottleneck = "bottleneck" in saved["state"]
    if bottleneck_kind not in (None, "conditional", "shared_query") or (bottleneck_kind and not has_bottleneck):
        raise ValueError(f"{path} has incompatible bottleneck metadata")
    shared = kind == "shared"
    if shared != ("text_aux" in saved["state"]) or (shared and saved.get("direct_source") is not False):
        raise ValueError(f"{path} has incompatible shared query metadata")
    if (shared and has_bottleneck and bottleneck_kind != "shared_query") or (not shared and bottleneck_kind == "shared_query"):
        raise ValueError(f"{path} has incompatible shared query bottleneck metadata")
    model = TinyUMM(cfg, bridge, bottleneck=has_bottleneck and bottleneck_kind is None, shared_understanding=shared)
    if bottleneck_kind == "conditional":
        options = saved.get("posttrain", {})
        model.bottleneck = ConditionalVisualBottleneck(
            cfg, max_std_ratio=options.get("max_std_ratio", 0.30), appearance_rank=options.get("appearance_rank", 0),
            protected_tokens=options.get("protected_tokens", 0),
        )
    if bottleneck_kind == "shared_query":
        model.bottleneck = SharedQueryBottleneck(saved["posttrain"]["posterior_std"])
    apply_flow_state(model, saved["state"])
    return model.to(device).eval()


# Independent sequence baselines (Tiny Emu3 / BAGEL on RGB patches).

BASE_TYPES = {"emu3": Emu3Base, "bagel": BagelBase}


def base_checkpoint_path(cfg: dict, kind: str) -> Path:
    return Path(cfg["paths"]["checkpoint_dir"]) / "base_models" / f"{kind}.pt"


def _base_signature(cfg: dict, kind: str) -> dict:
    return {
        **{key: cfg[key] for key in ("image_size", "counts", "colors", "shapes", "tokens")},
        "model": base_recipe(cfg, kind)["model"],
    }


def save_base_checkpoint(cfg: dict, kind: str, model: Emu3Base | BagelBase, epoch: int, score: float) -> Path:
    return _atomic_save(base_checkpoint_path(cfg, kind), {
        "version": 1, "kind": kind, "signature": _base_signature(cfg, kind),
        "epoch": epoch, "validation_score": score, "state": model.state_dict(),
    })


def load_base_checkpoint(
    cfg: dict, kind: str, device: torch.device, path: str | Path | None = None,
) -> Emu3Base | BagelBase:
    path = Path(path) if path is not None else base_checkpoint_path(cfg, kind)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}. Run python -m training.train_base --model {kind}.")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("version") != 1 or saved.get("kind") != kind:
        raise ValueError(f"{path} is not a compatible {kind} checkpoint")
    if saved.get("signature") != _base_signature(cfg, kind):
        raise ValueError(f"{path} does not match the model or token schema in this configuration")
    model = BASE_TYPES[kind](cfg, saved["signature"]["model"])
    model.load_state_dict(saved["state"])
    return model.to(device).eval()


# Latent Tiny Emu3 / BAGEL checkpoints (spatial codec plus sequence model).


def save_latent_checkpoint(path: Path, module: torch.nn.Module, **metadata) -> None:
    _atomic_save(Path(path), {"version": 2, "state": module.state_dict(), **metadata})


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
