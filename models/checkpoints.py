"""Stage checkpoint names, validation, and model assembly."""
from __future__ import annotations

from pathlib import Path

import torch

from .bottleneck import ConditionalVisualBottleneck, SharedQueryBottleneck
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


def checkpoint_path(cfg: dict, kind: str) -> Path:
    return Path(cfg["paths"]["checkpoint_dir"]) / CHECKPOINTS[kind]


def _signature(cfg: dict) -> dict:
    return {key: cfg[key] for key in ("image_size", "counts", "colors", "shapes", "tokens", "model")}


def save_checkpoint(cfg: dict, kind: str, state: dict, epoch: int, score: float, **metadata) -> Path:
    path = checkpoint_path(cfg, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save({
        "version": 1, "kind": kind, "signature": _signature(cfg),
        "epoch": epoch, "validation_score": score, "state": state, **metadata,
    }, temporary)
    temporary.replace(path)
    return path


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
