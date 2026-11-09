"""Recipes for the two sequence baselines, layered on the existing dataset config."""
from __future__ import annotations

from data.schema import deep_merge

BASE_MODELS = ("emu3", "bagel")


def base_recipe(cfg: dict, model: str) -> dict:
    if model not in BASE_MODELS:
        raise ValueError(f"unknown base model {model!r}")
    flow = cfg["training"]["flow"]
    recipe = {
        "model": {"width": 128, "layers": 4, "heads": 4, "patch_size": 4},
        "training": {"epochs": 40, "batch_size": flow["batch_size"], "lr": flow["lr"]},
        "sampling": {key: cfg["sampling"][key] for key in ("steps", "solver")},
    }
    if model == "emu3":
        recipe["model"]["codebook_size"] = 512
        recipe["training"].update(codebook_samples=65_536, kmeans_steps=20)
        recipe["sampling"] = {"temperature": 0.9, "top_k": 32}
    if cfg.get("_smoke"):
        recipe["model"].update(width=64, layers=2)
        recipe["training"].update(epochs=2, batch_size=32)
        if model == "emu3":
            recipe["model"]["codebook_size"] = 128
            recipe["training"].update(codebook_samples=8192, kmeans_steps=6)
    # Optional YAML section: base_models.<emu3|bagel>.{model,training,sampling}.
    return deep_merge(recipe, cfg.get("base_models", {}).get(model, {}))
