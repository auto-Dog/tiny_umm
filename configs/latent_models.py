"""Independent recipes for the Tiny Emu3 and BAGEL implementations."""
from data.schema import deep_merge


def latent_recipe(cfg: dict) -> dict:
    recipe = {
        "codec": {"width": 32, "latent_dim": 16, "codebook_size": 1024},
        "codec_training": {"warmup_epochs": 15, "quantized_epochs": 15, "batch_size": 128, "lr": 0.001},
        "model": {"width": 192, "layers": 4, "heads": 6, "patch_size": 4, "codebook_size": 1024},
        "training": {"epochs": 40, "batch_size": 96, "lr": 0.0003, "foreground_weight": 2.0, "condition_dropout": 0.10, "decoded_loss_weight": 0.25},
        "sampling": {"emu3": {"temperature": 0.85, "top_k": 64, "guidance": 1.5}, "bagel": {"steps": 40, "solver": "midpoint", "guidance": 2.0}},
        "selection": {"samples": 128, "every": 5},
    }
    if cfg.get("_smoke"):
        recipe = deep_merge(recipe, {
            "codec": {"width": 16, "latent_dim": 8, "codebook_size": 128},
            "codec_training": {"warmup_epochs": 1, "quantized_epochs": 1, "batch_size": 32},
            "model": {"width": 64, "layers": 2, "heads": 4, "codebook_size": 128},
            "training": {"epochs": 2, "batch_size": 32},
            "selection": {"samples": 8, "every": 1},
            "sampling": {"bagel": {"steps": 8}},
        })
    return deep_merge(recipe, cfg.get("latent_models", {}))
