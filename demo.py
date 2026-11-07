"""Create contact sheets for understanding, generation and single/multi-step editing."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data.dataset import load_group
from data.schema import get_device, load_config, save_json
from evaluation.evaluate import chain_prompts, sample_indices
from inference import Pipeline, positive_int, predict_understanding, save_grid
from models.checkpoints import load_understanding


def demo(args: argparse.Namespace) -> Path:
    cfg = load_config(args.config, args.smoke)
    device = get_device(cfg)
    out = args.out_dir or Path(cfg["paths"]["output_dir"])
    data = load_group(cfg, args.split)
    seed = cfg["seed"] + 800
    idx = sample_indices(data, args.samples, seed)
    source = data["images"][idx]
    u = load_understanding(cfg, device)
    pred = predict_understanding(u, source, cfg, device)
    pipeline = Pipeline(cfg, device, args.checkpoint)
    generated = pipeline.generate(data["prompts_gen"][idx], steps=args.steps, seed=seed)
    edits = pipeline.edit(source, data["prompts_edit"][idx], steps=args.steps, seed=seed + 1)
    chain_idx = idx[:min(3, len(idx))]
    rounds = chain_prompts(cfg, data, chain_idx, 3, seed)
    chain = pipeline.chain(data["images"][chain_idx], rounds, steps=args.steps, seed=seed + 2)

    def attrs(n: int, color: int, shape: int) -> dict:
        return {"count": int(n), "color": cfg["colors"]["names"][color], "shape": cfg["shapes"]["names"][shape]}

    def label(attributes: dict) -> str:
        return f"{attributes['count']} {attributes['color']} {attributes['shape']}"

    truth, predictions, requested = [], [], []
    for j, i in enumerate(idx):
        truth.append(attrs(data["counts"][i], data["colors"][i], data["shapes"][i]))
        predictions.append(attrs(pred["count"][j] + 1, pred["color"][j], pred["shape"][j]))
        requested.append(attrs(data["counts"][i], data["edit_colors"][i], data["edit_shapes"][i]))
    understood = [(image, label(p)) for image, p in zip(source, predictions)]
    files = {
        "understanding": save_grid(out / "understanding.png", [understood[i:i + 4] for i in range(0, len(idx), 4)], scale=5),
        "generation": save_grid(out / "generation.png", [
            [(src, label(attrs_)), (img, "Generated")] for src, img, attrs_ in zip(source, generated, truth)
        ], scale=5),
        "batch_edit": save_grid(out / "batch_edit.png", [
            [(src, "Source"), (data["edit_targets"][i], label(goal)), (raw, "Raw"), (fixed, "Constrained")]
            for i, src, goal, raw, fixed in zip(idx, source, requested, edits.raw, edits.constrained)
        ], scale=5),
        "chain_edit": save_grid(out / "chain_edit.png", [
            [(data["images"][i], "Source")] + [
                (result.constrained[j], label(attrs(data["counts"][i], prompt[j, 2], prompt[j, 3])))
                for prompt, result in zip(rounds, chain)
            ] for j, i in enumerate(chain_idx)
        ], scale=5),
    }
    summary = out / "demo_summary.json"
    save_json(summary, {
        "split": args.split, "smoke": args.smoke, "checkpoint": str(pipeline.checkpoint),
        "samples": [
            {"scene_id": int(data["scene_ids"][i]), "truth": t, "prediction": p, "edit_instruction": goal,
             "editable_pixels": int(edits.region[j].sum())}
            for j, (i, t, p, goal) in enumerate(zip(idx, truth, predictions, requested))
        ],
        "images": {name: str(path) for name, path in files.items()},
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--split", choices=("test_random", "test_comp", "val"), default="test_random")
    parser.add_argument("--samples", type=positive_int, default=6)
    parser.add_argument("--steps", type=positive_int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out-dir", type=Path)
    try:
        print(f"Demo: {demo(parser.parse_args())}")
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
