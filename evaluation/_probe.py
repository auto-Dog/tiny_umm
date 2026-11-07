"""Common harness for diagnostics that are restricted to the validation split."""
from __future__ import annotations

import argparse
from pathlib import Path

from data.dataset import load_group
from data.schema import get_device, load_config, save_json
from inference import Pipeline, positive_int

from .evaluate import editing_report, generation_report, sample_indices
from .metrics import SceneOracle


def arguments(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--samples", type=positive_int, default=512)
    parser.add_argument("--steps", type=positive_int)
    parser.add_argument("--out", type=Path)
    return parser


class Probe:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = load_config(args.config, args.smoke)
        self.device = get_device(self.cfg)
        self.seed = self.cfg["seed"] + 500
        self.data = load_group(self.cfg, "val")
        self.indices = sample_indices(self.data, args.samples, self.seed)
        self.oracle = SceneOracle(self.cfg)

    def load(self, path: Path) -> Pipeline:
        return Pipeline(self.cfg, self.device, path)

    def score(self, pipeline: Pipeline, *, gain: float | None = None, direct: bool | None = None) -> dict:
        return {
            "checkpoint": str(pipeline.checkpoint),
            "generation": generation_report(
                pipeline, self.data, self.indices, self.oracle, steps=self.args.steps, seed=self.seed, gain=gain,
            ),
            "editing": editing_report(
                pipeline, self.data, self.indices, self.oracle, steps=self.args.steps, seed=self.seed + 10_000,
                gain=gain, direct=direct,
            ),
        }

    def save(self, name: str, variants: dict) -> Path:
        path = self.args.out or Path(self.cfg["paths"]["output_dir"]) / f"{name}.json"
        save_json(path, {
            "split": "val", "samples": len(self.indices), "smoke": self.args.smoke,
            "steps": self.args.steps or self.cfg["sampling"]["steps"],
            "seed": self.seed, "scene_ids": self.data["scene_ids"][self.indices].tolist(),
            "variants": variants,
        })
        return path
