"""Evaluate an independent sequence baseline with the repository's scene metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from configs.base_models import BASE_MODELS, base_recipe
from data.dataset import denormalize_images, load_group, normalize_images, task_labels
from data.schema import TASK_UNDERSTAND, TASKS, get_device, load_config, save_json, understand_prompt
from inference import EditResult, constrain_edit, positive_int
from models.base_checkpoints import base_checkpoint_path, load_base_checkpoint
from models.understanding import ATTRIBUTES, class_counts

from .evaluate import batches, diversity_report, editing_report, generation_report, sample_indices
from .metrics import SceneOracle, classification_metrics


class BasePipeline:
    """Adapt the sequence models to the existing generation and editing reports."""

    def __init__(self, cfg: dict, device: torch.device, kind: str, checkpoint: Path | None = None):
        self.cfg = cfg
        self.device = device
        self.checkpoint = checkpoint or base_checkpoint_path(cfg, kind)
        self.model = load_base_checkpoint(cfg, kind, device, self.checkpoint)
        self.sampling = base_recipe(cfg, kind)["sampling"]
        self.metadata = {"direct_source": False}  # No optional Stage-B source hint.

    def _sample(self, prompts: np.ndarray, source: np.ndarray | None, steps: int | None, seed: int) -> np.ndarray:
        prompt = torch.as_tensor(prompts, dtype=torch.long, device=self.device)
        images = None if source is None else torch.as_tensor(normalize_images(source), device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        options = self.sampling if steps is None else {**self.sampling, "steps": steps}
        return denormalize_images(self.model.sample(prompt, images, generator=generator, **options))

    def generate(self, prompts, *, steps=None, seed, gain=None):
        if gain is not None:
            raise ValueError("instruction gain is only available for the original flow model")
        return self._sample(prompts, None, steps, seed)

    def edit(self, images, prompts, *, region, steps=None, seed, gain=None, direct=None):
        if gain is not None or direct is not None:
            raise ValueError("gain and direct-source ablations are only available for the original flow model")
        raw = self._sample(prompts, images, steps, seed)
        return EditResult(raw, constrain_edit(images, raw, region), region)

    def understand(self, images: np.ndarray) -> dict[str, np.ndarray]:
        prompt = torch.as_tensor(understand_prompt(self.cfg), dtype=torch.long, device=self.device)
        predictions = []
        for _, idx in batches(np.arange(len(images)), self.cfg):
            source = torch.as_tensor(normalize_images(images[idx]), device=self.device)
            predictions.append(self.model.predict(prompt.expand(len(idx), -1), source).cpu().numpy())
        return dict(zip(ATTRIBUTES, np.concatenate(predictions).T))


def evaluate(args: argparse.Namespace) -> Path:
    cfg = load_config(args.config, args.smoke)
    if args.batch_size is not None:
        cfg["evaluation"]["batch_size"] = args.batch_size
    if args.steps is not None and args.model != "bagel":
        raise ValueError("--steps is only used for BAGEL sampling")
    device = torch.device(args.device) if args.device else get_device(cfg)
    pipeline = BasePipeline(cfg, device, args.model, args.checkpoint)
    oracle = SceneOracle(cfg)
    seed = cfg["seed"] + 200
    report = {
        "model": args.model, "checkpoint": str(pipeline.checkpoint), "seed": seed,
        "sampling": {**pipeline.sampling, **({"steps": args.steps} if args.steps is not None else {})},
        "splits": {},
    }
    limit = args.samples if args.samples is not None else cfg["evaluation"]["max_samples"]
    for split in (args.split,) if args.split else ("test_random", "test_comp"):
        data = load_group(cfg, split)
        idx = sample_indices(data, limit, seed)
        if not len(idx):
            raise ValueError(f"{split} contains no scenes")
        print(f"[{split}] evaluating {args.model} on {len(idx)} scenes", flush=True)
        result = {"scenes": len(idx), "scene_ids": data["scene_ids"][idx].tolist()}
        if args.task in (None, TASK_UNDERSTAND):
            labels = {key: value[idx] for key, value in task_labels(data, TASK_UNDERSTAND).items()}
            result["understanding"] = classification_metrics(labels, pipeline.understand(data["images"][idx]), class_counts(cfg))
        if args.task in (None, "generate"):
            result["generation"] = generation_report(pipeline, data, idx, oracle, steps=args.steps, seed=seed)
            result["generation"]["diversity"] = diversity_report(pipeline, data, idx, steps=args.steps, seed=seed + 100_000)
        if args.task in (None, "edit"):
            result["editing"] = editing_report(pipeline, data, idx, oracle, steps=args.steps, seed=seed + 10_000)
        report["splits"][split] = result
    out = args.out or Path(cfg["paths"]["output_dir"]) / "base_models" / f"{args.model}_evaluation.json"
    save_json(out, report)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model", choices=BASE_MODELS, required=True)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--split", choices=("test_random", "test_comp", "val"))
    parser.add_argument("--samples", type=positive_int)
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--steps", type=positive_int)
    parser.add_argument("--out", type=Path)
    try:
        print(f"Report: {evaluate(parser.parse_args())}")
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
