"""Sample and evaluate the Tiny Emu3 and BAGEL implementations."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data.dataset import denormalize_images, load_group, normalize_images, task_labels
from data.schema import TASK_UNDERSTAND, save_json
from inference import positive_int, save_grid
from models.latent_checkpoints import load_latent_model, save_latent_checkpoint
from models.understanding import class_counts

from .evaluate import editing_report, generation_report, sample_indices
from .evaluate_base import BasePipeline
from .metrics import AttributeScores, SceneOracle, classification_metrics


class Emu3BagelPipeline(BasePipeline):
    def __init__(self, checkpoint: Path, device: torch.device):
        self.model, self.saved = load_latent_model(checkpoint, device)
        self.cfg = self.saved["config"]
        self.device = device
        self.checkpoint = checkpoint
        self.sampling = self.saved.get("sampling", self.saved["recipe"]["sampling"][self.saved["kind"]])
        self.metadata = {"direct_source": False}


@torch.inference_mode()
def sample_attributes(model, cfg, data, indices, sampling, device, task, seed):
    meter = AttributeScores()
    oracle = SceneOracle(cfg)
    outputs = []
    key = "prompts_edit" if task == "edit" else "prompts_gen"
    for start in range(0, len(indices), 32):
        idx = indices[start:start + 32]
        prompts = torch.as_tensor(data[key][idx], device=device, dtype=torch.long)
        source = None
        if task == "edit":
            source = torch.as_tensor(normalize_images(data["images"][idx]), device=device)
        generator = torch.Generator(device=device).manual_seed(seed + start)
        images = denormalize_images(model.sample(prompts, source, generator=generator, **sampling))
        outputs.append(images)
        for image, i in zip(images, idx):
            count = int(data["counts"][i])
            color, shape = map(int, data[key][i, 2:])
            meter.add(oracle.parse(image), (count, color, shape))
    return meter.report(), np.concatenate(outputs)


def save_samples(path: Path, data: dict, indices: np.ndarray, images: np.ndarray, task: str) -> None:
    rows = []
    for j, i in enumerate(indices[:8]):
        row = [(data["images"][i], "Reference")]
        if task == "edit":
            row.append((data["edit_targets"][i], "Edit target"))
        row.append((images[j], "Generated"))
        rows.append(row)
    save_grid(path, rows, scale=5)


def calibrate_sampling(checkpoint: Path, device: torch.device, guidance_values=(1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)) -> None:
    """Select guidance on validation scenes while keeping the trained weights fixed."""
    pipeline = Emu3BagelPipeline(checkpoint, device)
    cfg = pipeline.cfg
    data = load_group(cfg, "val")
    seed = cfg["seed"] + 200
    indices = sample_indices(data, pipeline.saved["recipe"]["selection"]["samples"], seed)
    candidates = list(dict.fromkeys((pipeline.sampling["guidance"], *guidance_values)))
    ranking = (-1.0, -1.0)
    selected = pipeline.sampling
    history = []
    for guidance in candidates:
        sampling = {**pipeline.sampling, "guidance": guidance}
        metrics = {}
        for task in ("generate", "edit"):
            metrics[task], _ = sample_attributes(pipeline.model, cfg, data, indices, sampling, device, task, seed)
        joint = sum(value["joint_accuracy"] for value in metrics.values()) / 2
        attrs = sum(value[f"{key}_accuracy"] for value in metrics.values() for key in ("count", "color", "shape")) / 6
        history.append({"guidance": guidance, "metrics": metrics})
        if (joint, attrs) > ranking:
            ranking, selected = (joint, attrs), sampling
        print(f"[guidance {guidance}] gen={metrics['generate']['joint_accuracy']:.3f} edit={metrics['edit']['joint_accuracy']:.3f}", flush=True)
    saved = pipeline.saved
    metadata = {key: value for key, value in saved.items() if key not in ("state", "version", "sampling")}
    save_latent_checkpoint(checkpoint, pipeline.model, **metadata, sampling=selected)
    save_json(checkpoint.parent / f"{saved['kind']}_sampling_calibration.json", {
        "checkpoint_epoch": saved["epoch"], "validation_indices": indices.tolist(), "seed": seed,
        "selected": selected, "candidates": history,
    })


def evaluate(checkpoint: Path, *, samples: int, device: torch.device, baseline: Path | None = None) -> Path:
    pipeline = Emu3BagelPipeline(checkpoint, device)
    cfg = pipeline.cfg
    pipelines = {"corrected": pipeline}
    if baseline is not None:
        pipelines["baseline"] = BasePipeline(cfg, device, pipeline.saved["kind"], baseline)
    oracle = SceneOracle(cfg)
    seed = cfg["seed"] + 200
    report = {"kind": pipeline.saved["kind"], "model_name": pipeline.model.display_name,
              "samples_per_split": samples, "seed": seed, "variants": {}}
    for name, current in pipelines.items():
        variant = {"checkpoint": str(current.checkpoint), "sampling": current.sampling, "splits": {}}
        for split in ("test_random", "test_comp"):
            data = load_group(cfg, split)
            indices = sample_indices(data, samples, seed)
            print(f"[{pipeline.model.display_name} {name} {split}] {len(indices)} scenes", flush=True)
            labels = {key: value[indices] for key, value in task_labels(data, TASK_UNDERSTAND).items()}
            result = {
                "scene_ids": data["scene_ids"][indices].tolist(),
                "understanding": classification_metrics(labels, current.understand(data["images"][indices]), class_counts(cfg)),
                "generation": generation_report(current, data, indices, oracle, steps=None, seed=seed),
                "editing": editing_report(current, data, indices, oracle, steps=None, seed=seed + 10_000),
            }
            preview_idx = indices[:8]
            generated = current.generate(data["prompts_gen"][preview_idx], seed=seed)
            save_samples(checkpoint.parent / f"{pipeline.saved['kind']}_{name}_{split}_generate.png", data, preview_idx, generated, "generate")
            source = data["images"][preview_idx]
            region = np.stack([oracle.region(oracle.parse(image)) for image in source])
            edits = current.edit(source, data["prompts_edit"][preview_idx], region=region, seed=seed + 10_000)
            save_samples(checkpoint.parent / f"{pipeline.saved['kind']}_{name}_{split}_edit.png", data, preview_idx, edits.raw, "edit")
            variant["splits"][split] = result
        report["variants"][name] = variant
    path = checkpoint.parent / f"{pipeline.saved['kind']}_evaluation.json"
    save_json(path, report)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--samples", default=256, type=positive_int)
    parser.add_argument("--calibrate", action="store_true", help="select text guidance on validation scenes before testing")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    if args.calibrate:
        calibrate_sampling(args.checkpoint, torch.device(args.device))
    print(evaluate(args.checkpoint, samples=args.samples, device=torch.device(args.device), baseline=args.baseline))


if __name__ == "__main__":
    main()
