"""Evaluate understanding, generation and editing on separate scene splits."""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from data.dataset import load_group, task_labels
from data.schema import TASK_UNDERSTAND, get_device, load_config, palette, save_json
from inference import Pipeline, constrain_edit, positive_int, predict_understanding
from models.checkpoints import checkpoint_path, load_understanding
from models.understanding import class_counts

from .metrics import (
    AttributeScores, GeometryScores, PixelScores, SceneOracle, classification_metrics, localization_scores,
)


def sample_indices(data: dict, count: int | None, seed: int) -> np.ndarray:
    n = len(data["images"])
    if count is None or count >= n:
        return np.arange(n)
    if count <= 0:
        raise ValueError("number of evaluation samples must be positive")
    return np.sort(np.random.default_rng(seed).choice(n, size=count, replace=False))


def batches(indices: np.ndarray, cfg: dict):
    size = cfg["evaluation"]["batch_size"]
    if size < 1:
        raise ValueError("evaluation.batch_size must be positive")
    for start in range(0, len(indices), size):
        yield start, indices[start:start + size]


def generation_report(
    pipeline: Pipeline, data: dict, indices: np.ndarray, oracle: SceneOracle, *,
    steps: int | None, seed: int, gain: float | None = None, understanding=None,
) -> dict:
    scores = AttributeScores()
    truth = data["prompts_gen"][indices, 1:].astype(np.int64)
    generated = []
    for start, idx in batches(indices, pipeline.cfg):
        images = pipeline.generate(data["prompts_gen"][idx], steps=steps, seed=seed + start, gain=gain)
        for image, attrs in zip(images, truth[start:start + len(idx)]):
            scores.add(oracle.parse(image), tuple(attrs))
        if understanding is not None:
            generated.append(images)
    result = {"oracle": scores.report()}
    if understanding is not None:
        cfg = pipeline.cfg
        predicted = predict_understanding(understanding, np.concatenate(generated), cfg, pipeline.device)
        labels = dict(zip(("count", "color", "shape"), (truth - np.array([1, 0, 0])).T))
        result["understanding_aux"] = classification_metrics(labels, predicted, class_counts(cfg))
    return result


def diversity_report(
    pipeline: Pipeline, data: dict, indices: np.ndarray, *, steps: int | None, seed: int,
) -> dict | None:
    settings = pipeline.cfg["evaluation"]
    repeats = settings["diversity_repeats"]
    prompts = np.unique(data["prompts_gen"][indices], axis=0)
    n = min(len(prompts), settings["diversity_conditions"])
    if n < 1 or repeats < 2:
        return None
    rng = np.random.default_rng(seed)
    prompts = np.repeat(prompts[rng.choice(len(prompts), n, replace=False)], repeats, axis=0)
    images = np.concatenate([
        pipeline.generate(prompts[idx], steps=steps, seed=seed + start)
        for start, idx in batches(np.arange(len(prompts)), pipeline.cfg)
    ]).reshape(n, repeats, *data["images"].shape[1:]).astype(np.float32) / 255.0
    distances = [np.abs(images[:, i] - images[:, j]).mean().item() for i, j in combinations(range(repeats), 2)]
    return {"conditions": n, "samples_per_condition": repeats, "pairwise_pixel_mae": float(np.mean(distances))}


def editing_report(
    pipeline: Pipeline, data: dict, indices: np.ndarray, oracle: SceneOracle, *,
    steps: int | None, seed: int, gain: float | None = None, direct: bool | None = None,
) -> dict:
    names = ("raw", "constrained", "gt_region_oracle")
    attrs = {key: AttributeScores() for key in names}
    outside = {key: PixelScores() for key in names}
    inpaint = {key: PixelScores() for key in names}
    geometry = {key: GeometryScores() for key in names}
    outside_selected = PixelScores()
    localization = []
    for start, idx in batches(indices, pipeline.cfg):
        sources = data["images"][idx]
        prompts = data["prompts_edit"][idx]
        region = np.stack([oracle.region(oracle.parse(image)) for image in sources])
        sampled = pipeline.edit(sources, prompts, region=region, steps=steps, seed=seed + start, gain=gain, direct=direct)
        # Everything below is scoring only: no target image, target mask or
        # ground-truth region enters the sampler or the ordinary constraints.
        for j, i in enumerate(idx):
            gt_region = data["region_masks"][i].astype(bool)
            target = data["edit_targets"][i]
            n = int(data["counts"][i])
            truth = (n, int(prompts[j, 2]), int(prompts[j, 3]))
            exposed = data["src_masks"][i].astype(bool) & ~data["tgt_masks"][i].astype(bool)
            localization.append(localization_scores(sampled.region[j], gt_region))
            upper = constrain_edit(sources[j], sampled.raw[j], gt_region)
            for name, image in zip(names, (sampled.raw[j], sampled.constrained[j], upper)):
                parsed = oracle.parse(image)
                attrs[name].add(parsed, truth)
                outside[name].add(image, sources[j], ~gt_region)
                inpaint[name].add(image, target, exposed)
                geometry[name].add(parsed, data["centers"][i, :n], data["sizes"][i, :n])
            outside_selected.add(sampled.constrained[j], sources[j], ~sampled.region[j])
    result = {}
    for name in names:
        attr = attrs[name].report()
        attr["count_preservation"] = attr.pop("count_accuracy")
        result[name] = {
            **attr, "outside_gt_region": outside[name].report(), "inpainting": inpaint[name].report(),
            "geometry": geometry[name].report(),
        }
    result["constrained"]["outside_selected_region"] = outside_selected.report()
    result["localization"] = dict(zip(("iou", "coverage", "precision"), np.mean(localization, axis=0).tolist()))
    result["direct_source"] = pipeline.metadata["direct_source"] if direct is None else direct
    return result


def chain_prompts(cfg: dict, data: dict, indices: np.ndarray, n_rounds: int, seed: int) -> list[np.ndarray]:
    prompts = [data["prompts_edit"][indices].copy()]
    pal = palette(cfg)
    sources = data["images"][indices]
    border = np.concatenate((sources[:, 0], sources[:, -1], sources[:, 1:-1, 0], sources[:, 1:-1, -1]), axis=1)
    bg = np.median(border, axis=1) / 255.0
    dist = np.linalg.norm(pal[None] - bg[:, None], axis=-1)
    threshold = cfg["data"]["background"]["min_fg_bg_distance"] / 255.0
    rng = np.random.default_rng(seed)
    for _ in range(1, n_rounds):
        previous = prompts[-1]
        prompt = previous.copy()
        for j, row in enumerate(dist):
            choices = np.flatnonzero((row >= threshold) & (np.arange(len(pal)) != previous[j, 2]))
            prompt[j, 2] = rng.choice(choices) if len(choices) else row.argmax()
        prompt[:, 3] = (previous[:, 3] + rng.integers(1, len(cfg["shapes"]["names"]), size=len(previous))) % len(cfg["shapes"]["names"])
        prompts.append(prompt)
    return prompts


def chain_report(
    pipeline: Pipeline, data: dict, indices: np.ndarray, oracle: SceneOracle, *, steps: int | None, seed: int,
) -> dict | None:
    cfg = pipeline.cfg
    n = min(cfg["evaluation"]["chain_samples"], len(indices))
    n_rounds = cfg["evaluation"]["chain_rounds"]
    if n < 1 or n_rounds < 1:
        return None
    indices = np.sort(np.random.default_rng(seed).choice(indices, n, replace=False))
    rounds = chain_prompts(cfg, data, indices, n_rounds, seed)
    results = {}
    for name in ("raw", "constrained"):
        scores = [AttributeScores() for _ in rounds]
        drift = [PixelScores() for _ in rounds]
        for start, idx in batches(np.arange(n), cfg):
            rows = indices[idx]
            sources = data["images"][rows]
            region = np.stack([oracle.region(oracle.parse(image)) for image in sources])
            outputs = pipeline.chain(
                sources, [prompt[idx] for prompt in rounds], region=region,
                steps=steps, seed=seed + start, use_raw=name == "raw",
            )
            for k, output in enumerate(outputs):
                images = output.raw if name == "raw" else output.constrained
                for j, i in enumerate(rows):
                    attrs = (int(data["counts"][i]), *map(int, rounds[k][idx[j], 2:]))
                    scores[k].add(oracle.parse(images[j]), attrs)
                    drift[k].add(images[j], sources[j], ~data["region_masks"][i].astype(bool))
        results[name] = [
            {"round": k + 1, **attr.report(), "background_drift": pix.report()}
            for k, (attr, pix) in enumerate(zip(scores, drift))
        ]
    return results


def _options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    task = parser.add_mutually_exclusive_group()
    task.add_argument("--all", action="store_true", help="include baselines, direct-path ablation and multi-round edits")
    task.add_argument("--task", choices=("understand", "generate", "edit"))
    parser.add_argument("--split", choices=("test_random", "test_comp", "val"))
    parser.add_argument("--checkpoint", action="append", nargs=2, metavar=("NAME", "PATH"),
                        help="evaluate a named checkpoint, repeat for matched comparisons")
    parser.add_argument("--samples", type=positive_int, help="limit samples per split")
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--steps", type=positive_int)
    parser.add_argument("--out", type=Path)
    return parser


def evaluate(args: argparse.Namespace) -> Path:
    cfg = load_config(args.config, args.smoke)
    if args.batch_size is not None:
        cfg["evaluation"]["batch_size"] = args.batch_size
    device = get_device(cfg)
    seed = cfg["seed"] + 200
    splits = (args.split,) if args.split else ("test_random", "test_comp")
    stages = {}
    missing = []
    need_u = args.task != "edit"
    understanding = load_understanding(cfg, device) if need_u else None
    if args.checkpoint:
        if args.all:
            raise ValueError("--checkpoint and --all cannot be combined")
        paths = dict(args.checkpoint)
        if len(paths) != len(args.checkpoint):
            raise ValueError("--checkpoint names must be unique")
        stages = {name: Pipeline(cfg, device, path) for name, path in paths.items()}
    elif args.task != "understand":
        if args.all:
            stages["gold_structured"] = Pipeline(cfg, device, checkpoint_path(cfg, "flow"))
            discrete = checkpoint_path(cfg, "discrete")
            if discrete.is_file():
                stages["discrete"] = Pipeline(cfg, device, discrete)
            else:
                missing.append("discrete")
        stages["connector"] = Pipeline(cfg, device)
    oracle = SceneOracle(cfg)
    report = {
        "seed": seed, "smoke": args.smoke, "steps": args.steps or cfg["sampling"]["steps"],
        "batch_size": cfg["evaluation"]["batch_size"],
        "optional_baselines_missing": missing,
        "checkpoints": {name: str(pipeline.checkpoint) for name, pipeline in stages.items()},
        "splits": {},
    }
    for split in splits:
        data = load_group(cfg, split)
        limit = args.samples if args.samples is not None else cfg["evaluation"]["max_samples"]
        idx = sample_indices(data, limit, seed)
        result = {"scenes": len(idx)}
        if args.checkpoint:
            result["scene_ids"] = data["scene_ids"][idx].tolist()
        if args.task in (None, "understand"):
            targets = {k: v[idx] for k, v in task_labels(data, TASK_UNDERSTAND).items()}
            before = predict_understanding(understanding, data["images"][idx], cfg, device)
            result["understanding"] = classification_metrics(targets, before, class_counts(cfg))
            if args.all:
                after = predict_understanding(stages["connector"].model.understanding, data["images"][idx], cfg, device)
                result["understanding_after_connection"] = {
                    **classification_metrics(targets, after, class_counts(cfg)),
                    "predictions_identical": all(np.array_equal(before[key], after[key]) for key in before),
                }
        if stages:
            result["variants"] = {}
        for name, pipeline in stages.items():
            print(f"[{split}] evaluating {name} on {len(idx)} scenes", flush=True)
            variant = {}
            if args.checkpoint and args.task in (None, "understand") and pipeline.model.bridge is not None:
                after = predict_understanding(pipeline.model.understanding, data["images"][idx], cfg, device)
                variant["understanding"] = {
                    **classification_metrics(targets, after, class_counts(cfg)),
                    "predictions_identical": all(np.array_equal(before[key], after[key]) for key in before),
                }
            if args.task in (None, "generate"):
                variant["generation"] = generation_report(
                    pipeline, data, idx, oracle, steps=args.steps, seed=seed, understanding=understanding,
                )
                variant["generation"]["diversity"] = diversity_report(
                    pipeline, data, idx, steps=args.steps, seed=seed + 100_000,
                )
            if args.task in (None, "edit"):
                variant["editing"] = editing_report(
                    pipeline, data, idx, oracle, steps=args.steps, seed=seed + 10_000,
                )
            result["variants"][name] = variant
        if args.all:
            connected = stages["connector"]
            variant = result["variants"]["connector"]
            variant["direct_source_ablation"] = editing_report(
                connected, data, idx, oracle, steps=args.steps, seed=seed + 10_000,
                direct=not connected.metadata["direct_source"],
            )
            variant["chain"] = chain_report(connected, data, idx, oracle, steps=args.steps, seed=seed + 20_000)
            result["connection_comparison"] = {
                name: {
                    "generation_joint": row["generation"]["oracle"]["joint_accuracy"],
                    "raw_edit_joint": row["editing"]["raw"]["joint_accuracy"],
                    "constrained_edit_joint": row["editing"]["constrained"]["joint_accuracy"],
                    "count_preservation": row["editing"]["constrained"]["count_preservation"],
                }
                for name, row in result["variants"].items()
            }
        report["splits"][split] = result
    path = args.out or Path(cfg["paths"]["output_dir"]) / "evaluation.json"
    save_json(path, report)
    return path


def main() -> None:
    parser = _options()
    try:
        print(f"Report: {evaluate(parser.parse_args())}")
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
