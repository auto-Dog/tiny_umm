"""Train a base Emu3-style or BAGEL-style model on the existing scene shards."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from configs.base_models import BASE_MODELS, base_recipe
from data.dataset import load_group, normalize_images, take_images, task_labels, task_prompts
from data.schema import TASK_EDIT, TASK_GENERATE, TASK_UNDERSTAND, TASKS
from models.base_checkpoints import BASE_TYPES, save_base_checkpoint
from models.emu3 import TinyEmu3
from models.understanding import ATTRIBUTES

from .common import arguments, batch_slices, log_epoch, mixed_precision, optimize, setup, write_history, writer_for
from .train_understanding import augment_images, dihedral_permutations


def _batches(n: int, size: int, rng: np.random.Generator | None) -> list[tuple[str, np.ndarray]]:
    result = []
    for task in TASKS:
        order = np.arange(n) if rng is None else rng.permutation(n)
        result.extend((task, order[part]) for part in batch_slices(n, size))
    if rng is not None:
        rng.shuffle(result)
    return result


def _image_batch(data: dict, task: str, idx: np.ndarray, device: torch.device):
    prompt = torch.as_tensor(task_prompts(data, task)[idx], dtype=torch.long, device=device)
    image = take_images(data, idx).to(device)
    if task == TASK_EDIT:
        target = torch.as_tensor(normalize_images(data["edit_targets"][idx]), device=device)
        return prompt, image, target
    if task == TASK_GENERATE:
        return prompt, None, image
    return prompt, image, None


def _fit_codebook(model: TinyEmu3, data: dict, settings: dict, seed: int, device: torch.device) -> None:
    samples = settings["codebook_samples"]
    if samples < 1:
        raise ValueError("codebook_samples must be positive")
    rng = np.random.default_rng(seed)
    n = min(len(data["images"]), math.ceil(samples / (2 * model.patches.count)))
    idx = rng.choice(len(data["images"]), n, replace=False)
    pixels = np.concatenate((data["images"][idx], data["edit_targets"][idx]))
    patches = model.patches.pack(torch.as_tensor(normalize_images(pixels), device=device)).flatten(0, 1)
    generator = torch.Generator(device=device).manual_seed(seed)
    order = torch.randperm(len(patches), device=device, generator=generator)[:samples]
    model.codebook.fit(patches[order], settings["kmeans_steps"], generator)


@torch.no_grad()
def _log_samples(model, cfg: dict, recipe: dict, data: dict, device: torch.device, writer, epoch: int) -> None:
    settings = cfg["logging"]
    n = min(settings["sample_count"], len(data["images"]))
    if n < 1:
        return
    idx = np.random.default_rng(cfg["seed"] + 31).choice(len(data["images"]), n, replace=False)
    sampling = recipe["sampling"].copy()
    if "steps" in sampling:
        sampling["steps"] = settings["sample_steps"]
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 32)
    for task in (TASK_GENERATE, TASK_EDIT):
        prompt, source, target = _image_batch(data, task, idx, device)
        result = model.sample(prompt, source, generator=generator, **sampling)
        images = (target, result) if source is None else (source, target, result)
        grid = torch.cat(images, dim=2).float().permute(0, 3, 1, 2)
        writer.add_images(f"samples/{task}", (grid + 1) / 2, epoch)


def train(cfg: dict, device: torch.device, kind: str) -> Path:
    recipe = base_recipe(cfg, kind)
    settings = recipe["training"]
    epochs, batch_size = settings["epochs"], settings["batch_size"]
    if epochs < 1 or batch_size < 1 or not math.isfinite(settings["lr"]) or settings["lr"] <= 0:
        raise ValueError("epochs, batch_size, and lr must be positive")
    train_data, val_data = [load_group(cfg, split) for split in ("train", "val")]
    if not len(train_data["images"]) or not len(val_data["images"]):
        raise ValueError("training and validation need at least one scene each")
    model = BASE_TYPES[kind](cfg, recipe["model"]).to(device)
    if isinstance(model, TinyEmu3):
        _fit_codebook(model, train_data, settings, cfg["seed"] + 2, device)
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=settings["lr"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    clip = cfg["training"]["grad_clip"]
    amp = mixed_precision(cfg, device)
    rng = np.random.default_rng(cfg["seed"])
    permutations = dihedral_permutations(cfg["image_size"], device) if cfg["training"]["understanding"]["augment"] else None

    def run_epoch(data: dict, training: bool) -> dict:
        model.train(training)
        n = len(data["images"])
        labels = task_labels(data, TASK_UNDERSTAND)
        totals = {task: torch.zeros((), device=device) for task in TASKS}
        joint = torch.zeros((), device=device)
        flow_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 10)
        loss_options = {"generator": flow_rng} if not training and kind == "bagel" else {}
        with torch.set_grad_enabled(training):
            for task, idx in _batches(n, batch_size, rng if training else None):
                prompt, source, target = _image_batch(data, task, idx, device)
                with amp():
                    if task == TASK_UNDERSTAND:
                        if training and permutations is not None:
                            source = augment_images(source, permutations)
                        answer = torch.as_tensor(np.stack([labels[key][idx] for key in ATTRIBUTES], axis=1), device=device)
                        logits = model.answer_logits(prompt, source, answer)
                        loss = F.cross_entropy(logits.flatten(0, 1), model.vocab.answer_ids(answer).flatten())
                        # A teacher-forced sequence is all-correct iff its greedy rollout is all-correct.
                        joint += (model.vocab.greedy_answers(logits.detach()) == answer).all(dim=1).sum()
                    else:
                        loss = model.image_loss(prompt, target, source, **loss_options)
                if training:
                    optimize(loss, optimizer, params, clip)
                totals[task] += loss.detach() * len(idx)
        return {
            "understand_ce": totals[TASK_UNDERSTAND].item() / n,
            "understand_joint": joint.item() / n,
            **{f"{task}_{model.image_objective}": totals[task].item() / n for task in (TASK_GENERATE, TASK_EDIT)},
        }

    history = []
    best = math.inf
    output = Path(cfg["paths"]["output_dir"]) / "base_models"
    log_cfg = {**cfg, "paths": {**cfg["paths"], "output_dir": str(output)}}
    print(f"[{kind}] {sum(p.numel() for p in params):,} trainable parameters on {device}", flush=True)
    with writer_for(log_cfg, kind) as writer:
        for epoch in range(1, epochs + 1):
            train_metrics, val_metrics = run_epoch(train_data, True), run_epoch(val_data, False)
            score = sum(val_metrics[key] for key in ("understand_ce", f"generate_{model.image_objective}", f"edit_{model.image_objective}")) / 3
            if not math.isfinite(score):
                raise RuntimeError("validation loss is not finite")
            log_epoch(writer, history, epoch, {
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in val_metrics.items()},
                "lr": optimizer.param_groups[0]["lr"],
            })
            if score < best:
                best = score
                path = save_base_checkpoint(cfg, kind, model, epoch, score)
            every = cfg["logging"]["sample_every"]
            if every > 0 and (epoch == 1 or epoch % every == 0 or epoch == epochs):
                with amp():
                    _log_samples(model, cfg, recipe, val_data, device, writer, epoch)
            scheduler.step()
            print(f"[{kind} {epoch}/{epochs}] val_loss={score:.4f} val_joint={val_metrics['understand_joint']:.3f}", flush=True)
    write_history(log_cfg, kind, history)
    return path


def main() -> None:
    parser = arguments(__doc__)
    parser.add_argument("--model", required=True, choices=BASE_MODELS)
    parser.add_argument("--device", help="override training.device")
    args = parser.parse_args()
    try:
        cfg, device = setup(args, device_override=args.device)
        print(f"Checkpoint: {train(cfg, device, args.model)}")
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
