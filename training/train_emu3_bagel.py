"""Train the Tiny Emu3 and BAGEL implementations with their spatial image codec."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from configs.latent_models import latent_recipe
from data.dataset import denormalize_images, load_group, normalize_images, task_labels
from data.schema import TASKS, TASK_UNDERSTAND, get_device, load_config, save_json, set_seed
from evaluation.evaluate import sample_indices
from evaluation.evaluate_emu3_bagel import sample_attributes, save_samples
from evaluation.metrics import AttributeScores, SceneOracle
from inference import save_grid
from models.latent_checkpoints import load_codec, save_latent_checkpoint
from models.spatial_codec import SpatialCodec, reconstruction_errors
from models.emu3_bagel import TinyBagel, TinyEmu3
from models.understanding import ATTRIBUTES

from .common import batch_slices, mixed_precision, optimize
from .train_base import _batches, _image_batch
from .train_understanding import augment_images, dihedral_permutations


def codec_batch(data: dict, indices: np.ndarray, device: torch.device):
    n = len(data["images"])
    rows = indices % n
    edited = indices >= n
    images = data["images"][rows].copy()
    masks = data["src_masks"][rows].copy()
    images[edited] = data["edit_targets"][rows[edited]]
    masks[edited] = data["tgt_masks"][rows[edited]]
    return torch.as_tensor(normalize_images(images), device=device), torch.as_tensor(masks, device=device)


@torch.no_grad()
def codec_features(codec: SpatialCodec, data: dict, device: torch.device, seed: int) -> torch.Tensor:
    n = len(data["images"])
    rows = np.random.default_rng(seed).choice(n, min(n, 1024), replace=False)
    indices = np.concatenate((rows, rows + n))
    features = []
    for part in batch_slices(len(indices), 128):
        images, _ = codec_batch(data, indices[part], device)
        features.append(codec.encode_raw(images).flatten(0, 2))
    return torch.cat(features).float()


@torch.no_grad()
def validate_codec(codec: SpatialCodec, cfg: dict, data: dict, device: torch.device, quantized: bool) -> dict:
    rows = np.random.default_rng(cfg["seed"] + 871).choice(len(data["images"]), min(128, len(data["images"])), replace=False)
    oracle = SceneOracle(cfg)
    meter = AttributeScores()
    total_error = 0.0
    for part in batch_slices(len(rows), 32):
        idx = rows[part]
        images, mask = codec_batch(data, idx, device)
        reconstruction, _, _ = codec(images, quantized)
        total_error += reconstruction_errors(reconstruction, images, mask).sum().item()
        for image, i in zip(denormalize_images(reconstruction), idx):
            attrs = tuple(map(int, data["prompts_gen"][i, 1:]))
            meter.add(oracle.parse(image), attrs)
    return {"reconstruction_loss": total_error / len(rows), **meter.report()}


def train_codec(cfg: dict, recipe: dict, run_dir: Path, device: torch.device) -> Path:
    train_data, val_data = [load_group(cfg, group) for group in ("train", "val")]
    codec = SpatialCodec(cfg["image_size"], **recipe["codec"]).to(device)
    settings = recipe["codec_training"]
    params = list(codec.parameters())
    optimizer = torch.optim.AdamW(params, lr=settings["lr"], weight_decay=cfg["training"]["weight_decay"])
    warmup, quantized_epochs = settings["warmup_epochs"], settings["quantized_epochs"]
    if warmup < 1 or quantized_epochs < 1:
        raise ValueError("codec needs positive warmup_epochs and quantized_epochs")
    epochs = warmup + quantized_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=settings["lr"] * 0.1)
    amp = mixed_precision(cfg, device)
    rng = np.random.default_rng(cfg["seed"] + 17)
    best = (-math.inf, -math.inf, -math.inf)
    history = []
    path = run_dir / "codec.pt"
    print(f"[codec] {sum(p.numel() for p in params):,} parameters on {device}", flush=True)
    for epoch in range(1, epochs + 1):
        codec.train()
        quantized = epoch > warmup
        if epoch == warmup + 1:
            codec.eval()
            features = codec_features(codec, train_data, device, cfg["seed"])
            codec.codebook.initialize(features, torch.Generator(device=device).manual_seed(cfg["seed"]))
            codec.train()
        order = rng.permutation(2 * len(train_data["images"]))
        total = torch.zeros((), device=device)
        for part in batch_slices(len(order), settings["batch_size"]):
            images, masks = codec_batch(train_data, order[part], device)
            with amp():
                output, continuous, commitment = codec(images, quantized)
                loss = reconstruction_errors(output, images, masks).mean() + commitment
                if quantized:
                    loss = loss + 0.5 * reconstruction_errors(continuous, images, masks).mean()
            optimize(loss, optimizer, params, cfg["training"]["grad_clip"])
            total += loss.detach() * len(images)
        codec.eval()
        metrics = validate_codec(codec, cfg, val_data, device, quantized)
        history.append({"epoch": epoch, "quantized": quantized, "train_loss": total.item() / len(order), **metrics})
        save_json(run_dir / "codec_history.json", history)
        ranking = (metrics["joint_accuracy"], metrics["shape_accuracy"], -metrics["reconstruction_loss"])
        if quantized and ranking > best:
            best = ranking
            save_latent_checkpoint(path, codec, kind="codec", image_size=cfg["image_size"], settings=recipe["codec"], epoch=epoch, metrics=metrics, complete=False)
        scheduler.step()
        print(f"[codec {epoch}/{epochs}] quantized={quantized} loss={metrics['reconstruction_loss']:.4f} "
              f"shape={metrics['shape_accuracy']:.3f} joint={metrics['joint_accuracy']:.3f}", flush=True)
    codec, saved = load_codec(path, device)
    features = codec_features(codec, train_data, device, cfg["seed"])
    codec.latent_mean.copy_(features.mean(dim=0))
    codec.latent_std.copy_(features.std(dim=0).clamp_min(0.1))
    metadata = {key: value for key, value in saved.items() if key not in ("version", "state", "complete")}
    save_latent_checkpoint(path, codec, **metadata, complete=True)
    idx = np.array([7, 18, 29, 44]) % len(val_data["images"])
    with torch.no_grad():
        images, _ = codec_batch(val_data, idx, device)
        continuous = denormalize_images(codec.decode(codec.encode(images)))
        discrete = denormalize_images(codec.decode_ids(codec.image_ids(images)))
    save_grid(run_dir / "codec_reconstructions.png", [
        [(val_data["images"][i], "Original"), (continuous[j], "Continuous"), (discrete[j], "Discrete")]
        for j, i in enumerate(idx)
    ], scale=5)
    return path


def train_model(cfg: dict, recipe: dict, run_dir: Path, device: torch.device, kind: str) -> Path:
    codec, codec_saved = load_codec(run_dir / "codec.pt", device)
    if not codec_saved.get("complete", True):
        raise RuntimeError("codec training was interrupted before latent normalization completed. Use a new run directory")
    if not cfg.get("_smoke") and codec_saved["metrics"]["shape_accuracy"] < 0.90:
        raise RuntimeError("codec reconstruction shape accuracy must reach 90% before sequence training")
    train_data, val_data = [load_group(cfg, group) for group in ("train", "val")]
    model_type = TinyEmu3 if kind == "emu3" else TinyBagel
    model = model_type(cfg, recipe, codec).to(device)
    settings = recipe["training"]
    epochs, batch_size = settings["epochs"], settings["batch_size"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=settings["lr"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    amp = mixed_precision(cfg, device)
    rng = np.random.default_rng(cfg["seed"])
    permutations = dihedral_permutations(cfg["image_size"], device) if cfg["training"]["understanding"]["augment"] else None
    path, latest = run_dir / f"{kind}.pt", run_dir / f"{kind}.latest.pt"
    history_path = run_dir / f"{kind}_history.json"
    history = []
    best = (-1.0, -1.0, -math.inf)
    first_epoch = 1
    if latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        if saved["config"] != cfg or saved["recipe"] != recipe or saved["kind"] != kind:
            raise ValueError("resume checkpoint configuration does not match this experiment")
        model.load_state_dict(saved["state"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        first_epoch, best = saved["epoch"] + 1, tuple(saved["best_ranking"])
        history = json.loads(history_path.read_text(encoding="utf-8"))[:first_epoch - 1]

    def run_epoch(data: dict, training: bool) -> dict:
        model.train(training)
        n = len(data["images"])
        labels = task_labels(data, TASK_UNDERSTAND)
        totals = {task: torch.zeros((), device=device) for task in TASKS}
        joint = torch.zeros((), device=device)
        flow_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 10)
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
                        joint += (model.vocab.greedy_answers(logits.detach()) == answer).all(dim=1).sum()
                    else:
                        mask_key = "tgt_masks" if task == "edit" else "src_masks"
                        mask = torch.as_tensor(data[mask_key][idx], device=device)
                        loss = model.image_loss(prompt, target, source, mask=mask, generator=None if training else flow_rng)
                if training:
                    optimize(loss, optimizer, params, cfg["training"]["grad_clip"])
                totals[task] += loss.detach() * len(idx)
        return {**{f"{task}_loss": value.item() / n for task, value in totals.items()}, "understand_joint": joint.item() / n}

    indices = sample_indices(val_data, recipe["selection"]["samples"], cfg["seed"] + 200)
    print(f"[{model.display_name}] {sum(p.numel() for p in params):,} trainable parameters on {device}", flush=True)
    for epoch in range(first_epoch, epochs + 1):
        train_metrics, val_metrics = run_epoch(train_data, True), run_epoch(val_data, False)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        every = recipe["selection"]["every"]
        if epoch == 1 or epoch % every == 0 or epoch == epochs:
            attributes = {}
            with amp():
                for task in ("generate", "edit"):
                    metrics, images = sample_attributes(model, cfg, val_data, indices, recipe["sampling"][kind], device, task, cfg["seed"] + 200)
                    attributes[task] = metrics
                    save_samples(run_dir / f"{kind}_val_{task}_{epoch:03d}.png", val_data, indices, images, task)
            row["samples"] = attributes
            joint = sum(value["joint_accuracy"] for value in attributes.values()) / 2
            attrs = sum(value[f"{key}_accuracy"] for value in attributes.values() for key in ATTRIBUTES) / 6
            ranking = (joint, attrs, -sum(val_metrics[f"{task}_loss"] for task in TASKS))
            if ranking > best:
                best = ranking
                save_latent_checkpoint(path, model, kind=kind, config=cfg, recipe=recipe, epoch=epoch, metrics=row)
            print(f"[{model.display_name} {epoch}/{epochs}] sample_gen={attributes['generate']['joint_accuracy']:.3f} "
                  f"sample_edit={attributes['edit']['joint_accuracy']:.3f}", flush=True)
        if not all(math.isfinite(value) for value in val_metrics.values()):
            raise RuntimeError("non-finite validation metrics")
        history.append(row)
        save_json(history_path, history)
        scheduler.step()
        save_latent_checkpoint(latest, model, kind=kind, config=cfg, recipe=recipe, epoch=epoch,
                               optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), best_ranking=best,
                               numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                               cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
        print(f"[{model.display_name} {epoch}/{epochs}] val={val_metrics}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--stage", choices=("codec", "emu3", "bagel", "all"), default="all")
    parser.add_argument("--device")
    args = parser.parse_args()
    prefix = "tiny_emu3_bagel_smoke" if args.smoke else "tiny_emu3_bagel"
    run_dir = args.run_dir or Path("artifacts/experiments") / f"{prefix}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        recipe = json.loads((run_dir / "recipe.json").read_text(encoding="utf-8"))
    else:
        cfg = load_config(args.config, args.smoke)
        recipe = latent_recipe(cfg)
        config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        save_json(run_dir / "recipe.json", recipe)
    device = torch.device(args.device) if args.device else get_device(cfg)
    set_seed(cfg["seed"])
    torch.set_num_threads(2)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"Run directory: {run_dir}", flush=True)
    codec_path = run_dir / "codec.pt"
    if args.stage == "codec" and codec_path.is_file():
        parser.error("codec checkpoint already exists. Use a new run directory for another codec experiment")
    if args.stage == "codec" or (args.stage == "all" and not codec_path.is_file()):
        train_codec(cfg, recipe, run_dir, device)
    if args.stage != "codec":
        for kind in (("emu3", "bagel") if args.stage == "all" else (args.stage,)):
            train_model(cfg, recipe, run_dir, device, kind)


if __name__ == "__main__":
    main()
