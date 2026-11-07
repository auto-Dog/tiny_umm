"""Stage B: pretrain the pixel Flow on generation and source-conditioned editing."""
from __future__ import annotations

import math

import numpy as np
import torch

from data.dataset import load_group, normalize_images
from models.checkpoints import flow_state, save_checkpoint
from models.flow import flow_matching_errors, sample_flow
from models.tiny_umm import TinyUMM

from .common import (
    arguments, batch_slices, log_epoch, mixed_indices, mixed_precision, optimize, setup, write_history, writer_for,
)


def flow_batch(
    data: dict, idx: np.ndarray, edit: np.ndarray, device: torch.device, reconstruct: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = normalize_images(data["images"][idx])
    target = images.copy()
    prompts = data["prompts_gen"][idx].copy()
    edit_idx = idx[edit]
    prompts[edit] = data["prompts_edit"][edit_idx]
    if reconstruct:
        # Identity edit instructions: keep count, and request the source color/shape.
        prompts[edit, 2] = data["colors"][edit_idx]
        prompts[edit, 3] = data["shapes"][edit_idx]
    else:
        target[edit] = normalize_images(data["edit_targets"][edit_idx])
    return (
        torch.as_tensor(images, device=device),
        torch.as_tensor(target, device=device),
        torch.as_tensor(prompts, device=device, dtype=torch.long),
    )


@torch.no_grad()
def log_samples(
    model: TinyUMM, cfg: dict, data: dict, device: torch.device, writer, epoch: int, direct: bool,
) -> None:
    settings = cfg["logging"]
    rng = np.random.default_rng(cfg["seed"] + 31)
    n = min(settings["sample_count"], len(data["images"]))
    if n < 1:
        return
    idx = rng.choice(len(data["images"]), size=n, replace=False)
    src, tgt, edit = flow_batch(data, idx, np.ones(n, dtype=bool), device)
    generate = torch.as_tensor(data["prompts_gen"][idx], device=device, dtype=torch.long)
    kwargs = {
        "steps": settings["sample_steps"], "solver": cfg["sampling"]["solver"],
        "generator": torch.Generator(device=device).manual_seed(cfg["seed"] + 32),
    }
    gen_images = sample_flow(model.flow, model.condition(generate), **kwargs)
    edit_images = sample_flow(model.flow, model.condition(edit, src, direct=direct), **kwargs)
    for tag, images in {
        "samples/generate": torch.cat((src, gen_images), dim=2),
        "samples/edit": torch.cat((src, tgt, edit_images), dim=2),
    }.items():
        writer.add_images(tag, (images.float().permute(0, 3, 1, 2) + 1) / 2, epoch)


def train_flow_model(
    cfg: dict, device: torch.device, model: TinyUMM, kind: str, settings: dict, *,
    generation_ratio: float, joint: bool = False, reconstruct_epochs: int = 0,
) -> None:
    """Shared B/C loop; C computes derivatives through Flow with its weights frozen."""
    connected = kind != "flow"
    model.to(device)
    train_data, val_data = [load_group(cfg, split) for split in ("train", "val")]
    n_train, n_val = len(train_data["images"]), len(val_data["images"])
    batch_size = settings["batch_size"]
    epochs = settings["epochs"] + reconstruct_epochs
    if settings["epochs"] < 1 or reconstruct_epochs < 0:
        raise ValueError("need at least one training epoch and a nonnegative reconstruction warmup")
    direct = bool(settings["direct_source"])
    drop = float(settings.get("direct_dropout", 0.0))
    if not 0 <= drop <= 1:
        raise ValueError("direct_dropout must be in [0, 1]")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=settings["lr"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    clip = cfg["training"]["grad_clip"]
    amp = mixed_precision(cfg, device)
    rng = np.random.default_rng(cfg["seed"] + 1)
    history = []
    best = math.inf
    # Separate log directories keep the instruction ablation comparable.
    stage = kind

    @torch.no_grad()
    def validate(is_edit: bool) -> float:
        generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 10 + int(is_edit))
        total = 0.0
        order = np.arange(n_val)
        for part in batch_slices(n_val, batch_size):
            idx = order[part]
            src, target, prompts = flow_batch(val_data, idx, np.full(len(idx), is_edit), device)
            with amp():
                cond = model.condition(prompts, src if is_edit else None, direct=direct)
                total += flow_matching_errors(model.flow, target, cond, generator).sum().item()
        return total / n_val

    print(f"[{stage}] {sum(p.numel() for p in params):,} trainable parameters on {device}")
    with writer_for(cfg, stage) as writer:
        for epoch in range(1, epochs + 1):
            if connected:
                model.train_bridge(joint_instruction=joint, adapters=kind == "adapter")
            else:
                model.train()
            warmup = epoch <= reconstruct_epochs
            idx, edit = mixed_indices(n_train, 0.0 if warmup else generation_ratio, rng)
            total = 0.0
            for part in batch_slices(len(idx), batch_size):
                edit_slice = edit[part]
                src, target, prompt = flow_batch(train_data, idx[part], edit_slice, device, reconstruct=warmup)
                source_hint = direct
                if direct and drop > 0:
                    source_hint = torch.rand(len(src), device=device) >= drop
                with amp():
                    condition = model.condition(prompt, src, direct=source_hint)
                    loss = flow_matching_errors(model.flow, target, condition).mean()
                if not connected or joint or kind == "adapter" or edit_slice.any():
                    optimize(loss, optimizer, params, clip)
                total += loss.detach().item() * len(src)
            model.eval()
            val_generate, val_edit = validate(False), validate(True)
            score = val_edit if connected else (val_edit + val_generate) / 2
            if not math.isfinite(score):
                raise RuntimeError("validation Flow Matching loss is not finite")
            log_epoch(writer, history, epoch, {
                "train/fm": total / len(idx), "val/generate_fm": val_generate, "val/edit_fm": val_edit,
                "lr": optimizer.param_groups[0]["lr"],
            })
            every = cfg["logging"]["sample_every"]
            if every > 0 and (epoch == 1 or epoch % every == 0 or epoch == epochs):
                log_samples(model, cfg, val_data, device, writer, epoch, direct)
            if not warmup and score < best:
                best = score
                save_checkpoint(cfg, kind, flow_state(model, connected), epoch, score, direct_source=direct)
            scheduler.step()
            label = "reconstruction" if warmup else stage
            print(f"[{label} {epoch}/{epochs}] FM={total / len(idx):.4f} "
                  f"val_gen={val_generate:.4f} val_edit={val_edit:.4f}", flush=True)
    write_history(cfg, stage, history)


def main() -> None:
    args = arguments(__doc__).parse_args()
    cfg, device = setup(args)
    model = TinyUMM(cfg)
    model.understanding.requires_grad_(False)
    train_flow_model(cfg, device, model, "flow", cfg["training"]["flow"], generation_ratio=0.5)


if __name__ == "__main__":
    main()
