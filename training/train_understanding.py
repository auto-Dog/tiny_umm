"""Stage A: train the three understanding heads with masked instructions."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from data.dataset import load_group, take_images, task_labels
from data.schema import TASK_UNDERSTAND
from models.checkpoints import save_checkpoint
from models.understanding import ATTRIBUTES, UnderstandingMLP

from .common import arguments, batch_slices, log_epoch, mixed_precision, optimize, setup, write_history, writer_for


def dihedral_permutations(size: int, device: torch.device) -> torch.Tensor:
    positions = np.arange(size * size).reshape(size, size)
    transforms = [np.rot90(positions, k)[:, ::flip] for k in range(4) for flip in (1, -1)]
    return torch.as_tensor(np.stack(transforms).reshape(8, -1), device=device)


def augment_images(images: torch.Tensor, permutations: torch.Tensor) -> torch.Tensor:
    transform = torch.randint(len(permutations), (len(images),), device=images.device)
    indices = permutations[transform][..., None].expand(-1, -1, 3)
    return images.flatten(1, 2).gather(1, indices).reshape_as(images)


def train(cfg: dict, device: torch.device) -> None:
    train_data, val_data = [load_group(cfg, split) for split in ("train", "val")]
    settings = cfg["training"]["understanding"]
    epochs, batch_size = settings["epochs"], settings["batch_size"]
    if epochs < 1:
        raise ValueError("understanding epochs must be positive")
    model = UnderstandingMLP(cfg).to(device)
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=settings["lr"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    clip = cfg["training"]["grad_clip"]
    permutations = dihedral_permutations(cfg["image_size"], device)
    amp = mixed_precision(cfg, device)
    rng = np.random.default_rng(cfg["seed"])

    def run_epoch(data: dict, train_mode: bool) -> dict:
        model.train(train_mode)
        n = len(data["images"])
        labels = task_labels(data, TASK_UNDERSTAND)
        order = rng.permutation(n) if train_mode else np.arange(n)
        loss_sum = 0.0
        correct = torch.zeros(4, device=device)
        with torch.set_grad_enabled(train_mode):
            for part in batch_slices(n, batch_size):
                idx = order[part]
                x = take_images(data, idx).to(device)
                if train_mode and settings["augment"]:
                    x = augment_images(x, permutations)
                prompt = torch.as_tensor(data["prompts_und"][idx], device=device, dtype=torch.long)
                targets = {k: torch.as_tensor(v[idx], device=device) for k, v in labels.items()}
                with amp():
                    outputs = model(x, prompt)
                    loss = sum(F.cross_entropy(outputs[key], targets[key]) for key in ATTRIBUTES)
                if train_mode:
                    optimize(loss, optimizer, params, clip)
                loss_sum += loss.detach().item() * len(idx)
                matches = torch.stack([outputs[key].detach().argmax(1) == targets[key] for key in ATTRIBUTES])
                correct[:3] += matches.sum(dim=1)
                correct[3] += matches.all(dim=0).sum()
        return {
            "loss": loss_sum / n,
            **{f"{key}_accuracy": value for key, value in zip((*ATTRIBUTES, "joint"), (correct / n).tolist())},
        }

    history = []
    best = -1.0
    print(f"Stage A on {device}: {sum(p.numel() for p in params):,} trainable parameters")
    with writer_for(cfg, "understanding") as writer:
        for epoch in range(1, epochs + 1):
            train_metrics, val_metrics = run_epoch(train_data, True), run_epoch(val_data, False)
            metrics = {
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
                "lr": optimizer.param_groups[0]["lr"],
            }
            log_epoch(writer, history, epoch, metrics)
            score = val_metrics["joint_accuracy"]
            if score > best:
                best = score
                save_checkpoint(cfg, "understanding", model.state_dict(), epoch, score)
            scheduler.step()
            print(f"[understanding {epoch}/{epochs}] loss={train_metrics['loss']:.4f} val_joint={score:.3f}", flush=True)
    write_history(cfg, "understanding", history)


def main() -> None:
    args = arguments(__doc__).parse_args()
    train(*setup(args))


if __name__ == "__main__":
    main()
