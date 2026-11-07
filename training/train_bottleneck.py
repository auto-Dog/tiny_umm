"""Post-train a connected UMM with FM, a latent L2 penalty, or a variational IB."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from data.dataset import load_group
from data.schema import save_json
from inference import positive_int
from models.bottleneck import GaussianBottleneck, latent_mean
from models.checkpoints import checkpoint_path, flow_state, load_flow, save_checkpoint
from models.flow import flow_matching_errors
from models.tiny_umm import TinyUMM

from .common import arguments, batch_slices, mixed_indices, mixed_precision, optimize, setup
from .train_flow import flow_batch

GENERATION_RATIO = 0.5
DEFAULT_BETA = 3e-6


@torch.no_grad()
def validate(
    model: TinyUMM, cfg: dict, device: torch.device, data: dict, batch_size: int, samples: int,
) -> dict:
    """Compare task-specific FM with the same noise and time at the mean and sampled latent."""
    model.eval()
    seed = cfg["seed"] + 10
    n = min(samples, len(data["images"]))
    indices = np.sort(np.random.default_rng(seed).choice(len(data["images"]), size=n, replace=False))
    amp = mixed_precision(cfg, device)
    result = {}
    for task, edit in (("generate", False), ("edit", True)):
        fm_noise = torch.Generator(device=device).manual_seed(seed + int(edit))
        sampled_fm_noise = torch.Generator(device=device).manual_seed(seed + int(edit))
        latent_noise = torch.Generator(device=device).manual_seed(seed + 2 + int(edit))
        metrics = {"mean_fm": 0.0, "mean_l2": 0.0}
        if model.bottleneck is not None:
            metrics.update(sampled_fm=0.0, kl_nats=0.0, posterior_std=0.0)
        for part in batch_slices(n, batch_size):
            idx = indices[part]
            source, target, prompt = flow_batch(data, idx, np.full(len(idx), edit), device)
            with amp():
                mean = model.condition(prompt, source if edit else None)
                metrics["mean_fm"] += flow_matching_errors(model.flow, target, mean, fm_noise).sum().item()
                metrics["mean_l2"] += (0.5 * latent_mean(mean).square().sum()).item()
                if model.bottleneck is not None:
                    condition, kl, std = model.bottleneck(mean, latent_noise)
                    metrics["sampled_fm"] += flow_matching_errors(model.flow, target, condition, sampled_fm_noise).sum().item()
                    metrics["kl_nats"] += kl.sum().item()
                    metrics["posterior_std"] += std.sum().item()
        result[task] = {key: value / n for key, value in metrics.items()}
    return {"scenes": n, "tasks": result}


def posttrain(cfg: dict, device: torch.device, args) -> Path:
    base = args.base_checkpoint or checkpoint_path(cfg, "connector")
    model, saved = load_flow(cfg, base, device)
    if saved["kind"] not in ("connector", "joint", "adapter") or saved["direct_source"]:
        raise ValueError("post-training requires a Query Connector checkpoint with direct_source=False")
    if args.mode == "ib":
        model.bottleneck = GaussianBottleneck(cfg).to(device)
    batch_size = args.batch_size or cfg["training"]["connector"]["batch_size"]
    lr = cfg["training"]["adapter"]["lr"] if args.lr is None else args.lr
    beta = (0.0 if args.mode == "continue" else DEFAULT_BETA) if args.beta is None else args.beta
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("--lr must be finite and positive")
    if not math.isfinite(beta) or beta < 0 or (args.mode == "continue" and beta != 0):
        raise ValueError("--beta must be finite and nonnegative, and zero in continue mode")
    seed = cfg["seed"] if args.seed is None else args.seed
    if seed < 0:
        raise ValueError("--seed must be nonnegative")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    train, val = (load_group(cfg, split) for split in ("train", "val"))
    if not len(train["images"]) or not len(val["images"]):
        raise ValueError("post-training needs nonempty train and val splits")
    before = validate(model, cfg, device, val, batch_size, args.val_samples)
    model.train_bridge(joint_instruction=True)
    if model.bottleneck is not None:
        model.bottleneck.train().requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg["training"]["weight_decay"])
    rng = np.random.default_rng(seed)
    fm_noise = torch.Generator(device=device).manual_seed(seed + 1)
    latent_noise = torch.Generator(device=device).manual_seed(seed + 2)
    amp = mixed_precision(cfg, device)
    step, seen, seen_edit = 0, 0, 0
    history = []
    totals = torch.zeros(3, device=device)
    logged_steps = 0
    print(f"[{args.mode}] {sum(p.numel() for p in params):,} parameters on {device}, "
          f"beta={beta:g}, {args.steps} steps, val before={before['tasks']}", flush=True)
    while step < args.steps:
        indices, edit = mixed_indices(len(train["images"]), GENERATION_RATIO, rng)
        for part in batch_slices(len(indices), batch_size):
            source, target, prompt = flow_batch(train, indices[part], edit[part], device)
            with amp():
                mean = model.condition(prompt, source)
                condition = mean
                penalty = target.new_zeros(())
                if model.bottleneck is not None:
                    condition, kl, _ = model.bottleneck(mean, latent_noise)
                    penalty = kl.mean()
                elif args.mode == "l2":
                    penalty = 0.5 * latent_mean(mean).square().sum(dim=1).mean()
                fm = flow_matching_errors(model.flow, target, condition, fm_noise).mean()
                loss = fm + beta * penalty
            optimize(loss, optimizer, params, cfg["training"]["grad_clip"])
            step += 1
            seen += len(prompt)
            seen_edit += int(edit[part].sum())
            totals += torch.stack((fm.detach(), penalty.detach(), loss.detach()))
            if step % args.log_every == 0 or step == args.steps:
                fm_avg, penalty_avg, loss_avg = (totals / (step - logged_steps)).tolist()
                history.append({"step": step, "fm": fm_avg, "penalty": penalty_avg, "loss": loss_avg})
                print(f"[{args.mode} {step}/{args.steps}] FM={fm_avg:.5f} penalty={penalty_avg:.4f} "
                      f"loss={loss_avg:.5f}", flush=True)
                totals.zero_()
                logged_steps = step
            if step == args.steps:
                break
    after = validate(model, cfg, device, val, batch_size, args.val_samples)
    score = float(np.mean([task["mean_fm"] for task in after["tasks"].values()]))
    if not math.isfinite(score):
        raise RuntimeError("validation Flow Matching loss is not finite")
    settings = {
        "base_checkpoint": str(base.resolve()), "base_epoch": saved["epoch"], "mode": args.mode,
        "seed": seed, "steps": step, "batch_size": batch_size, "lr": lr, "beta": beta,
        "generation_ratio": GENERATION_RATIO, "generate_examples": seen - seen_edit, "edit_examples": seen_edit,
        "device": str(device),
    }
    output_cfg = {**cfg, "paths": {**cfg["paths"], "checkpoint_dir": str(run_dir)}}
    path = save_checkpoint(
        output_cfg, "posttrain", flow_state(model, connected=True), saved["epoch"], score,
        direct_source=False, posttrain=settings,
    )
    save_json(run_dir / "history.json", {
        **settings, "validation_before": before, "validation_after": after, "train": history,
    })
    return path


def main() -> None:
    parser = arguments(__doc__)
    parser.add_argument("--mode", choices=("continue", "l2", "ib"), required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True, help="new directory for checkpoint and metrics")
    parser.add_argument("--steps", type=positive_int, required=True, help="number of optimizer updates")
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--beta", type=float, help="weight on KL or latent L2, summed over latent dimensions")
    parser.add_argument("--seed", type=int, help="seed for training order, FM noise, and latent noise")
    parser.add_argument("--device", help="override the configured training device")
    parser.add_argument("--val-samples", type=positive_int, default=512)
    parser.add_argument("--log-every", type=positive_int, default=100)
    args = parser.parse_args()
    try:
        cfg, device = setup(args, device_override=args.device)
        print(f"Checkpoint: {posttrain(cfg, device, args)}", flush=True)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
