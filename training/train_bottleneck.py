"""Continue Query Connector training with FM or a visual bottleneck."""
from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data.dataset import load_group, normalize_images
from data.paired_edits import read_pairs
from data.schema import OP_EDIT, save_json
from inference import positive_int
from models.bottleneck import ConditionalVisualBottleneck, GaussianBottleneck, latent_mean
from models.checkpoints import checkpoint_path, flow_state, load_flow, save_checkpoint
from models.flow import flow_matching_errors
from models.tiny_umm import TinyUMM

from .common import arguments, batch_slices, mixed_indices, mixed_precision, optimize, setup
from .train_flow import flow_batch

GENERATION_RATIO = 0.5
DEFAULT_BETA = 3e-6
CONDITIONAL_BETA = 0.002
TargetReference = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def frozen_target_reference(model: TinyUMM, encoder: str) -> TargetReference:
    teacher = deepcopy(model).eval().requires_grad_(False)
    if encoder == "source":
        return lambda prompt, target: teacher.source_encoder(target)
    return lambda prompt, target: teacher.condition(prompt, target).visual


def online_target_reference(model: TinyUMM) -> TargetReference:
    @torch.no_grad()
    def reference(prompt: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return model.condition(prompt, target).visual

    return reference


def text_retention(student: TinyUMM, teacher: TinyUMM, source: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        reference = teacher.classify(source)
    actual = student.classify(source)
    return torch.stack([
        F.kl_div(actual[key].float().log_softmax(dim=1), logits.float().softmax(dim=1), reduction="batchmean")
        for key, logits in reference.items()
    ]).mean()


def layout_relevance(head: torch.nn.Linear, visual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Score a layout readout of the first query against a target-derived coverage map."""
    border = torch.cat((target[:, 0], target[:, -1], target[:, 1:-1, 0], target[:, 1:-1, -1]), dim=1)
    background = border.median(dim=1).values[:, None, None]
    contrast = torch.linalg.vector_norm(target - background, dim=-1)
    coverage = ((contrast - 0.16) / 0.64).clamp(0, 1)
    pooled = F.avg_pool2d(coverage[:, None], 4).flatten(1)
    return (head(visual[:, 0]).float().sigmoid() - pooled).square().mean(dim=1)


def count_relevance(
    teacher: TinyUMM, count: torch.Tensor, weight: torch.Tensor,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def score(clean: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        logits = teacher.classify(clean)["count"].float()
        return weight * (4 * time * (1 - time)) * F.cross_entropy(logits, count, reduction="none")

    return score


@torch.no_grad()
def validate(
    model: TinyUMM, cfg: dict, device: torch.device, data: dict, batch_size: int, samples: int,
    *, target_reference: TargetReference | None = None, same_color_edits: bool = False,
    rate_diagnostic: bool = True, source_reference: bool = False,
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
            metrics.update(sampled_fm=0.0, posterior_std=0.0)
            if rate_diagnostic:
                metrics["kl_nats"] = 0.0
        for part in batch_slices(n, batch_size):
            idx = indices[part]
            source, target, prompt = flow_batch(data, idx, np.full(len(idx), edit), device)
            with amp():
                mean = model.condition(prompt, source if edit else None)
                metrics["mean_fm"] += flow_matching_errors(model.flow, target, mean, fm_noise).sum().item()
                metrics["mean_l2"] += (0.5 * latent_mean(mean).square().sum()).item()
                if model.bottleneck is not None:
                    reference_image = source if source_reference else target
                    reference = target_reference(prompt, reference_image) if target_reference is not None and edit else None
                    active = model.keeps_source_color(prompt, source) if same_color_edits and edit else None
                    condition, kl, std = model.bottleneck(
                        mean, latent_noise, prompt=prompt, reference=reference, active=active,
                    )
                    metrics["sampled_fm"] += flow_matching_errors(model.flow, target, condition, sampled_fm_noise).sum().item()
                    if rate_diagnostic:
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
    default_beta = {"continue": 0.0, "l2": DEFAULT_BETA, "ib": DEFAULT_BETA, "conditional": CONDITIONAL_BETA}
    beta = default_beta[args.mode] if args.beta is None else args.beta
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("--lr must be finite and positive")
    if not math.isfinite(beta) or beta < 0 or (args.mode == "continue" and beta != 0):
        raise ValueError("--beta must be finite and nonnegative, and zero in continue mode")
    conditional = args.mode == "conditional"
    if conditional and args.steps < 2:
        raise ValueError("conditional mode needs at least two steps to fit and use its prior")
    if args.max_std_ratio is not None and not conditional:
        raise ValueError("--max-std-ratio requires conditional mode")
    if args.target_prior and not conditional:
        raise ValueError("--target-prior requires conditional mode")
    paired_prior = args.target_prior == "paired"
    source_reference = args.target_prior == "anchor"
    if paired_prior and (args.edit_pairs is None or args.same_color_edits):
        raise ValueError("paired prior requires --edit-pairs and applies to both views of each edit")
    if args.edit_pairs is not None and batch_size % 2:
        raise ValueError("--edit-pairs requires an even batch size")
    if args.same_color_edits and not conditional:
        raise ValueError("--same-color-edits requires conditional mode")
    if args.appearance_rank and args.target_prior != "query":
        raise ValueError("--appearance-rank requires a query target prior")
    if args.protected_tokens and not conditional:
        raise ValueError("--protected-tokens requires conditional mode")
    max_std_ratio = 0.30 if args.max_std_ratio is None else args.max_std_ratio
    if not 0.02 <= max_std_ratio <= 1.0:
        raise ValueError("--max-std-ratio must be in [0.02, 1]")
    if not 0 <= args.generation_ratio < 1:
        raise ValueError("--generation-ratio must be in [0, 1)")
    if not math.isfinite(args.spatial_teacher) or args.spatial_teacher < 0:
        raise ValueError("--spatial-teacher must be finite and nonnegative")
    if args.spatial_teacher_kind == "attention" and (args.adapters or args.full_flow):
        raise ValueError("attention supervision requires a frozen Flow decoder")
    tune_features = args.encoder_lr is not None
    if tune_features and (not math.isfinite(args.encoder_lr) or args.encoder_lr <= 0):
        raise ValueError("--encoder-lr must be finite and positive")
    if not math.isfinite(args.text_retention) or args.text_retention < 0:
        raise ValueError("--text-retention must be finite and nonnegative")
    if not math.isfinite(args.layout_weight) or args.layout_weight < 0:
        raise ValueError("--layout-weight must be finite and nonnegative")
    if not math.isfinite(args.count_weight) or args.count_weight < 0:
        raise ValueError("--count-weight must be finite and nonnegative")
    seed = cfg["seed"] if args.seed is None else args.seed
    if seed < 0:
        raise ValueError("--seed must be nonnegative")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    train, val = (load_group(cfg, split) for split in ("train", "val"))
    if not len(train["images"]) or not len(val["images"]):
        raise ValueError("post-training needs nonempty train and val splits")
    peer_images = read_pairs(args.edit_pairs, train) if args.edit_pairs is not None else None
    before = validate(model, cfg, device, val, batch_size, args.val_samples)
    target_reference = None
    if args.target_prior == "online":
        target_reference = online_target_reference(model)
    elif args.target_prior and not paired_prior:
        target_reference = frozen_target_reference(model, args.target_prior)
    teacher = None
    if args.spatial_teacher or tune_features or args.count_weight:
        teacher = deepcopy(model).eval().requires_grad_(False)
        teacher.bridge = None
    model.train_bridge(joint_instruction=not args.freeze_instruction, adapters=args.adapters)
    if args.full_flow:
        model.flow.train().requires_grad_(True)
    prior_window = min(128, max(1, args.steps // 10)) if conditional else 0
    prior_steps = max(prior_window, args.steps // 2) if conditional and args.curriculum else prior_window
    prior_start = prior_steps - prior_window
    ramp_steps = min(256, max(1, args.steps // 5)) if conditional else 0
    if conditional:
        bottleneck = ConditionalVisualBottleneck(
            cfg, max_std_ratio=max_std_ratio, appearance_rank=args.appearance_rank,
            protected_tokens=args.protected_tokens,
        ).to(device)
        model.bottleneck = bottleneck
        counts = torch.zeros(len(bottleneck.prior_mean), device=device)
        first = torch.zeros_like(bottleneck.prior_mean)
        second = torch.zeros_like(bottleneck.prior_mean)
        latent_dims = bottleneck.prior_mean[0].numel()
        differences, prior_groups = [], []
    if model.bottleneck is not None:
        model.bottleneck.train().requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    layout_head = None
    if args.layout_weight:
        layout_head = torch.nn.Linear(
            cfg["model"]["visual_dim"], (cfg["image_size"] // 4) ** 2, device=device,
        )
        torch.nn.init.zeros_(layout_head.weight)
        torch.nn.init.constant_(layout_head.bias, math.log(1 / 3))
        params.extend(layout_head.parameters())
    param_groups = [{"params": params}]
    if tune_features:
        encoder_params = list(model.understanding.backbone.requires_grad_(True).parameters())
        param_groups.append({"params": encoder_params, "lr": args.encoder_lr})
        params = params + encoder_params
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps) if args.curriculum else None
    rng = np.random.default_rng(seed)
    order_hash = sha256()
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
        indices, edit = mixed_indices(len(train["images"]), args.generation_ratio, rng)
        for part in batch_slices(len(indices), batch_size if peer_images is None else batch_size // 2):
            idx, is_edit = indices[part], edit[part]
            pair_count = len(idx)
            if peer_images is not None:
                idx, is_edit = np.tile(idx, 2), np.tile(is_edit, 2)
            order_hash.update(idx.astype("<i8", copy=False).tobytes())
            order_hash.update(is_edit.tobytes())
            source, target, prompt = flow_batch(train, idx, is_edit, device)
            if peer_images is not None:
                peer_rows = np.flatnonzero(is_edit[pair_count:]) + pair_count
                source[torch.as_tensor(peer_rows, device=device)] = torch.as_tensor(
                    normalize_images(peer_images[idx[peer_rows]]), device=device,
                )
            with amp():
                mean = model.condition(prompt, source, train_features=tune_features)
                reference_image = source if source_reference else target
                reference = target_reference(prompt, reference_image) if target_reference is not None else None
                if paired_prior:
                    first_view, second_view = mean.visual.detach().float().chunk(2, dim=0)
                    midpoint = (first_view + second_view) / 2
                    reference = torch.cat((midpoint, midpoint), dim=0)
                active = model.keeps_source_color(prompt, source) if args.same_color_edits else None
                condition = mean
                penalty = target.new_zeros(())
                weight = beta
                if prior_start <= step < prior_steps:
                    group = bottleneck.prior_index(prompt)
                    edit_rows = prompt[:, 0] == OP_EDIT
                    if active is not None:
                        edit_rows = edit_rows & active
                    visual = mean.visual.detach().float()
                    if reference is not None:
                        visual = visual - reference.float()
                    visual = visual[:, args.protected_tokens:]
                    if args.appearance_rank:
                        differences.append(visual[edit_rows])
                        prior_groups.append(group[edit_rows])
                        if step + 1 == prior_steps:
                            bottleneck.fit_appearance_prior(torch.cat(differences), torch.cat(prior_groups))
                            differences.clear()
                            prior_groups.clear()
                    else:
                        observed = edit_rows.float()
                        visual = visual * observed[:, None, None]
                        counts.index_add_(0, group, observed)
                        if args.target_prior not in ("source", "anchor"):
                            first.index_add_(0, group, visual)
                        second.index_add_(0, group, visual.square())
                        if step + 1 == prior_steps:
                            bottleneck.fit_prior(counts, first, second)
                elif step >= prior_steps and model.bottleneck is not None:
                    condition, kl, _ = model.bottleneck(mean, latent_noise, prompt=prompt, reference=reference, active=active)
                    penalty = kl.mean()
                    if conditional:
                        penalty = penalty / latent_dims
                        weight *= min(1.0, (step - prior_steps + 1) / ramp_steps)
                elif args.mode == "l2":
                    penalty = 0.5 * latent_mean(mean).square().sum(dim=1).mean()
                teacher_velocity = None
                teacher_attention = None
                clean_loss = None
                if args.spatial_teacher:
                    with torch.no_grad():
                        teacher_condition = teacher.condition(prompt, source)
                    teacher_strength = args.spatial_teacher
                    if args.curriculum:
                        teacher_strength *= min(1.0, 2 * (1 - step / args.steps))
                    edit_weight = teacher_strength * (prompt[:, 0] == OP_EDIT)
                    if args.spatial_teacher_kind == "attention":
                        teacher_attention = teacher_condition.visual, edit_weight
                    else:
                        teacher_velocity = teacher.flow, teacher_condition, edit_weight
                if args.count_weight:
                    count = torch.as_tensor(train["counts"][idx], device=device, dtype=torch.long) - 1
                    clean_loss = count_relevance(teacher, count, args.count_weight * (prompt[:, 0] == OP_EDIT))
                fm = flow_matching_errors(
                    model.flow, target, condition, fm_noise, teacher=teacher_velocity, attention_teacher=teacher_attention,
                    clean_loss=clean_loss,
                ).mean()
                loss = fm + weight * penalty
                if tune_features and args.text_retention:
                    loss = loss + args.text_retention * text_retention(model, teacher, source)
                if layout_head is not None:
                    layout = layout_relevance(layout_head, condition.visual, target)
                    loss = loss + args.layout_weight * (layout * (prompt[:, 0] == OP_EDIT)).mean()
            optimize(loss, optimizer, params, cfg["training"]["grad_clip"])
            if scheduler is not None:
                scheduler.step()
            step += 1
            seen += len(prompt)
            seen_edit += int(is_edit.sum())
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
    after = validate(
        model, cfg, device, val, batch_size, args.val_samples,
        target_reference=target_reference, same_color_edits=args.same_color_edits, rate_diagnostic=not paired_prior,
        source_reference=source_reference,
    )
    score = float(np.mean([task.get("sampled_fm", task["mean_fm"]) for task in after["tasks"].values()]))
    if not math.isfinite(score):
        raise RuntimeError("validation Flow Matching loss is not finite")
    settings = {
        "base_checkpoint": str(base.resolve()), "base_epoch": saved["epoch"], "mode": args.mode,
        "seed": seed, "steps": step, "batch_size": batch_size, "lr": lr, "beta": beta,
        "generation_ratio": args.generation_ratio, "generate_examples": seen - seen_edit, "edit_examples": seen_edit,
        "freeze_instruction": args.freeze_instruction,
        "curriculum": args.curriculum, "prior_start": prior_start,
        "encoder_lr": args.encoder_lr, "text_retention": args.text_retention, "spatial_teacher": args.spatial_teacher,
        "spatial_teacher_kind": args.spatial_teacher_kind,
        "layout_weight": args.layout_weight, "count_weight": args.count_weight,
        "device": str(device), "adapters": args.adapters, "full_flow": args.full_flow,
        "data_order_sha256": order_hash.hexdigest(), "prior_steps": prior_steps, "kl_ramp_steps": ramp_steps,
        "trainable_parameters": sum(p.numel() for p in params),
    }
    if conditional:
        settings["max_std_ratio"] = max_std_ratio
        settings["target_prior"] = args.target_prior
        settings["appearance_rank"] = args.appearance_rank
        settings["protected_tokens"] = args.protected_tokens
        settings["same_color_edits"] = args.same_color_edits
    if peer_images is not None:
        settings["edit_pairs"] = str(args.edit_pairs.resolve())
        settings["edit_pairs_sha256"] = sha256(memoryview(peer_images).cast("B")).hexdigest()
    output_cfg = {**cfg, "paths": {**cfg["paths"], "checkpoint_dir": str(run_dir)}}
    path = save_checkpoint(
        output_cfg, "posttrain", flow_state(model, connected=True), saved["epoch"], score,
        direct_source=False, posttrain=settings, **({"bottleneck_kind": "conditional"} if conditional else {}),
    )
    save_json(run_dir / "history.json", {
        **settings, "validation_before": before, "validation_after": after, "train": history,
    })
    return path


def main() -> None:
    parser = arguments(__doc__)
    parser.add_argument("--mode", choices=("continue", "l2", "ib", "conditional"), required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True, help="new directory for checkpoint and metrics")
    parser.add_argument("--steps", type=positive_int, required=True, help="number of optimizer updates")
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--lr", type=float)
    decoder = parser.add_mutually_exclusive_group()
    decoder.add_argument("--adapters", action="store_true", help="also update the Flow conditioning adapters")
    decoder.add_argument("--full-flow", action="store_true", help="also update all parameters of the visual decoder")
    parser.add_argument("--freeze-instruction", action="store_true", help="keep the Stage C instruction encoder frozen")
    parser.add_argument("--generation-ratio", type=float, default=GENERATION_RATIO)
    parser.add_argument(
        "--curriculum", action="store_true",
        help="decay the learning rate, fade the teacher and start the conditional bottleneck halfway through",
    )
    parser.add_argument("--encoder-lr", type=float, help="fine-tune the original image backbone")
    parser.add_argument("--text-retention", type=float, default=0.5, help="text distillation weight when tuning the backbone")
    parser.add_argument("--spatial-teacher", type=float, default=0.0, help="weight of frozen Stage B supervision")
    parser.add_argument("--spatial-teacher-kind", choices=("velocity", "attention"), default="velocity")
    parser.add_argument("--layout-weight", type=float, default=0.0, help="target-layout supervision on the first query")
    parser.add_argument("--count-weight", type=float, default=0.0, help="count preservation on denoised edit predictions")
    parser.add_argument("--edit-pairs", type=Path, help="second training source views produced by data.paired_edits")
    parser.add_argument("--beta", type=float, help="KL or L2 weight (KL per visual dimension in conditional mode)")
    parser.add_argument("--max-std-ratio", type=float, help="cap posterior std relative to the conditional prior std")
    parser.add_argument(
        "--target-prior", nargs="?", const="query", choices=("query", "source", "online", "paired", "anchor"),
        help="conditional training prior from the target, paired views, or the initial source query",
    )
    parser.add_argument(
        "--appearance-rank", type=positive_int, default=0, help="restrict the bottleneck to paired appearance directions",
    )
    parser.add_argument("--protected-tokens", type=positive_int, default=0, help="leading queries that bypass the rate term")
    parser.add_argument("--same-color-edits", action="store_true", help="apply IB only when the edit preserves source color")
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
