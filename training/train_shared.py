"""Train shared TinyUMM with conditional IB, task matching and optional learned KL weights."""
from __future__ import annotations

import math
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from data.dataset import load_group, task_labels
from data.schema import OP_EDIT, TASK_UNDERSTAND, save_json, set_seed
from inference import positive_int
from models.bottleneck import SharedQueryBottleneck
from models.checkpoints import apply_flow_state, checkpoint_path, flow_state, load_flow, save_checkpoint
from models.flow import flow_matching_errors
from models.tiny_umm import TinyUMM
from models.understanding import ATTRIBUTES, class_counts

from .common import arguments, batch_slices, mixed_indices, mixed_precision, setup
from .train_flow import flow_batch
from .query_objectives import MatchingPairs, QueryObjectives, masked_mean, validate_matching
from .signed_beta import CONTEXTS, TEXT_CONTEXT, AlignmentScale, SignedBeta, directional_derivatives, grouped_nce


def labels_at(labels: dict, idx: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.stack([labels[key][idx] for key in ATTRIBUTES], axis=1), device=device)


def text_errors(logits: dict, labels: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        F.cross_entropy(logits[key].float(), labels[:, i], reduction="none") for i, key in enumerate(ATTRIBUTES)
    ], dim=1).mean(dim=1)


class TargetPrior:
    """Frozen conditional query means calibrated on the training split."""

    def __init__(
        self, model: TinyUMM, cfg: dict, data: dict, labels: dict, device: torch.device, amp, seed: int, kind: str,
    ):
        self.model = deepcopy(model).eval().requires_grad_(False)
        self.colors, self.shapes = class_counts(cfg)[1:]
        self.kind = kind
        self.geometry_basis = None
        self.centers = self.fit(cfg, data, labels, device, amp, seed)

    def groups(self, labels: torch.Tensor) -> torch.Tensor:
        return (labels[:, 0] * self.colors + labels[:, 1]) * self.shapes + labels[:, 2]

    @torch.no_grad()
    def fit(
        self, cfg: dict, data: dict, labels: dict, device: torch.device, amp, seed: int,
    ) -> torch.Tensor:
        n = min(len(data["images"]), 8192)
        indices = np.random.default_rng(seed + 99).choice(len(data["images"]), size=n, replace=False)
        groups = math.prod(class_counts(cfg))
        sums = self.model.flow.null_visual.new_zeros((groups, *self.model.flow.null_visual.shape))
        counts = sums.new_zeros(groups)
        target_features, source_queries, geometry_targets = [], [], []
        for part in batch_slices(n, 128):
            idx = indices[part]
            source, target, edit_prompt = flow_batch(data, idx, np.ones(len(idx), dtype=bool), device)
            prompt = self.model.understand_tokens[None].expand(len(idx), -1)
            group = self.groups(labels_at(labels, idx, device))
            with amp():
                visual = self.model.condition(prompt, source).visual.float()
                if self.kind in ("regressed", "spatial", "geometry"):
                    target_features.append(self.edit_features(edit_prompt, target).float())
                    source_queries.append(self.model.condition(edit_prompt, source).visual.float().flatten(1))
            if self.kind == "geometry":
                count = F.one_hot(labels_at(labels, idx, device)[:, 0], class_counts(cfg)[0])
                mask = torch.as_tensor(data["tgt_masks"][idx] != 0, device=device).float()
                coverage = F.avg_pool2d(mask[:, None], 4).flatten(1)
                geometry_targets.append(torch.cat((count, coverage), dim=1))
            sums.index_add_(0, group, visual)
            counts.index_add_(0, group, torch.ones_like(group, dtype=counts.dtype))
        global_mean = sums.sum(dim=0) / n
        if not target_features:
            return (sums + 4 * global_mean) / (counts[:, None, None] + 4)
        features, queries = torch.cat(target_features), torch.cat(source_queries)
        self.feature_mean = features.mean(dim=0)
        self.feature_scale = features.std(dim=0, unbiased=False).clamp_min(0.05)
        self.query_mean = queries.mean(dim=0)
        features = (features - self.feature_mean) / self.feature_scale
        gram = features.T @ features
        gram.diagonal().add_(0.05 * n)
        self.edit_weights = torch.linalg.solve(gram, features.T @ (queries - self.query_mean))
        if geometry_targets:
            target_geometry = torch.cat(geometry_targets)
            query_scale = queries.std(dim=0, unbiased=False).clamp_min(0.10)
            standardized = (queries - self.query_mean) / query_scale
            gram = standardized.T @ standardized
            gram.diagonal().add_(0.015 * n)
            probe = torch.linalg.solve(gram, standardized.T @ (target_geometry - target_geometry.mean(dim=0)))
            vectors, values, _ = torch.linalg.svd((probe / query_scale[:, None]).cpu().double(), full_matrices=False)
            self.geometry_basis = vectors[:, values > values[0] * 1e-5].to(device=device, dtype=torch.float32)

        # Additive label effects also define a prior for unseen attribute combinations.
        classes = class_counts(cfg)
        combos = torch.cartesian_prod(*(torch.arange(k, device=device) for k in classes))
        design = torch.cat([F.one_hot(combos[:, i], k) for i, k in enumerate(classes)], dim=1).float()
        gram = design.T @ (counts[:, None] * design)
        gram.diagonal().add_(0.005 * n)
        centered_sums = (sums - counts[:, None, None] * global_mean).flatten(1)
        effects = torch.linalg.solve(gram, design.T @ centered_sums)
        return global_mean + (design @ effects).reshape_as(sums)

    def edit_features(self, prompt: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.model.source_encoder(target).flatten(1), self.model.instruction(prompt)), dim=1)

    @torch.no_grad()
    def edit(self, prompt: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.kind == "query":
            return self.model.condition(prompt, target).visual
        if self.kind == "anchor":
            return self.model.condition(prompt, source).visual
        features = self.edit_features(prompt, target).float()
        with torch.autocast(device_type=target.device.type, enabled=False):
            normalized = (features - self.feature_mean) / self.feature_scale
            loc = self.query_mean + normalized @ self.edit_weights
            if self.kind == "spatial":
                loc = loc.lerp(features[:, :loc.shape[1]], 0.30)
        return loc.reshape(-1, *self.model.flow.null_visual.shape)

    def text(self, labels: torch.Tensor) -> torch.Tensor:
        return self.centers[self.groups(labels)]

    def edit_rate(
        self, diagnostic: SharedQueryBottleneck, visual: torch.Tensor, reference: torch.Tensor, prior_std: float,
    ) -> torch.Tensor:
        return diagnostic.rate(
            visual, reference, prior_std, relaxed_basis=self.geometry_basis, relaxed_std=4 * prior_std,
        )


def trainable_groups(model: TinyUMM, lr: float) -> list[dict]:
    model.eval().requires_grad_(False)
    # Keep the pretrained encoder dropout disabled. Posterior sampling has its own matched RNG.
    modules = [
        ("theta_image", model.understanding.backbone, 0.35),
        ("theta_instruction", model.instruction, 1.0),
        ("phi_queries", model.bridge, 2.0),
        ("phi_flow", model.flow, 1.0),
        ("psi_text", model.text_aux, 5.0),
    ]
    return [
        {"name": name, "params": list(module.requires_grad_(True).parameters()), "lr": lr * scale}
        for name, module, scale in modules
    ]


def sampled_condition(model: TinyUMM, mean, generator: torch.Generator, prompt: torch.Tensor):
    if model.bottleneck is None:
        return mean
    return model.bottleneck(mean, generator, prompt=prompt)[0]


@torch.inference_mode()
def validate(
    model: TinyUMM, cfg: dict, data: dict, device: torch.device, args,
    prior: TargetPrior | None, diagnostic: SharedQueryBottleneck | None,
) -> dict:
    seed = cfg["seed"] + 31
    n = min(args.val_samples, len(data["images"]))
    idx = np.sort(np.random.default_rng(seed).choice(len(data["images"]), size=n, replace=False))
    labels = task_labels(data, TASK_UNDERSTAND)
    amp = mixed_precision(cfg, device)
    totals = torch.zeros(8, device=device)
    fm = torch.Generator(device=device).manual_seed(seed)
    latent = torch.Generator(device=device).manual_seed(seed + 1)
    for part in batch_slices(n, args.batch_size):
        rows = idx[part]
        for column, is_edit in enumerate((False, True)):
            source, target, prompt = flow_batch(data, rows, np.full(len(rows), is_edit), device)
            with amp():
                mean = model.condition(prompt, source if is_edit else None)
                condition = sampled_condition(model, mean, latent, prompt)
                totals[column] += flow_matching_errors(model.flow, target, condition, fm).sum()
                if is_edit:
                    visual = mean.visual.float()
                    totals[6] += visual.square().flatten(1).mean(dim=1).sum()
                    if prior is not None:
                        reference = prior.edit(prompt, source, target)
                        totals[4] += prior.edit_rate(diagnostic, visual, reference, args.prior_std).sum()
                        totals[7] += (visual - reference.float()).square().flatten(1).mean(dim=1).sum()
        gold = labels_at(labels, rows, device)
        prompt = model.understand_tokens[None].expand(len(rows), -1)
        with amp():
            mean = model.condition(prompt, source)
            condition = sampled_condition(model, mean, latent, prompt)
            logits = model.read_text(condition)
            totals[2] += text_errors(logits, gold).sum()
            prediction = torch.stack([logits[key].argmax(dim=1) for key in ATTRIBUTES], dim=1)
            totals[3] += (prediction == gold).all(dim=1).sum()
            if prior is not None:
                totals[5] += diagnostic.rate(mean.visual, prior.text(gold), args.prior_std).sum()
    keys = ("generation_fm", "edit_fm", "text_ce", "understanding_joint", "edit_kl_per_dim", "text_kl_per_dim",
            "edit_query_mean_sq", "edit_residual_mean_sq")
    result = dict(zip(keys, (totals / n).tolist()))
    result["edit_query_rms"] = math.sqrt(result.pop("edit_query_mean_sq"))
    residual = result.pop("edit_residual_mean_sq")
    if prior is None:
        del result["edit_kl_per_dim"], result["text_kl_per_dim"]
    else:
        result["edit_residual_rms"] = math.sqrt(residual)
    return {"scenes": n, **result}


@torch.no_grad()
def calibrate_nce_temperature(model: TinyUMM, data: dict, device: torch.device, amp, seed: int) -> torch.Tensor:
    """Fix tau_c^2 to half the mean same-context squared query distance of the starting model."""
    n = min(len(data["images"]), 512)
    idx = np.sort(np.random.default_rng(seed + 77).choice(len(data["images"]), size=n, replace=False))
    queries, contexts = [], []
    for part in batch_slices(n, 128):
        rows = idx[part]
        source, _, prompt = flow_batch(data, rows, np.ones(len(rows), dtype=bool), device)
        text_prompt = model.understand_tokens[None].expand(len(rows), -1)
        with amp():
            queries += [model.condition(prompt, source).visual.float().flatten(1),
                        model.condition(text_prompt, source).visual.float().flatten(1)]
        contexts += [torch.as_tensor(data["edit_types"][rows].astype(np.int64), device=device),
                     torch.full((len(rows),), TEXT_CONTEXT, device=device)]
    queries, contexts = torch.cat(queries), torch.cat(contexts)
    tau_sq = []
    for c in range(len(CONTEXTS)):
        group = queries[contexts == c]
        if len(group) < 2:
            raise ValueError(f"NCE temperature calibration needs two {CONTEXTS[c]} examples")
        distance = torch.cdist(group, group).square()
        tau_sq.append(distance.sum() / (len(group) * (len(group) - 1)) / 2)
    return torch.stack(tau_sq)


def meta_task_loss(model: TinyUMM, data: dict, labels: dict, rng, generators, device, args, amp) -> torch.Tensor:
    """Task loss on an independent training batch; uses its own RNG streams only."""
    idx = rng.choice(len(data["images"]), size=args.batch_size)
    is_edit = rng.random(args.batch_size) >= args.generation_ratio
    source, target, prompt = flow_batch(data, idx, is_edit, device)
    text_prompt = model.understand_tokens[None].expand(len(idx), -1)
    with amp():
        condition = sampled_condition(model, model.condition(prompt, source), generators[0], prompt)
        fm_loss = flow_matching_errors(model.flow, target, condition, generators[1]).mean()
        text_condition = sampled_condition(model, model.condition(text_prompt, source), generators[2], text_prompt)
        ce = text_errors(model.read_text(text_condition), labels_at(labels, idx, device)).mean()
    return fm_loss + args.text_weight * ce


def configure_objective(args) -> None:
    """Keep legacy defaults and give the paired ablations the same low-noise channel."""
    uses_ib = args.objective in ("ib", "ib_itm", "dynamic_itm", "signed_itm")
    if args.edit_beta is None:
        args.edit_beta = 0.0006 if uses_ib else 0.0
    if args.posterior_std is None:
        args.posterior_std = 0.003 if args.objective else 0.0
    if args.objective:
        args.target_prior = args.target_prior or "geometry"
        if args.objective in ("baseline", "itm") and (args.edit_beta or args.text_beta):
            raise ValueError("baseline and itm objectives require zero edit and text KL weights")
        if uses_ib and args.edit_beta <= 0:
            raise ValueError("IB objectives require a positive edit KL weight")
        if args.posterior_std <= 0:
            raise ValueError("paired objectives require a positive fixed posterior std")
    if not 0 < args.itm_label_fraction <= 1:
        raise ValueError("ITM label fraction must be in (0, 1]")
    if not 0 <= args.beta_span < 1 or not 0 < args.probe_erase < 1:
        raise ValueError("beta span must be in [0, 1) and probe erasure probability in (0, 1)")
    if args.objective == "signed_itm" and not (
        0 < args.edit_beta < args.beta_max and abs(args.text_beta) < args.beta_max and 0 <= args.signed_span < 0.5
        and args.nce_scale > 0 and 0 < args.fd_step < 1e-2 and args.controller_lr > 0 and args.sign_lr > 0
        and args.beta_anchor > 0
    ):
        raise ValueError("signed_itm needs 0 < edit beta < beta max, |text beta| < beta max, span in [0, 0.5), "
                         "positive NCE scale and controller lr, and a small finite-difference step")
    if any(not math.isfinite(value) or value <= 0 for value in (
        args.itm_weight, args.controller_weight, args.controller_temperature,
    )):
        raise ValueError("matching and controller weights and temperature must be finite and positive")


def train(cfg: dict, device: torch.device, args) -> Path:
    configure_objective(args)
    started = perf_counter()
    if any(not math.isfinite(value) or value <= 0 for value in (args.lr, args.text_weight, args.prior_std)):
        raise ValueError("learning rate, text weight and prior std must be finite and positive")
    if not 0 < args.generation_ratio < 1:
        raise ValueError("generation ratio must be in (0, 1)")
    if args.seed < 0 or not math.isfinite(args.posterior_std) or args.posterior_std < 0:
        raise ValueError("seed and posterior std must be nonnegative")
    if not math.isfinite(args.edit_beta):
        raise ValueError("edit KL weight must be finite")
    if not math.isfinite(args.text_beta) or (args.text_beta < 0 and args.objective != "signed_itm"):
        raise ValueError("text KL weight must be finite and nonnegative")
    if not math.isfinite(args.foreground_weight) or args.foreground_weight < 0:
        raise ValueError("foreground weight must be finite and nonnegative")
    if not math.isfinite(args.spatial_teacher) or args.spatial_teacher < 0:
        raise ValueError("spatial teacher weight must be finite and nonnegative")
    if (args.edit_beta or args.text_beta) and (not args.target_prior or not args.posterior_std):
        raise ValueError("nonzero KL weight needs --target-prior and positive --posterior-std")
    set_seed(args.seed)
    base = args.base_checkpoint or checkpoint_path(cfg, "connector")
    loaded, saved = load_flow(cfg, base, device)
    if saved["kind"] not in ("connector", "joint", "adapter", "shared") or saved["direct_source"]:
        raise ValueError("shared training needs a Query Connector checkpoint with direct_source=False")
    if loaded.text_aux is None:
        model = TinyUMM(cfg, "connector", shared_understanding=True)
        apply_flow_state(model, saved["state"])
        model.to(device)
        del loaded
    else:
        model = loaded
    model.bottleneck = SharedQueryBottleneck(args.posterior_std) if args.posterior_std else None
    model.to(memory_format=torch.channels_last)
    train_data, val = (load_group(cfg, split) for split in ("train", "val"))
    n = len(train_data["images"])
    if not n or not len(val["images"]):
        raise ValueError("shared training needs nonempty train and val splits")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    labels = task_labels(train_data, TASK_UNDERSTAND)
    amp = mixed_precision(cfg, device)
    prior = TargetPrior(
        model, cfg, train_data, labels, device, amp, args.seed, args.target_prior,
    ) if args.target_prior else None
    diagnostic = SharedQueryBottleneck(args.posterior_std or 0.03) if prior is not None else None
    auxiliary = None
    pairs = None
    if args.objective in ("itm", "ib_itm", "dynamic_itm"):
        auxiliary = QueryObjectives(cfg, args).to(device)
        if "matching_state" in saved:
            auxiliary.matcher.load_state_dict(saved["matching_state"]["matcher"])
            if auxiliary.controller is not None and "controller" in saved["matching_state"]:
                auxiliary.controller.load_state_dict(saved["matching_state"]["controller"])
        pairs = MatchingPairs(train_data, args.seed + 4, args.itm_label_fraction)
    signed = None
    if args.objective == "signed_itm":
        with torch.random.fork_rng(devices=[]):
            # Leave the global RNG untouched so data and noise streams stay paired with the other objectives.
            torch.manual_seed(args.seed + 8)
            signed = SignedBeta(cfg, args.beta_max, (args.edit_beta,) * 3 + (args.text_beta,), args.signed_span)
        signed.to(device)
        # Momentum SGD keeps the context step proportional to the alignment; the bounded A head uses AdamW.
        signed_optimizers = (
            torch.optim.SGD(signed.sign.parameters(), lr=args.sign_lr, momentum=0.9),
            torch.optim.AdamW(signed.modulation.parameters(), lr=args.controller_lr,
                              weight_decay=cfg["training"]["weight_decay"]),
        )
        nce_tau_sq = calibrate_nce_temperature(model, train_data, device, amp, args.seed)
        alignment = AlignmentScale(device)
        meta_rng = np.random.default_rng(args.seed + 6)
        meta_generators = [torch.Generator(device=device).manual_seed(args.seed + 6 + i) for i in range(1, 4)]
        signed_initial = signed.table(device)
    teacher_path = checkpoint_path(cfg, "flow") if args.spatial_teacher else None
    teacher = None
    if teacher_path is not None:
        teacher, teacher_saved = load_flow(cfg, teacher_path, device)
        if teacher_saved["kind"] != "flow":
            raise ValueError("spatial distillation requires a Stage B checkpoint")
        teacher.requires_grad_(False).to(memory_format=torch.channels_last)
    groups = trainable_groups(model, args.lr)
    if auxiliary is not None:
        groups += auxiliary.optimizer_groups(args.lr)
    params = [p for group in groups for p in group["params"]]
    upstream = [p for group in groups if group["name"] in ("theta_image", "theta_instruction", "phi_queries")
                for p in group["params"]]
    start_weights = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["training"]["weight_decay"], fused=device.type == "cuda")
    grad_clip = cfg["training"]["grad_clip"]
    fm = torch.Generator(device=device).manual_seed(args.seed + 1)
    latent = torch.Generator(device=device).manual_seed(args.seed + 2)
    text_latent = torch.Generator(device=device).manual_seed(args.seed + 3)
    negatives = torch.Generator(device=device).manual_seed(args.seed + 4)
    probes = torch.Generator(device=device).manual_seed(args.seed + 5)
    rng = np.random.default_rng(args.seed)
    order_hash = sha256()
    history = []
    step, seen, seen_edit, last_log = 0, 0, 0, 0
    totals = torch.zeros(6, device=device)
    matching_totals = {}
    ramp = max(1, args.steps // 5)
    print(f"[shared {args.objective or 'legacy'}] {sum(p.numel() for p in params):,} parameters on {device}, "
          f"seed={args.seed}, beta=({args.edit_beta:g}, {args.text_beta:g}), "
          f"std={args.posterior_std:g}, {args.steps} steps", flush=True)
    while step < args.steps:
        indices, edit = mixed_indices(n, args.generation_ratio, rng)
        for part in batch_slices(len(indices), args.batch_size):
            idx, is_edit = indices[part], edit[part]
            order_hash.update(train_data["scene_ids"][idx].astype("<i8", copy=False).tobytes())
            order_hash.update(is_edit.tobytes())
            source, target, prompt = flow_batch(train_data, idx, is_edit, device)
            pixel_weight = None
            if args.foreground_weight:
                mask = np.where(is_edit[:, None, None], train_data["tgt_masks"][idx], train_data["src_masks"][idx])
                pixel_weight = 1 + args.foreground_weight * torch.as_tensor(mask != 0, device=device)
                pixel_weight = pixel_weight / pixel_weight.mean(dim=(1, 2), keepdim=True)
            gold = labels_at(labels, idx, device)
            text_prompt = model.understand_tokens[None].expand(len(idx), -1)
            with amp():
                mean = model.condition(prompt, source)
                condition = sampled_condition(model, mean, latent, prompt)
                velocity_teacher = None
                if teacher is not None:
                    with torch.no_grad():
                        teacher_condition = teacher.condition(prompt, source, direct=False)
                    velocity_teacher = teacher.flow, teacher_condition, args.spatial_teacher * (prompt[:, 0] == OP_EDIT)
                fm_loss = flow_matching_errors(
                    model.flow, target, condition, fm, teacher=velocity_teacher, pixel_weight=pixel_weight,
                ).mean()
                text_mean = model.condition(text_prompt, source)
                text_condition = sampled_condition(model, text_mean, text_latent, text_prompt)
                text_features = model.query_features(text_condition)
                ce = text_errors(model.text_aux(text_features), gold).mean()
                edit_rows = prompt[:, 0] == OP_EDIT
                extra_loss = fm_loss.new_zeros(())
                edit_multiplier, text_multiplier = torch.ones_like(prompt[:, 0], dtype=torch.float32), 1.0
                matching_metrics = {}
                if auxiliary is not None:
                    batch = auxiliary.prepare(pairs, idx, gold, edit_rows, prior.model.source_encoder, negatives)
                    dynamic_scale = min(1.0, max(0.0, (step + 1 - ramp) / ramp))
                    extra_loss, edit_multiplier, text_multiplier, matching_metrics = auxiliary.losses(
                        model, condition, text_condition, text_features, batch, probes, dynamic_scale,
                    )
                edit_rate, text_rate = fm_loss.new_zeros(()), fm_loss.new_zeros(())
                rate_loss = fm_loss.new_zeros(())
                if prior is not None:
                    edit_reference = prior.edit(prompt, source, target)
                    edit_rates = prior.edit_rate(diagnostic, mean.visual, edit_reference, args.prior_std) * edit_rows
                    text_rates = diagnostic.rate(text_mean.visual, prior.text(gold), args.prior_std)
                    edit_rate, text_rate = edit_rates.mean(), text_rates.mean()
                    if signed is None:
                        rate_loss = args.edit_beta * (edit_rates * edit_multiplier).mean()
                        rate_loss += args.text_beta * (text_rates * text_multiplier).mean()
                    else:
                        edit_types = torch.as_tensor(train_data["edit_types"][idx].astype(np.int64), device=device)
                        edit_context = torch.where(edit_rows, edit_types, -1)
                        text_context = torch.full_like(edit_context, TEXT_CONTEXT)
                        all_rows = torch.ones_like(edit_rows)
                        edit_noise = (condition.visual.float() - mean.visual.float()).detach()
                        edit_nce, _ = grouped_nce(condition.visual, mean.visual, edit_context, nce_tau_sq)
                        text_nce, _ = grouped_nce(text_condition.visual, text_mean.visual, text_context, nce_tau_sq)
                        span_scale = min(1.0, max(0.0, (step + 1 - ramp) / ramp))
                        with torch.no_grad():
                            edit_features = model.query_features(condition)
                        signed_edit = signed(edit_features, mean.instruction, edit_context, edit_rows, span_scale)
                        signed_text = signed(text_features, text_mean.instruction, text_context, all_rows, span_scale)
                        be, bt = signed_edit.detach(), signed_text.detach()
                        # Positive weights use the KL upper bound, negative weights the InfoNCE lower bound.
                        edit_penalty = torch.where(be >= 0, be * edit_rates, args.nce_scale * be * edit_nce)
                        text_penalty = torch.where(bt >= 0, bt * text_rates, args.nce_scale * bt * text_nce)
                        rate_loss = edit_penalty.mean() + text_penalty.mean()
                        matching_metrics.update(
                            edit_nce=masked_mean(edit_nce, edit_rows), text_nce=text_nce.mean(),
                            edit_beta_mean=masked_mean(be, edit_rows),
                            edit_beta_std=masked_mean((be - masked_mean(be, edit_rows)).square(), edit_rows).sqrt(),
                            text_beta_mean=bt.mean(),
                        )
                if auxiliary is not None:
                    beta = args.edit_beta * edit_multiplier
                    beta_mean = masked_mean(beta, edit_rows)
                    matching_metrics.update(
                        edit_beta_mean=beta_mean,
                        edit_beta_std=masked_mean((beta - beta_mean).square(), edit_rows).sqrt(),
                    )
                rate_scale = min(1.0, (step + 1) / ramp)
                loss = fm_loss + args.text_weight * ce + rate_scale * rate_loss + extra_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if signed is not None:
                # One-step lookahead: d L_meta / d beta_i ~ -eta <grad L_meta, grad S_i>, S_i the active bound.
                meta = meta_task_loss(model, train_data, labels, meta_rng, meta_generators, device, args, amp)
                direction = [g if g is not None else torch.zeros_like(p) for p, g in zip(
                    upstream, torch.autograd.grad(meta, upstream, allow_unused=True))]
                text_noise = (text_condition.visual.float() - text_mean.visual.float()).detach()

                def surrogates():
                    with torch.autocast(device_type=device.type, enabled=False):
                        edit_q = model.condition(prompt, source).visual.float()
                        text_q = model.condition(text_prompt, source).visual.float()
                        return (
                            prior.edit_rate(diagnostic, edit_q, edit_reference, args.prior_std) * edit_rows,
                            diagnostic.rate(text_q, prior.text(gold), args.prior_std),
                            grouped_nce(edit_q + edit_noise, edit_q, edit_context, nce_tau_sq)[0],
                            grouped_nce(text_q + text_noise, text_q, text_context, nce_tau_sq)[0],
                        )

                (d_edit_kl, d_text_kl, d_edit_nce, d_text_nce), fd_scale = directional_derivatives(
                    upstream, direction, args.fd_step, surrogates)
                h_edit = torch.where(be >= 0, d_edit_kl, args.nce_scale * d_edit_nce)
                h_text = torch.where(bt >= 0, d_text_kl, args.nce_scale * d_text_nce)
                contexts = torch.cat((edit_context, text_context))
                active = torch.cat((edit_rows, all_rows))
                unit = alignment.normalize(torch.cat((h_edit, h_text)), contexts, active)
                betas = torch.cat((signed_edit, signed_text)) / args.beta_max
                controller_loss = signed.hyper_loss(unit, betas * args.beta_max, contexts, active, args.beta_anchor)
                for opt in signed_optimizers:
                    opt.zero_grad(set_to_none=True)
                controller_loss.backward()
                for opt in signed_optimizers:
                    opt.step()
                for c, name in enumerate(CONTEXTS):
                    rows = active & (contexts == c)
                    key = name.replace("/", "_")
                    matching_metrics[f"align_{key}"] = masked_mean(unit, rows)
                    matching_metrics[f"align_kl_{key}"] = masked_mean(torch.cat((d_edit_kl, d_text_kl)), rows)
                    matching_metrics[f"align_nce_{key}"] = masked_mean(torch.cat((d_edit_nce, d_text_nce)), rows)
                    matching_metrics[f"beta_{key}"] = masked_mean(betas.detach() * args.beta_max, rows)
                matching_metrics["meta_loss"] = meta.detach()
            if step == 0:
                first_gradient_l2 = {
                    group["name"]: sum(
                        p.grad.detach().float().square().sum().item() for p in group["params"] if p.grad is not None
                    ) ** 0.5 for group in groups
                }
                if any(not math.isfinite(value) or (value <= 0 and name not in ("itm", "beta_controller"))
                       for name, value in first_gradient_l2.items()):
                    raise RuntimeError(f"shared training path has missing or nonfinite gradients: {first_gradient_l2}")
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            seen += len(idx)
            seen_edit += int(is_edit.sum())
            edit_rows = (prompt[:, 0] == OP_EDIT).float()
            query_mean_sq = mean.visual.detach().float().square().flatten(1).mean(dim=1)
            query_mean_sq = (query_mean_sq * edit_rows).sum() / edit_rows.sum().clamp_min(1)
            totals += torch.stack((fm_loss, ce, edit_rate, text_rate, loss, query_mean_sq)).detach()
            for name, value in matching_metrics.items():
                matching_totals[name] = matching_totals.get(name, 0.0) + value.detach()
            if step % args.log_every == 0 or step == args.steps:
                values = (totals / (step - last_log)).tolist()
                names = ("fm", "text_ce", "edit_kl_per_example_per_dim", "text_kl_per_dim", "loss")
                query_rms = math.sqrt(values[5])
                matching_log = {name: (value / (step - last_log)).item() for name, value in matching_totals.items()}
                history.append({"step": step, **dict(zip(names, values[:5])), "edit_query_rms": query_rms, **matching_log})
                suffix = (f" ITM={matching_log['itm_loss']:.4f} beta_std={matching_log['edit_beta_std']:.6f}"
                          if "itm_loss" in matching_log else "")
                if signed is not None:
                    suffix = " beta=" + ",".join(f"{v:+.6f}" for v in signed.table(device)) + (
                        f" NCE=({matching_log['edit_nce']:.3f},{matching_log['text_nce']:.3f})")
                print(f"[shared {step}/{args.steps}] FM={values[0]:.5f} CE={values[1]:.5f} "
                      f"edit_KL={values[2]:.4f} text_KL={values[3]:.4f} edit_query_rms={query_rms:.4f}{suffix}", flush=True)
                totals.zero_()
                matching_totals.clear()
                last_log = step
            if step == args.steps:
                break
    parameter_delta_l2 = {}
    offset = 0
    for group in groups:
        weights = group["params"]
        parameter_delta_l2[group["name"]] = sum(
            (p.detach() - previous).float().square().sum().item()
            for p, previous in zip(weights, start_weights[offset:offset + len(weights)])
        ) ** 0.5
        offset += len(weights)
    after = validate(model, cfg, val, device, args, prior, diagnostic)
    if auxiliary is not None:
        after["matching"] = validate_matching(auxiliary, model, prior.model.source_encoder, cfg, val, device, args)
    score = (after["generation_fm"] + after["edit_fm"]) / 2
    if not math.isfinite(score) or not math.isfinite(after["text_ce"]):
        raise RuntimeError("shared validation loss is not finite")
    settings = {
        "seed": args.seed, "dataset_seed": cfg["seed"], "data_order_sha256": order_hash.hexdigest(),
        "base_checkpoint": str(base.resolve()), "base_sha256": sha256(base.read_bytes()).hexdigest(),
        "steps": step, "batch_size": args.batch_size, "generation_ratio": args.generation_ratio,
        "generate_examples": seen - seen_edit, "edit_examples": seen_edit, "text_examples": seen,
        "learning_rates": {g["name"]: g["lr"] for g in groups}, "text_weight": args.text_weight,
        "weight_decay": cfg["training"]["weight_decay"], "grad_clip": grad_clip,
        "posterior_std": args.posterior_std, "prior_std": args.prior_std,
        "rate_diagnostic_std": diagnostic.posterior_std if diagnostic is not None else None,
        "target_prior": args.target_prior, "edit_beta": args.edit_beta, "text_beta": args.text_beta,
        "geometry_prior_rank": (prior.geometry_basis.shape[1] if prior is not None and prior.geometry_basis is not None else 0),
        "foreground_weight": args.foreground_weight,
        "spatial_teacher": args.spatial_teacher,
        "teacher_sha256": sha256(teacher_path.read_bytes()).hexdigest() if teacher_path is not None else None,
        "kl_ramp_steps": ramp, "first_gradient_l2": first_gradient_l2, "parameter_delta_l2": parameter_delta_l2,
        "trainable_parameters": sum(p.numel() for p in params),
        "objective": args.objective or "legacy", "elapsed_seconds": perf_counter() - started,
        "fm_rng_sha256": sha256(fm.get_state().cpu().numpy().tobytes()).hexdigest(),
        "latent_rng_sha256": sha256(latent.get_state().cpu().numpy().tobytes()).hexdigest(),
        "text_latent_rng_sha256": sha256(text_latent.get_state().cpu().numpy().tobytes()).hexdigest(),
    }
    matching_state = {}
    if signed is not None:
        settings.update(
            beta_contexts=list(CONTEXTS), beta_initial=signed_initial, beta_final=signed.table(device),
            beta_max=args.beta_max, signed_span=args.signed_span, nce_scale=args.nce_scale,
            nce_tau_sq=nce_tau_sq.tolist(), fd_step=args.fd_step, controller_lr=args.controller_lr,
            sign_lr=args.sign_lr, beta_anchor=args.beta_anchor,
            modulation_ramp_steps=ramp, finite_difference_scale=fd_scale,
        )
        matching_state["signed_beta"] = signed.state_dict()
    if auxiliary is not None:
        settings.update(
            itm_weight=args.itm_weight, itm_label_fraction=args.itm_label_fraction,
            itm_labeled_scenes=int(pairs.labeled.sum()), itm_edit_negative_scenes=int((pairs.negative >= 0).sum()),
            matching_pairs_sha256=pairs.fingerprint(), beta_span=args.beta_span,
            controller_weight=args.controller_weight, controller_temperature=args.controller_temperature,
            probe_erase=args.probe_erase, dynamic_warmup_steps=ramp, dynamic_ramp_steps=ramp,
        )
        matching_state["matcher"] = auxiliary.matcher.state_dict()
        if auxiliary.controller is not None:
            matching_state["controller"] = auxiliary.controller.state_dict()
    output_cfg = {**cfg, "paths": {**cfg["paths"], "checkpoint_dir": str(args.run_dir)}}
    path = save_checkpoint(
        output_cfg, "shared", flow_state(model, connected=True), saved["epoch"], score, direct_source=False,
        posttrain=settings, **({"bottleneck_kind": "shared_query"} if model.bottleneck is not None else {}),
        **({"matching_state": matching_state} if matching_state else {}),
    )
    save_json(args.run_dir / "history.json", {**settings, "validation": after, "train": history})
    print(f"Validation: {after}", flush=True)
    return path


def _options():
    parser = arguments(__doc__)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=positive_int, required=True)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--text-weight", type=float, default=0.08)
    parser.add_argument("--generation-ratio", type=float, default=0.30)
    parser.add_argument("--posterior-std", type=float, help="fixed query noise, default 0.003 for named objectives")
    parser.add_argument("--prior-std", type=float, default=0.35)
    parser.add_argument(
        "--target-prior", nargs="?", const="query", choices=("query", "regressed", "spatial", "geometry", "anchor"),
        help="target query, target ridge, target ridge with spatial features or geometry covariance, or source anchor",
    )
    parser.add_argument("--edit-beta", type=float, help="edit KL weight, default 0.0006 for named IB objectives")
    parser.add_argument("--text-beta", type=float, default=0.0)
    parser.add_argument("--foreground-weight", type=float, default=0.0, help="extra training FM weight on target foreground")
    parser.add_argument("--spatial-teacher", type=float, default=0.0, help="training-only Stage B edit velocity loss weight")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--val-samples", type=positive_int, default=512)
    parser.add_argument("--log-every", type=positive_int, default=100)
    parser.add_argument("--objective", choices=("baseline", "ib", "itm", "ib_itm", "dynamic_itm", "signed_itm"))
    parser.add_argument("--beta-max", type=float, default=0.002, help="signed_itm bound on |beta(A, C)|")
    parser.add_argument("--signed-span", type=float, default=0.4, help="signed_itm bound on the A modulation")
    parser.add_argument("--nce-scale", type=float, default=1.0, help="signed_itm InfoNCE nats per unit beta")
    parser.add_argument("--fd-step", type=float, default=1e-4, help="relative finite-difference step")
    parser.add_argument("--controller-lr", type=float, default=1e-3, help="signed_itm A-modulation MLP learning rate")
    parser.add_argument("--sign-lr", type=float, default=3e-4, help="signed_itm context MLP momentum-SGD learning rate")
    parser.add_argument("--beta-anchor", type=float, default=0.25, help="signed_itm trust region on context logits")
    parser.add_argument("--itm-weight", type=float, default=0.02)
    parser.add_argument("--itm-label-fraction", type=float, default=1.0, help="fraction of scenes labeling the matching branch")
    parser.add_argument("--controller-weight", type=float, default=0.01)
    parser.add_argument("--controller-temperature", type=float, default=0.05, help="scale of matching harm after erasing queries")
    parser.add_argument("--beta-span", type=float, default=0.9, help="bound on relative beta variation around its batch mean")
    parser.add_argument("--probe-erase", type=float, default=0.5, help="probability of erasing a query in controller probes")
    return parser


def main() -> None:
    parser = _options()
    args = parser.parse_args()
    try:
        cfg, device = setup(args, device_override=args.device)
        print(f"Checkpoint: {train(cfg, device, args)}", flush=True)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
