"""Task-conditioned matching and training-only control of the query KL weight."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data.dataset import normalize_images, task_labels
from data.schema import TASK_UNDERSTAND
from models.flow import FlowCondition
from models.understanding import ATTRIBUTES, class_counts

from .common import batch_slices, mixed_precision
from .train_flow import flow_batch


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


class MatchingPairs:
    """Negatives stay in one split and preserve the edit's three target attributes."""

    def __init__(self, data: dict, seed: int, label_fraction: float = 1.0):
        self.data = data
        n = len(data["images"])
        rng = np.random.default_rng(seed)
        self.labeled = rng.random(n) < label_fraction
        if not self.labeled.any():
            self.labeled[rng.integers(n)] = True
        self.negative = np.full(n, -1, dtype=np.int64)
        labels = np.stack((data["counts"], data["edit_colors"], data["edit_shapes"]), axis=1)
        _, groups = np.unique(labels, axis=0, return_inverse=True)
        for group in np.unique(groups):
            rows = rng.permutation(np.flatnonzero(groups == group))
            if len(rows) > 1:
                self.negative[rows] = np.roll(rows, 1)
        different = (data["edit_targets"] != data["edit_targets"][self.negative.clip(min=0)]).any(axis=(1, 2, 3))
        self.negative[~different] = -1

    def fingerprint(self) -> str:
        digest = sha256(self.data["scene_ids"].astype("<i8", copy=False).tobytes())
        digest.update(self.negative.astype("<i8", copy=False).tobytes())
        digest.update(self.labeled.tobytes())
        return digest.hexdigest()


def negative_answers(labels: torch.Tensor, sizes: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    """Change exactly one answer slot, including schemas with singleton slots."""
    fields = torch.tensor([i for i, size in enumerate(sizes) if size > 1], device=labels.device)
    if not len(fields):
        raise ValueError("ITM needs an answer slot with at least two classes")
    field = fields[torch.randint(len(fields), (len(labels),), generator=generator, device=labels.device)]
    cardinality = torch.tensor(sizes, device=labels.device)[field]
    offset = 1 + (torch.rand(len(labels), generator=generator, device=labels.device) * (cardinality - 1)).long()
    negative = labels.clone()
    rows = torch.arange(len(labels), device=labels.device)
    negative[rows, field] = (negative[rows, field] + offset) % cardinality
    return negative


class TaskMatcher(nn.Module):
    """Match shared Flow features to answer text or an edit's candidate target image."""

    def __init__(self, cfg: dict):
        super().__init__()
        model = cfg["model"]
        width, token = 128, model["token_dim"]
        feature_dim = 16 * (model["flow"]["width"] + 3)
        self.query = nn.Sequential(nn.Linear(feature_dim, width), nn.LayerNorm(width), nn.GELU())
        self.words = nn.ModuleList(nn.Embedding(size, token) for size in class_counts(cfg))
        self.text = nn.Sequential(nn.Linear(3 * token, width), nn.LayerNorm(width))
        self.image = nn.Sequential(
            nn.Linear(model["visual_tokens"] * model["visual_dim"], width), nn.LayerNorm(width),
        )
        self.match = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(), nn.Linear(width, 1))

    def candidates(self, labels: torch.Tensor) -> torch.Tensor:
        return self.text(torch.cat([embedding(labels[:, i]) for i, embedding in enumerate(self.words)], dim=1))

    def errors(
        self, features: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query(features)
        logits = []
        for candidate in (positive, negative):
            fused = torch.cat((query, candidate, query * candidate, (query - candidate).abs()), dim=1)
            logits.append(self.match(fused).squeeze(1).float())
        pos, neg = logits
        errors = (F.softplus(-pos) + F.softplus(neg)) / 2
        accuracy = ((pos >= 0).float() + (neg < 0).float()) / 2
        return errors, accuracy


class DynamicBeta(nn.Module):
    """Predict safe compression from input-derived features, independently of the rate gradient."""

    def __init__(self, cfg: dict):
        super().__init__()
        width = 16 * (cfg["model"]["flow"]["width"] + 3) + cfg["model"]["instruction_dim"]
        self.predict = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, features: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        return self.predict(torch.cat((features.detach(), instruction.detach()), dim=1)).squeeze(1).float()

    @staticmethod
    def multipliers(logits: torch.Tensor, active: torch.Tensor, span: float) -> torch.Tensor:
        safety = logits.detach().sigmoid()
        # The active batch mean is exactly one and each weight is in [1-span, 1+span].
        centered = safety - masked_mean(safety, active)
        return torch.where(active, 1 + span * centered, torch.ones_like(safety))


@dataclass
class MatchingBatch:
    positive_text: torch.Tensor
    negative_text: torch.Tensor
    positive_image: torch.Tensor
    negative_image: torch.Tensor
    text_active: torch.Tensor
    edit_active: torch.Tensor
    edit_rows: torch.Tensor


class QueryObjectives(nn.Module):
    def __init__(self, cfg: dict, args):
        super().__init__()
        self.matcher = TaskMatcher(cfg)
        self.controller = DynamicBeta(cfg) if args.objective == "dynamic_itm" else None
        self.sizes = class_counts(cfg)
        self.temperature = args.controller_temperature
        self.span = args.beta_span
        self.erase_probability = args.probe_erase
        self.itm_weight = args.itm_weight
        self.controller_weight = args.controller_weight

    def optimizer_groups(self, lr: float) -> list[dict]:
        groups = [{"name": "itm", "params": list(self.matcher.parameters()), "lr": lr}]
        if self.controller is not None:
            groups.append({"name": "beta_controller", "params": list(self.controller.parameters()), "lr": lr})
        return groups

    def prepare(
        self, pairs: MatchingPairs, indices: np.ndarray, labels: torch.Tensor, edit: torch.Tensor,
        encoder: nn.Module, negative_rng: torch.Generator,
    ) -> MatchingBatch:
        device = labels.device
        wrong = negative_answers(labels, self.sizes, negative_rng)
        positive_text = self.matcher.candidates(labels)
        negative_text = self.matcher.candidates(wrong)
        partner = pairs.negative[indices]
        targets = np.concatenate((pairs.data["edit_targets"][indices], pairs.data["edit_targets"][partner.clip(min=0)]))
        with torch.no_grad():
            spatial = encoder(torch.as_tensor(normalize_images(targets), device=device)).flatten(1)
        positive_image, negative_image = self.matcher.image(spatial).chunk(2)
        labeled = torch.as_tensor(pairs.labeled[indices], device=device)
        return MatchingBatch(
            positive_text, negative_text, positive_image, negative_image, labeled,
            edit & labeled & torch.as_tensor(partner >= 0, device=device), edit,
        )

    def losses(
        self, model, condition: FlowCondition, text_condition: FlowCondition, text_features: torch.Tensor,
        batch: MatchingBatch, probe_rng: torch.Generator, dynamic_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        edit_features = model.query_features(condition)
        edit_errors, edit_accuracy = self.matcher.errors(edit_features, batch.positive_image, batch.negative_image)
        text_errors, text_accuracy = self.matcher.errors(text_features, batch.positive_text, batch.negative_text)
        itm = (masked_mean(edit_errors, batch.edit_active) + masked_mean(text_errors, batch.text_active)) / 2
        extra_loss = self.itm_weight * itm
        edit_multiplier = torch.ones_like(edit_errors)
        text_multiplier = torch.ones_like(text_errors)
        metrics = {
            "itm_loss": itm, "itm_edit_accuracy": masked_mean(edit_accuracy, batch.edit_active),
            "itm_text_accuracy": masked_mean(text_accuracy, batch.text_active),
            "itm_edit_loss": masked_mean(edit_errors, batch.edit_active),
            "itm_text_loss": masked_mean(text_errors, batch.text_active),
            "itm_edit_coverage": batch.edit_active.float().mean(),
        }
        if self.controller is not None:
            controller_losses, harms = [], []
            multipliers = []
            for current, features, errors, positive, negative, active, allocation in (
                (condition, edit_features, edit_errors, batch.positive_image, batch.negative_image,
                 batch.edit_active, batch.edit_rows),
                (text_condition, text_features, text_errors, batch.positive_text, batch.negative_text,
                 batch.text_active, torch.ones_like(batch.text_active)),
            ):
                with torch.no_grad():
                    erase = torch.rand((*current.visual.shape[:2], 1), device=current.visual.device, generator=probe_rng)
                    probe_visual = torch.where(erase < self.erase_probability, model.flow.null_visual, current.visual)
                    probe = FlowCondition(current.instruction, probe_visual)
                    probe_errors, _ = self.matcher.errors(model.query_features(probe), positive, negative)
                    harm = probe_errors - errors.detach()
                    safety = (-harm / self.temperature).sigmoid()
                logits = self.controller(features, current.instruction)
                controller_errors = F.binary_cross_entropy_with_logits(logits, safety, reduction="none")
                controller_losses.append(masked_mean(controller_errors, active))
                harms.append(masked_mean(harm, active))
                multipliers.append(self.controller.multipliers(logits, allocation, self.span * dynamic_scale))
            edit_multiplier, text_multiplier = multipliers
            controller_loss = torch.stack(controller_losses).mean()
            extra_loss = extra_loss + self.controller_weight * controller_loss
            metrics.update(controller_loss=controller_loss, probe_itm_harm=torch.stack(harms).mean())
        return extra_loss, edit_multiplier, text_multiplier, metrics


@torch.inference_mode()
def validate_matching(auxiliary, model, encoder, cfg, data, device, args) -> dict:
    seed = cfg["seed"] + 31
    n = min(args.val_samples, len(data["images"]))
    indices = np.sort(np.random.default_rng(seed).choice(len(data["images"]), n, replace=False))
    pairs = MatchingPairs(data, seed)
    labels = task_labels(data, TASK_UNDERSTAND)
    gold = np.stack([labels[key] for key in ATTRIBUTES], axis=1)
    generators = [torch.Generator(device=device).manual_seed(seed + i) for i in range(4)]
    amp = mixed_precision(cfg, device)
    totals = np.zeros(7)
    for part in batch_slices(n, args.batch_size):
        rows = indices[part]
        source, _, prompt = flow_batch(data, rows, np.ones(len(rows), dtype=bool), device)
        text_prompt = model.understand_tokens[None].expand(len(rows), -1)
        with amp():
            edit = model.bottleneck(model.condition(prompt, source), generators[0], prompt=prompt)[0]
            text = model.bottleneck(model.condition(text_prompt, source), generators[1], prompt=text_prompt)[0]
            batch = auxiliary.prepare(
                pairs, rows, torch.as_tensor(gold[rows], device=device), torch.ones(len(rows), device=device, dtype=torch.bool),
                encoder, generators[2],
            )
            _, multiplier, _, metrics = auxiliary.losses(
                model, edit, text, model.query_features(text), batch, generators[3], 1.0,
            )
        available = int(batch.edit_active.sum())
        totals += np.array([
            metrics["itm_text_loss"].item() * len(rows), metrics["itm_text_accuracy"].item() * len(rows),
            metrics["itm_edit_loss"].item() * available, metrics["itm_edit_accuracy"].item() * available,
            available, multiplier.sum().item(), multiplier.square().sum().item(),
        ])
    beta_mean = args.edit_beta * totals[5] / n
    return {
        "text_loss": totals[0] / n, "text_accuracy": totals[1] / n,
        "edit_loss": totals[2] / max(1, totals[4]), "edit_accuracy": totals[3] / max(1, totals[4]),
        "edit_pair_coverage": totals[4] / n, "edit_beta_mean": beta_mean,
        "edit_beta_std": args.edit_beta * max(0.0, totals[6] / n - (totals[5] / n) ** 2) ** 0.5,
    }
