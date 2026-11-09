"""Variational bottlenecks for the visual decoder."""
from __future__ import annotations

import math

import torch
from torch import nn

from data.schema import OP_EDIT, OP_GENERATE

from .flow import FlowCondition


def latent_mean(condition: FlowCondition) -> torch.Tensor:
    return torch.cat((condition.instruction, condition.visual.flatten(1)), dim=1).float()


class GaussianBottleneck(nn.Module):
    """Diagonal Gaussian with one learned log variance per instruction and visual token."""

    def __init__(self, cfg: dict):
        super().__init__()
        i, d = (cfg["model"][key] for key in ("instruction_dim", "visual_dim"))
        self.instruction_logvar = nn.Linear(i, 1)
        self.visual_logvar = nn.Linear(i + d, 1)
        for head in (self.instruction_logvar, self.visual_logvar):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, 2 * math.log(0.1))

    def forward(
        self, mean: FlowCondition, generator: torch.Generator, *, prompt: torch.Tensor | None = None,
        reference: torch.Tensor | None = None, active: torch.Tensor | None = None,
    ) -> tuple[FlowCondition, torch.Tensor, torch.Tensor]:
        if mean.source is not None:
            raise ValueError("the visual bottleneck requires direct source pixels to be disabled")
        instruction, visual = mean.instruction, mean.visual
        context = instruction[:, None].expand(-1, visual.shape[1], -1)
        logvar_i = self.instruction_logvar(instruction).expand_as(instruction)
        logvar_v = self.visual_logvar(torch.cat((context, visual), dim=-1)).expand_as(visual)
        logvar = torch.cat((logvar_i, logvar_v.flatten(1)), dim=1).float().clamp(-12, 2)
        loc = latent_mean(mean)
        std = (0.5 * logvar).exp()
        noise = torch.randn(loc.shape, device=loc.device, dtype=loc.dtype, generator=generator)
        z = loc + std * noise
        kl = 0.5 * (loc.square() + logvar.expm1() - logvar).sum(dim=1)
        width = instruction.shape[1]
        return FlowCondition(z[:, :width], z[:, width:].reshape_as(visual)), kl, std.mean(dim=1)


class ConditionalVisualBottleneck(nn.Module):
    """Compress the edit queries given the observed instruction."""

    def __init__(
        self, cfg: dict, *, max_std_ratio: float = 0.30, appearance_rank: int = 0, protected_tokens: int = 0,
    ):
        super().__init__()
        if not 0.02 <= max_std_ratio <= 1.0:
            raise ValueError("the maximum posterior std ratio must be in [0.02, 1]")
        model = cfg["model"]
        k, d = model["visual_tokens"], model["visual_dim"]
        if not 0 <= protected_tokens < k or (protected_tokens and appearance_rank):
            raise ValueError("protected tokens require an unprojected bottleneck and at least one compressible token")
        k -= protected_tokens
        if not 0 <= appearance_rank <= k * d:
            raise ValueError("appearance rank exceeds the number of visual coordinates")
        self.appearance_rank = appearance_rank
        self.protected_tokens = protected_tokens
        self.shape_count = len(cfg["shapes"]["names"])
        self.max_log_ratio = math.log(max_std_ratio)
        groups = len(cfg["colors"]["names"]) * self.shape_count
        prior_shape = (groups, appearance_rank) if appearance_rank else (groups, k, d)
        self.register_buffer("prior_mean", torch.zeros(prior_shape))
        self.register_buffer("prior_scale", torch.ones(prior_shape))
        if appearance_rank:
            self.register_buffer("basis", torch.eye(k * d, appearance_rank))
        self.log_ratio = nn.Linear(model["instruction_dim"] + d, 1)
        nn.init.zeros_(self.log_ratio.weight)
        nn.init.constant_(self.log_ratio.bias, math.log(min(0.08, max_std_ratio)))

    def prior_index(self, prompt: torch.Tensor) -> torch.Tensor:
        groups = prompt[:, 2] * self.shape_count + prompt[:, 3]
        return torch.where(prompt[:, 0] == OP_EDIT, groups, 0)

    @torch.no_grad()
    def fit_prior(self, counts: torch.Tensor, first: torch.Tensor, second: torch.Tensor) -> None:
        """Shrink warmup moments toward the marginal for rare edit prompts."""
        n = counts.sum()
        global_mean = first.sum(dim=0) / n.clamp_min(1)
        global_second = second.sum(dim=0) / n.clamp_min(1)
        weight = counts.reshape(-1, *([1] * (first.ndim - 1))) + 32
        loc = (first + 32 * global_mean) / weight
        var = (second + 32 * global_second) / weight - loc.square()
        self.prior_mean.copy_(loc)
        self.prior_scale.copy_(torch.where(n > 0, var.clamp_min(0.15 ** 2).sqrt(), 1.0))

    @torch.no_grad()
    def fit_appearance_prior(self, differences: torch.Tensor, index: torch.Tensor) -> None:
        """Fit the active subspace from same-scene before/after changes."""
        flat = differences.flatten(1)
        if len(flat) == 0:
            return
        centered = flat - flat.mean(dim=0)
        scatter = centered.double().T @ centered.double()
        _, vectors = torch.linalg.eigh(scatter.cpu())
        self.basis.copy_(vectors[:, -self.appearance_rank:])
        with torch.autocast(device_type=flat.device.type, enabled=False):
            projected = flat @ self.basis
        counts = torch.bincount(index, minlength=len(self.prior_mean)).float()
        first = torch.zeros_like(self.prior_mean).index_add_(0, index, projected)
        second = torch.zeros_like(self.prior_mean).index_add_(0, index, projected.square())
        self.fit_prior(counts, first, second)

    def forward(
        self, mean: FlowCondition, generator: torch.Generator, *, prompt: torch.Tensor,
        reference: torch.Tensor | None = None, active: torch.Tensor | None = None,
    ) -> tuple[FlowCondition, torch.Tensor, torch.Tensor]:
        if mean.source is not None:
            raise ValueError("the visual bottleneck requires direct source pixels to be disabled")
        visual_tokens = mean.visual[:, self.protected_tokens:]
        visual = visual_tokens.float()
        index = self.prior_index(prompt)
        edit = prompt[:, 0] == OP_EDIT
        if active is not None:
            edit = edit & active
        context = mean.instruction[:, None].expand(-1, visual.shape[1], -1)
        log_ratio = self.log_ratio(torch.cat((context, visual_tokens), dim=-1)).float()
        if self.appearance_rank:
            log_ratio = log_ratio.mean(dim=1)
        log_ratio = log_ratio.clamp(math.log(0.02), self.max_log_ratio)
        scale = self.prior_scale[index]
        std = scale * log_ratio.exp()
        noise = torch.randn(std.shape, device=visual.device, dtype=visual.dtype, generator=generator)
        residual = visual if reference is None else visual - reference[:, self.protected_tokens:].float()
        perturbation = std * noise
        if self.appearance_rank:
            # Keep both the projection and the rate in FP32 under autocast.
            with torch.autocast(device_type=visual.device.type, enabled=False):
                residual = residual.flatten(1) @ self.basis
                perturbation = (perturbation @ self.basis.T).reshape_as(visual)
        latent = torch.where(edit[:, None, None], visual + perturbation, visual)
        if self.protected_tokens:
            latent = torch.cat((mean.visual[:, :self.protected_tokens].float(), latent), dim=1)
        centered = (residual - self.prior_mean[index]) / scale
        logvar_ratio = 2 * log_ratio
        kl = 0.5 * (centered.square() + logvar_ratio.expm1() - logvar_ratio).flatten(1).sum(dim=1)
        spread = std.flatten(1).mean(dim=1)
        return FlowCondition(mean.instruction, latent), kl * edit, spread * edit


class SharedQueryBottleneck(nn.Module):
    """Fixed-width Gaussian queries for both image-input tasks."""

    def __init__(self, posterior_std: float):
        super().__init__()
        if not math.isfinite(posterior_std) or posterior_std <= 0:
            raise ValueError("posterior std must be finite and positive")
        self.posterior_std = posterior_std

    def forward(
        self, mean: FlowCondition, generator: torch.Generator, *, prompt: torch.Tensor,
        active: torch.Tensor | None = None,
    ) -> tuple[FlowCondition, torch.Tensor, torch.Tensor]:
        """Sample without labels. The training reference is used separately by rate()."""
        if mean.source is not None or active is not None:
            raise ValueError("shared queries require image information to pass through the connector")
        visual = mean.visual.float()
        noise = torch.randn(visual.shape, device=visual.device, generator=generator)
        source_present = prompt[:, 0] != OP_GENERATE
        visual = visual + self.posterior_std * noise * source_present[:, None, None]
        spread = self.posterior_std * source_present.float()
        return FlowCondition(mean.instruction, visual), torch.zeros_like(spread), spread

    def rate(
        self, visual: torch.Tensor, reference: torch.Tensor, prior_std: float, *,
        relaxed_basis: torch.Tensor | None = None, relaxed_std: float | None = None,
    ) -> torch.Tensor:
        """Per-image KL in nats per coordinate to a conditional Gaussian prior."""
        difference = (visual.float() - reference.float()).flatten(1)
        ratio = self.posterior_std / prior_std
        rate = 0.5 * (difference / prior_std).square().mean(dim=1) + 0.5 * (ratio ** 2 - 1) - math.log(ratio)
        if relaxed_basis is not None:
            if relaxed_std is None or not math.isfinite(relaxed_std) or relaxed_std <= 0:
                raise ValueError("a relaxed query subspace needs a positive finite prior std")
            # Replace the isotropic contribution on the orthonormal subspace.
            with torch.autocast(device_type=difference.device.type, enabled=False):
                projected = difference @ relaxed_basis
            relaxed_ratio = self.posterior_std / relaxed_std
            precision_change = 1 / relaxed_std ** 2 - 1 / prior_std ** 2
            variance_change = 0.5 * (relaxed_ratio ** 2 - ratio ** 2) + math.log(relaxed_std / prior_std)
            width = difference.shape[1]
            rate = rate + (0.5 * precision_change * projected.square().sum(dim=1)
                           + relaxed_basis.shape[1] * variance_change) / width
        return rate
