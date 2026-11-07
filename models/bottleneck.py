"""Variational bottleneck over all inputs to the visual decoder."""
from __future__ import annotations

import math

import torch
from torch import nn

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
        self, mean: FlowCondition, generator: torch.Generator,
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
