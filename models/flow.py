"""Small conditional pixel U-Net and the linear flow-matching path."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FlowCondition:
    instruction: torch.Tensor       # [B, instruction_dim]
    visual: torch.Tensor            # [B, K, visual_dim]
    source: torch.Tensor | None = None
    direct: torch.Tensor | None = None  # [B], mask for the optional source pixels


class _ResBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, condition_dim: int):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(math.gcd(8, dim_in), dim_in), nn.SiLU(), nn.Conv2d(dim_in, dim_out, 3, padding=1)
        )
        self.out_norm = nn.GroupNorm(math.gcd(8, dim_out), dim_out)
        self.out_conv = nn.Conv2d(dim_out, dim_out, 3, padding=1)
        self.modulation = nn.Linear(condition_dim, 2 * dim_out)
        self.skip = nn.Identity() if dim_in == dim_out else nn.Conv2d(dim_in, dim_out, 1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        y = self.out_norm(self.in_layers(x))
        scale, bias = self.modulation(condition)[:, :, None, None].chunk(2, dim=1)
        return self.skip(x) + self.out_conv(F.silu(y * (1 + scale) + bias))


class PixelFlow(nn.Module):
    """Velocity predictor; image tensors use [B, H, W, 3] throughout the public API."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.image_size = cfg["image_size"]
        model = cfg["model"]
        flow = model["flow"]
        w, time_dim, cond_dim = flow["width"], flow["time_dim"], flow["condition_dim"]
        if self.image_size % 4 or time_dim < 2 or time_dim % 2:
            raise ValueError("Flow requires an image size divisible by 4 and an even time_dim")
        self.register_buffer("time_freqs", 2 * math.pi * torch.logspace(0, 1.7, time_dim // 2))
        self.time = nn.Sequential(nn.Linear(time_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.instruction = nn.Linear(model["instruction_dim"], cond_dim)
        self.null_visual = nn.Parameter(torch.zeros(model["visual_tokens"], model["visual_dim"]))
        # x_t, optional RGB source hint, and a flag indicating the hint is present.
        self.input = nn.Conv2d(7, w, 3, padding=1)
        self.high = _ResBlock(w, w, cond_dim)
        self.down_high = nn.Conv2d(w, 2 * w, 4, stride=2, padding=1)
        self.medium = _ResBlock(2 * w, 2 * w, cond_dim)
        self.down_medium = nn.Conv2d(2 * w, 3 * w, 4, stride=2, padding=1)
        self.middle = _ResBlock(3 * w, 3 * w, cond_dim)
        self.query_norm = nn.LayerNorm(3 * w)
        self.visual_attention = nn.MultiheadAttention(
            3 * w, flow["attention_heads"], kdim=model["visual_dim"], vdim=model["visual_dim"],
            batch_first=True,
        )
        self.up_medium = _ResBlock(5 * w, 2 * w, cond_dim)
        self.up_high = _ResBlock(3 * w, w, cond_dim)
        self.output = nn.Sequential(nn.GroupNorm(math.gcd(8, w), w), nn.SiLU(), nn.Conv2d(w, 3, 3, padding=1))

    def adapters(self) -> list[nn.Module]:
        """The optional Stage-D trainable conditioning layers."""
        return [self.instruction, self.query_norm, self.visual_attention, *(
            block.modulation for block in (self.high, self.medium, self.middle, self.up_medium, self.up_high)
        )]

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: FlowCondition) -> torch.Tensor:
        phase = t[:, None] * self.time_freqs
        c = self.time(torch.cat((phase.sin(), phase.cos()), dim=-1)) + self.instruction(condition.instruction)
        c = F.silu(c)
        if condition.source is None:
            source = torch.zeros_like(x_t)
            flag = torch.zeros_like(x_t[..., :1])
        else:
            active = condition.direct[:, None, None, None]
            source = torch.where(active, condition.source, 0.0)
            flag = active.to(x_t.dtype).expand_as(x_t[..., :1])
        x = self.input(torch.cat((x_t, source, flag), dim=-1).permute(0, 3, 1, 2))
        high = self.high(x, c)
        medium = self.medium(self.down_high(high), c)
        low = self.middle(self.down_medium(medium), c)
        q = self.query_norm(low.flatten(2).transpose(1, 2))
        attended, _ = self.visual_attention(q, condition.visual, condition.visual, need_weights=False)
        low = low + attended.transpose(1, 2).reshape_as(low)
        y = self.up_medium(torch.cat((F.interpolate(low, scale_factor=2), medium), dim=1), c)
        y = self.up_high(torch.cat((F.interpolate(y, scale_factor=2), high), dim=1), c)
        return self.output(y).permute(0, 2, 3, 1)


def linear_path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x(t) = (1 - t) x0 + t x1; its velocity is x1 - x0."""
    return torch.lerp(x0, x1, t.reshape(-1, 1, 1, 1)), x1 - x0


def flow_matching_errors(
    flow: PixelFlow, target: torch.Tensor, condition: FlowCondition,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-image FM MSE. The forward pass stays differentiable for Stage C."""
    noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
    t = torch.rand(len(target), device=target.device, dtype=target.dtype, generator=generator)
    x_t, velocity = linear_path(noise, target, t)
    return (flow(x_t, t, condition).float() - velocity).square().mean(dim=(1, 2, 3))


@torch.no_grad()
def sample_flow(
    flow: PixelFlow, condition: FlowCondition, *, steps: int, solver: str = "midpoint",
    generator: torch.Generator | None = None, noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if steps < 1:
        raise ValueError("sampling steps must be positive")
    if solver not in ("euler", "midpoint"):
        raise ValueError(f"unknown ODE solver: {solver}")
    batch = len(condition.instruction)
    size = flow.image_size
    x = torch.randn((batch, size, size, 3), device=condition.instruction.device, generator=generator) if noise is None else noise
    dt = 1.0 / steps
    for i in range(steps):
        t = x.new_full((batch,), i * dt)
        v = flow(x, t, condition)
        if solver == "midpoint":
            v = flow(x + (0.5 * dt) * v, t + 0.5 * dt, condition)
        x = x + dt * v
    return x.clamp(-1.0, 1.0)
