"""Encode the four discrete slots and the Stage-B source image."""
from __future__ import annotations

import math

import torch
from torch import nn


class InstructionEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        t = cfg["tokens"]
        dim = cfg["model"]["token_dim"]
        out = cfg["model"]["instruction_dim"]
        slots = (
            3,
            max(max(cfg["counts"]), t["count_keep"], t["count_mask"]) + 1,
            max(len(cfg["colors"]["names"]) - 1, t["color_mask"]) + 1,
            max(len(cfg["shapes"]["names"]) - 1, t["shape_mask"]) + 1,
        )
        self.slots = nn.ModuleList(nn.Embedding(size, dim) for size in slots)
        self.fusion = nn.Sequential(
            nn.Linear(4 * dim, out), nn.GELU(), nn.Linear(out, out), nn.LayerNorm(out)
        )

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([emb(prompt[:, i]) for i, emb in enumerate(self.slots)], dim=-1))


class SourceEncoder(nn.Module):
    """Spatial source encoder used to pretrain the visual input of the Flow."""

    def __init__(self, cfg: dict):
        super().__init__()
        model = cfg["model"]
        width = model["source_width"]
        self.num_tokens = model["visual_tokens"]
        self.dim = model["visual_dim"]
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.GroupNorm(math.gcd(8, width), width), nn.SiLU(),
            nn.Conv2d(width, 2 * width, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(2 * width, 2 * width, 4, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
        )
        self.project = nn.Linear(2 * width * 4 * 4, self.num_tokens * self.dim)
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        features = self.features(source.permute(0, 3, 1, 2))
        return self.norm(self.project(features).reshape(-1, self.num_tokens, self.dim))
