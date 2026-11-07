"""Image-only representation plus instruction-conditioned classification."""
from __future__ import annotations

import torch
from torch import nn

from .conditioning import InstructionEncoder

ATTRIBUTES = ("count", "color", "shape")


def class_counts(cfg: dict) -> tuple[int, int, int]:
    return len(cfg["counts"]), len(cfg["colors"]["names"]), len(cfg["shapes"]["names"])


class UnderstandingMLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        model = cfg["model"]
        u = model["understanding"]
        self.feature_dim = u["feature_dim"]
        dims = [3 * cfg["image_size"] ** 2, *u["hidden"], self.feature_dim]
        layers = [nn.Flatten()]
        for dim_in, dim_out in zip(dims, dims[1:]):
            layers += [
                nn.Linear(dim_in, dim_out), nn.LayerNorm(dim_out), nn.GELU(), nn.Dropout(u["dropout"])
            ]
        self.backbone = nn.Sequential(*layers)
        self.instruction = InstructionEncoder(cfg)
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim + model["instruction_dim"], self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict({
            name: nn.Linear(self.feature_dim, n) for name, n in zip(ATTRIBUTES, class_counts(cfg))
        })

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)

    def forward(self, image: torch.Tensor, prompt: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode_image(image)
        fused = self.fusion(torch.cat((h, self.instruction(prompt)), dim=-1))
        return {name: head(fused) for name, head in self.heads.items()}
