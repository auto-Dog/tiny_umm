"""Map frozen understanding features to the shared K × D visual interface."""
from __future__ import annotations

import torch
from torch import nn

from .understanding import ATTRIBUTES, class_counts


class QueryConnector(nn.Module):
    def __init__(self, cfg: dict, feature_dim: int | None = None, width: int | None = None):
        super().__init__()
        model = cfg["model"]
        k, d = model["visual_tokens"], model["visual_dim"]
        if feature_dim is None:
            feature_dim = model["understanding"]["feature_dim"]
        if width is None:
            width = model["connector"]["width"]
        self.queries = nn.Parameter(torch.empty(k, d))
        nn.init.normal_(self.queries, std=0.02)
        self.project = nn.Sequential(
            nn.Linear(d + feature_dim + model["instruction_dim"], width),
            nn.LayerNorm(width), nn.GELU(),
            nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(),
            nn.Linear(width, d), nn.LayerNorm(d),
        )

    def forward(self, features: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        queries = self.queries[None].expand(len(features), -1, -1)
        inputs = torch.cat((features, instruction), dim=-1)
        return queries + self.project(torch.cat((queries, inputs[:, None].expand(-1, queries.shape[1], -1)), dim=-1))


class DiscreteBridge(QueryConnector):
    """Same interface using only the three classifiers' probability vectors."""

    def __init__(self, cfg: dict):
        super().__init__(cfg, sum(class_counts(cfg)), cfg["model"]["connector"]["discrete_width"])

    def forward(self, logits: dict[str, torch.Tensor], instruction: torch.Tensor) -> torch.Tensor:
        probabilities = torch.cat([logits[key].softmax(dim=-1) for key in ATTRIBUTES], dim=-1)
        return super().forward(probabilities, instruction)
