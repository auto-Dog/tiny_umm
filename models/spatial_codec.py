"""Trainable spatial image codec shared by Tiny Emu3 and BAGEL."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .emu3 import PatchCodebook


class CodecResidual(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class SpatialCodebook(PatchCodebook):
    """EMA vector quantizer. Initialization uses encoder features from training data."""

    def __init__(self, size: int, dim: int):
        super().__init__(size, dim)
        self.register_buffer("counts", torch.ones(size))
        self.register_buffer("sums", torch.zeros(size, dim))

    @torch.no_grad()
    def initialize(self, vectors: torch.Tensor, generator: torch.Generator) -> None:
        self.fit(vectors.float(), 15, generator)
        self.counts.fill_(1)
        self.sums.copy_(self.centers)

    def forward(self, vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.encode(vectors.detach())
        quantized = self.decode(ids).to(vectors.dtype)
        if self.training:
            with torch.no_grad():
                flat, flat_ids = vectors.detach().float().flatten(0, 1), ids.flatten()
                counts = torch.bincount(flat_ids, minlength=len(self.centers)).float()
                sums = torch.zeros_like(self.centers).index_add_(0, flat_ids, flat)
                self.counts.lerp_(counts, 0.01)
                self.sums.lerp_(sums, 0.01)
                self.centers.copy_(self.sums / self.counts[:, None].clamp_min(1e-5))
                dead = self.counts < 0.1
                replacements = flat[torch.randint(len(flat), (len(self.centers),), device=flat.device)]
                self.centers[dead] = replacements[dead]
                self.counts[dead] = 1
                self.sums[dead] = replacements[dead]
        commitment = F.mse_loss(vectors.float(), quantized.detach().float())
        return vectors + (quantized - vectors).detach(), commitment


class SpatialCodec(nn.Module):
    def __init__(self, image_size: int, width: int = 32, latent_dim: int = 16, codebook_size: int = 1024):
        super().__init__()
        if image_size % 4 or width < 8 or width % 8:
            raise ValueError("codec needs image_size divisible by four and width divisible by eight")
        self.image_size = image_size
        self.grid = image_size // 4
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), CodecResidual(width),
            nn.Conv2d(width, 2 * width, 4, stride=2, padding=1), CodecResidual(2 * width),
            nn.Conv2d(2 * width, 2 * width, 4, stride=2, padding=1), CodecResidual(2 * width),
            nn.Conv2d(2 * width, latent_dim, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 2 * width, 3, padding=1), CodecResidual(2 * width),
            nn.ConvTranspose2d(2 * width, 2 * width, 4, stride=2, padding=1), CodecResidual(2 * width),
            nn.ConvTranspose2d(2 * width, width, 4, stride=2, padding=1), CodecResidual(width),
            nn.Conv2d(width, 3, 3, padding=1), nn.Tanh(),
        )
        self.codebook = SpatialCodebook(codebook_size, latent_dim)
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))

    def encode_raw(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    def decode_raw(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return (self.encode_raw(images) - self.latent_mean) / self.latent_std

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decode_raw(latents * self.latent_std + self.latent_mean)

    def image_ids(self, images: torch.Tensor) -> torch.Tensor:
        return self.codebook.encode(self.encode_raw(images).flatten(1, 2))

    def decode_ids(self, ids: torch.Tensor) -> torch.Tensor:
        z = self.codebook.decode(ids).reshape(len(ids), self.grid, self.grid, self.latent_dim)
        return self.decode_raw(z)

    def forward(self, images: torch.Tensor, quantized: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode_raw(images)
        continuous = self.decode_raw(z)
        commitment = z.float().square().mean() * 0.0001
        if quantized:
            vectors, error = self.codebook(z.flatten(1, 2))
            reconstruction = self.decode_raw(vectors.reshape_as(z))
            commitment = commitment + 0.25 * error
        else:
            reconstruction = continuous
        return reconstruction, continuous, commitment


def reconstruction_errors(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-image foreground-weighted reconstruction and edge errors."""
    weights = 1 + 3 * mask.float()
    pixels = (prediction.float() - target.float()).square().mean(dim=-1)
    loss = (pixels * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2))
    for dim in (1, 2):
        edges = (prediction.float().diff(dim=dim) - target.float().diff(dim=dim)).abs()
        loss = loss + 0.25 * edges.mean(dim=(1, 2, 3))
    return loss
