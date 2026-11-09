"""Structured tokens, RGB patches, and decoder projections shared by the baselines."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from data.schema import OP_NAMES

from .understanding import class_counts


class StructuredVocabulary(nn.Module):
    """Give each prompt slot its own ID range in a single decoder vocabulary."""

    def __init__(self, cfg: dict):
        super().__init__()
        counts, colors, shapes = class_counts(cfg)
        tokens = cfg["tokens"]
        count_offset = len(OP_NAMES)
        color_offset = count_offset + max(max(cfg["counts"]), tokens["count_keep"], tokens["count_mask"]) + 1
        shape_offset = color_offset + max(colors, tokens["color_mask"] + 1)
        specials = shape_offset + max(shapes, tokens["shape_mask"] + 1)
        self.image_start, self.image_end, self.answer_start = range(specials, specials + 3)
        self.size = specials + 3
        offsets = (count_offset + 1, color_offset, shape_offset)
        self.answer_slices = tuple(slice(start, start + n) for start, n in zip(offsets, (counts, colors, shapes)))
        self.register_buffer("prompt_offsets", torch.tensor([0, count_offset, color_offset, shape_offset]), persistent=False)
        self.register_buffer("answer_offsets", torch.tensor(offsets), persistent=False)

    def prompt_ids(self, prompt: torch.Tensor) -> torch.Tensor:
        return prompt + self.prompt_offsets

    def answer_ids(self, labels: torch.Tensor) -> torch.Tensor:
        return labels + self.answer_offsets

    def greedy_answers(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.stack([logits[:, i, ids].argmax(dim=-1) for i, ids in enumerate(self.answer_slices)], dim=1)


def marker(like: torch.Tensor, token_id: int) -> torch.Tensor:
    return like.new_full((len(like), 1), token_id)


@dataclass(frozen=True)
class ImagePatches:
    image_size: int
    patch_size: int

    def __post_init__(self):
        if self.patch_size < 1 or self.image_size < 1 or self.image_size % self.patch_size:
            raise ValueError("patch_size must be positive and divide image_size")

    @property
    def grid(self) -> int:
        return self.image_size // self.patch_size

    @property
    def count(self) -> int:
        return self.grid ** 2

    @property
    def dim(self) -> int:
        return 3 * self.patch_size ** 2

    def pack(self, images: torch.Tensor) -> torch.Tensor:
        """NHWC images to raster-order patch vectors."""
        p, g = self.patch_size, self.grid
        if images.ndim != 4 or images.shape[1:] != (self.image_size, self.image_size, 3):
            raise ValueError(f"expected a batch of {self.image_size}x{self.image_size} RGB images")
        return images.reshape(len(images), g, p, g, p, 3).permute(0, 1, 3, 2, 4, 5).reshape(len(images), self.count, self.dim)

    def unpack(self, patches: torch.Tensor) -> torch.Tensor:
        p, g = self.patch_size, self.grid
        return patches.reshape(len(patches), g, g, p, p, 3).permute(0, 1, 3, 2, 4, 5).reshape(
            len(patches), self.image_size, self.image_size, 3,
        )


class DecoderExpert(nn.Module):
    """Pre-norm attention projections and a SwiGLU feed-forward network."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        if width < 1 or heads < 1 or width % heads:
            raise ValueError("width must be positive and divisible by the number of attention heads")
        self.heads = heads
        self.head_dim = width // heads
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.attention_out = nn.Linear(width, width, bias=False)
        self.mlp_norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, 4 * width, bias=False)
        self.down = nn.Linear(2 * width, width, bias=False)

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.qkv(self.attention_norm(x)).reshape(len(x), x.shape[1], 3, self.heads, self.head_dim)
        return tuple(qkv.permute(2, 0, 3, 1, 4).unbind(0))

    def update(self, x: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        x = x + self.attention_out(attention.transpose(1, 2).reshape_as(x))
        gate, value = self.up(self.mlp_norm(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * value)
