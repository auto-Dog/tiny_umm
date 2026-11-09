"""BAGEL-style text decoding and rectified flow in a shared Transformer."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .flow import linear_path
from .sequence import DecoderExpert, ImagePatches, StructuredVocabulary, marker


class ModalityBlock(nn.Module):
    """Hard route context and noisy image tokens through separate projections and MLPs."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.context = DecoderExpert(width, heads)
        self.generation = DecoderExpert(width, heads)

    def forward(self, x: torch.Tensor, context_length: int, mask: torch.Tensor) -> torch.Tensor:
        if context_length == x.shape[1]:
            q, k, v = self.context.project(x)
            return self.context.update(x, F.scaled_dot_product_attention(q, k, v, attn_mask=mask))

        context, noisy = x[:, :context_length], x[:, context_length:]
        q, k, v = (torch.cat(pair, dim=-2) for pair in zip(self.context.project(context), self.generation.project(noisy)))
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        context = self.context.update(context, attention[:, :, :context_length])
        noisy = self.generation.update(noisy, attention[:, :, context_length:])
        return torch.cat((context, noisy), dim=1)


class TinyBagel(nn.Module):
    """Use continuous RGB patches as the image representation for the flow expert."""

    image_objective = "fm"

    def __init__(self, cfg: dict, settings: dict):
        super().__init__()
        self.vocab = StructuredVocabulary(cfg)
        self.patches = ImagePatches(cfg["image_size"], settings["patch_size"])
        width, depth, heads = settings["width"], settings["layers"], settings["heads"]
        if depth < 1:
            raise ValueError("layers must be positive")
        self.embedding = nn.Embedding(self.vocab.size, width)
        nn.init.normal_(self.embedding.weight, std=0.02)
        length = 2 * self.patches.count + 9
        self.position = nn.Parameter(torch.randn(length, width) * 0.02)
        self.row_position = nn.Parameter(torch.randn(self.patches.grid, width) * 0.02)
        self.col_position = nn.Parameter(torch.randn(self.patches.grid, width) * 0.02)
        self.source_projection = nn.Sequential(nn.Linear(self.patches.dim, width), nn.GELU(), nn.Linear(width, width))
        self.noisy_projection = nn.Linear(self.patches.dim, width)
        self.time_projection = nn.Sequential(nn.Linear(32, width), nn.SiLU(), nn.Linear(width, width))
        self.layers = nn.ModuleList(ModalityBlock(width, heads) for _ in range(depth))
        self.text_norm = nn.LayerNorm(width)
        self.image_norm = nn.LayerNorm(width)
        self.velocity_head = nn.Linear(width, self.patches.dim)
        self.register_buffer("time_frequencies", 2 * math.pi * torch.logspace(0, 1.7, 16), persistent=False)
        self.register_buffer("causal", torch.ones(length, length, dtype=torch.bool).tril(), persistent=False)

    def _patch_position(self) -> torch.Tensor:
        return (self.row_position[:, None] + self.col_position).flatten(0, 1)

    def _prefix(self, prompt: torch.Tensor, source: torch.Tensor | None, start: int) -> torch.Tensor:
        embeddings = [self.embedding(self.vocab.prompt_ids(prompt))]
        if source is not None:
            source_tokens = self.source_projection(self.patches.pack(source)) + self._patch_position()
            embeddings.extend((
                self.embedding(marker(prompt, self.vocab.image_start)),
                source_tokens,
                self.embedding(marker(prompt, self.vocab.image_end)),
            ))
        return torch.cat((*embeddings, self.embedding(marker(prompt, start))), dim=1)

    def _mask(self, context_length: int, noisy_length: int, has_source: bool) -> torch.Tensor:
        length = context_length + noisy_length
        mask = self.causal[:length, :length].clone()
        if has_source:
            start = len(self.vocab.prompt_offsets) + 1
            source = slice(start, start + self.patches.count)
            mask[source, source] = True
        mask[context_length:, context_length:] = True
        return mask

    def _hidden(self, x: torch.Tensor, context_length: int, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.position[:x.shape[1]]
        for layer in self.layers:
            x = layer(x, context_length, mask)
        return x

    def _text_logits(self, x: torch.Tensor, response_start: int) -> torch.Tensor:
        length = x.shape[1]
        hidden = self._hidden(x, length, self._mask(length, 0, True))
        return F.linear(self.text_norm(hidden[:, response_start:]), self.embedding.weight)

    def answer_logits(self, prompt: torch.Tensor, source: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        prefix = self._prefix(prompt, source, self.vocab.answer_start)
        previous = self.embedding(self.vocab.answer_ids(labels)[:, :-1])
        return self._text_logits(torch.cat((prefix, previous), dim=1), prefix.shape[1] - 1)

    @torch.no_grad()
    def predict(self, prompt: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        context = self._prefix(prompt, source, self.vocab.answer_start)
        labels = []
        for allowed in self.vocab.answer_slices:
            logits = self._text_logits(context, context.shape[1] - 1)[:, 0, allowed]
            label = logits.argmax(dim=-1, keepdim=True)
            labels.append(label)
            context = torch.cat((context, self.embedding(label + allowed.start)), dim=1)
        return torch.cat(labels, dim=1)

    def _velocity(self, x_t: torch.Tensor, t: torch.Tensor, prefix: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        phase = t[:, None] * self.time_frequencies
        time = self.time_projection(torch.cat((phase.sin(), phase.cos()), dim=-1))
        noisy = self.noisy_projection(self.patches.pack(x_t)) + self._patch_position() + time[:, None]
        start = prefix.shape[1]
        hidden = self._hidden(torch.cat((prefix, noisy), dim=1), start, mask)
        return self.patches.unpack(self.velocity_head(self.image_norm(hidden[:, start:])))

    def forward(
        self, prompt: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prefix = self._prefix(prompt, source, self.vocab.image_start)
        mask = self._mask(prefix.shape[1], self.patches.count, source is not None)
        return self._velocity(x_t, t, prefix, mask)

    def image_loss(
        self, prompt: torch.Tensor, target: torch.Tensor, source: torch.Tensor | None = None, *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
        t = torch.rand(len(target), device=target.device, dtype=target.dtype, generator=generator)
        x_t, velocity = linear_path(noise, target, t)
        return F.mse_loss(self(prompt, x_t, t, source).float(), velocity)

    @torch.no_grad()
    def sample(
        self, prompt: torch.Tensor, source: torch.Tensor | None = None, *,
        steps: int = 40, solver: str = "midpoint", generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if steps < 1 or solver not in ("euler", "midpoint"):
            raise ValueError("sampling needs positive steps and an euler or midpoint solver")
        prefix = self._prefix(prompt, source, self.vocab.image_start)
        mask = self._mask(prefix.shape[1], self.patches.count, source is not None)
        size = self.patches.image_size
        x = torch.randn((len(prompt), size, size, 3), device=prompt.device, generator=generator)
        dt = 1.0 / steps
        for i in range(steps):
            t = x.new_full((len(prompt),), i * dt)
            v = self._velocity(x, t, prefix, mask)
            if solver == "midpoint":
                v = self._velocity(x + 0.5 * dt * v, t + 0.5 * dt, prefix, mask)
            x = x + dt * v
        return x.clamp(-1, 1)
