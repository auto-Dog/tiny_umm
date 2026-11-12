"""Tiny Emu3 and BAGEL implementations with a learned spatial codec."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .bagel import TinyBagel as BagelTransformer
from .emu3 import TinyEmu3 as Emu3Decoder
from .sequence import DecoderExpert, StructuredVocabulary, marker
from .spatial_codec import SpatialCodec, reconstruction_errors

MODEL_NAMES = {"emu3": "Tiny Emu3", "bagel": "Tiny BAGEL"}


def unconditional_prompt(prompt: torch.Tensor, mask_tokens: torch.Tensor) -> torch.Tensor:
    return torch.cat((prompt[:, :1], mask_tokens[None].expand(len(prompt), -1)), dim=1)


def dropped_prompt(prompt: torch.Tensor, mask_tokens: torch.Tensor, probability: float) -> torch.Tensor:
    selected = torch.rand(len(prompt), device=prompt.device) < probability
    return torch.where(selected[:, None], unconditional_prompt(prompt, mask_tokens), prompt)


def foreground_weights(mask: torch.Tensor | None, grid: int, weight: float, like: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return like.new_ones((len(like), grid * grid))
    foreground = F.adaptive_max_pool2d(mask[:, None].float(), (grid, grid)).flatten(1)
    return 1 + weight * foreground


@dataclass
class LatentGrid:
    """The same raster API as ImagePatches, backed by an overlapping spatial codec."""

    codec: SpatialCodec
    normalized: bool = False

    @property
    def grid(self) -> int:
        return self.codec.grid

    @property
    def count(self) -> int:
        return self.grid ** 2

    @property
    def dim(self) -> int:
        return self.codec.latent_dim

    def pack(self, images: torch.Tensor) -> torch.Tensor:
        latents = self.codec.encode(images) if self.normalized else self.codec.encode_raw(images)
        return latents.flatten(1, 2)

    def unpack(self, vectors: torch.Tensor) -> torch.Tensor:
        latents = vectors.reshape(len(vectors), self.grid, self.grid, self.dim)
        return self.codec.decode(latents) if self.normalized else self.codec.decode_raw(latents)


class TinyEmu3(Emu3Decoder):
    """Tiny Emu3 with discrete visual tokens and one causal decoder."""

    display_name = MODEL_NAMES["emu3"]

    def __init__(self, cfg: dict, recipe: dict, codec: SpatialCodec):
        nn.Module.__init__(self)
        settings = recipe["model"]
        self.codec = codec.eval().requires_grad_(False)
        self.patches = LatentGrid(self.codec)
        self.vocab = StructuredVocabulary(cfg)
        width = settings["width"]
        self.embedding = nn.Embedding(self.vocab.size + len(codec.codebook.centers), width)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.position = nn.Parameter(torch.randn(2 * self.patches.count + 9, width) * 0.02)
        self.layers = nn.ModuleList(DecoderExpert(width, settings["heads"]) for _ in range(settings["layers"]))
        self.norm = nn.LayerNorm(width)
        self.foreground_weight = recipe["training"]["foreground_weight"]
        self.condition_dropout = recipe["training"]["condition_dropout"]
        self.register_buffer("mask_tokens", torch.tensor([cfg["tokens"][key] for key in ("count_mask", "color_mask", "shape_mask")]), persistent=False)

    @property
    def codebook(self):
        return self.codec.codebook

    def train(self, mode: bool = True):
        super().train(mode)
        self.codec.eval()
        return self

    def image_loss(self, prompt, target, source=None, *, mask=None, generator=None):
        if self.training:
            prompt = dropped_prompt(prompt, self.mask_tokens, self.condition_dropout)
        prefix = self._prefix(prompt, source, self.vocab.image_start)
        response = torch.cat((self.image_ids(target), marker(prompt, self.vocab.image_end)), dim=1)
        logits = self._response_logits(prefix, response)
        errors = F.cross_entropy(logits.flatten(0, 1), response.flatten(), reduction="none").reshape_as(response)
        weights = foreground_weights(mask, self.patches.grid, self.foreground_weight, errors)
        weights = torch.cat((weights, weights.new_ones((len(weights), 1))), dim=1)
        return ((errors * weights).sum(dim=1) / weights.sum(dim=1)).mean()

    @torch.no_grad()
    def sample(self, prompt, source=None, *, temperature=0.85, top_k=64, guidance=1.5, generator=None):
        if not math.isfinite(temperature) or temperature < 0 or not math.isfinite(guidance) or guidance < 0:
            raise ValueError("temperature and guidance must be finite and nonnegative")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")
        ids = self._prefix(prompt, source, self.vocab.image_start)
        null_ids = self._prefix(unconditional_prompt(prompt, self.mask_tokens), source, self.vocab.image_start)
        cache, null_cache, codes = [], [], []
        for _ in range(self.patches.count):
            logits = self._logits(self._hidden(ids, cache)[:, -1]).float()
            if guidance != 1:
                null_logits = self._logits(self._hidden(null_ids, null_cache)[:, -1]).float()
                logits = null_logits + guidance * (logits - null_logits)
            logits = logits[:, self.vocab.size:]
            if temperature == 0:
                code = logits.argmax(dim=1, keepdim=True)
            else:
                k = logits.shape[1] if top_k is None else min(top_k, logits.shape[1])
                values, candidates = logits.topk(k, dim=1)
                probs = ((values - values[:, :1]) / temperature).softmax(dim=1)
                code = candidates.gather(1, torch.multinomial(probs, 1, generator=generator))
            codes.append(code)
            ids = null_ids = code + self.vocab.size
        return self.codec.decode_ids(torch.cat(codes, dim=1))


class TinyBagel(BagelTransformer):
    """Tiny BAGEL with modality experts, text AR, and latent rectified flow."""

    display_name = MODEL_NAMES["bagel"]

    def __init__(self, cfg: dict, recipe: dict, codec: SpatialCodec):
        super().__init__(cfg, recipe["model"])
        self.codec = codec.eval().requires_grad_(False)
        self.patches = LatentGrid(self.codec, normalized=True)
        dim, width = codec.latent_dim, recipe["model"]["width"]
        self.source_projection = nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.Linear(width, width))
        self.noisy_projection = nn.Conv2d(dim, width, 3, padding=1)
        self.velocity_head = nn.Linear(width, dim)
        self.foreground_weight = recipe["training"]["foreground_weight"]
        self.condition_dropout = recipe["training"]["condition_dropout"]
        self.decoded_loss_weight = recipe["training"].get("decoded_loss_weight", 0.25)
        self.register_buffer("mask_tokens", torch.tensor([cfg["tokens"][key] for key in ("count_mask", "color_mask", "shape_mask")]), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.codec.eval()
        return self

    def _velocity(self, x_t, t, prefix, mask):
        phase = t[:, None] * self.time_frequencies
        time = self.time_projection(torch.cat((phase.sin(), phase.cos()), dim=-1))
        noisy = self.noisy_projection(x_t.permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)
        noisy = noisy + self._patch_position() + time[:, None]
        start = prefix.shape[1]
        hidden = self._hidden(torch.cat((prefix, noisy), dim=1), start, mask)
        return self.velocity_head(self.image_norm(hidden[:, start:])).reshape_as(x_t)

    def image_loss(self, prompt, target, source=None, *, mask=None, generator=None):
        if self.training:
            prompt = dropped_prompt(prompt, self.mask_tokens, self.condition_dropout)
        with torch.no_grad():
            clean = self.codec.encode(target)
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        t = torch.rand(len(clean), device=clean.device, generator=generator)
        x_t = torch.lerp(noise, clean, t[:, None, None, None])
        predicted = self(prompt, x_t, t, source)
        errors = (predicted.float() - (clean - noise)).square().mean(dim=-1).flatten(1)
        weights = foreground_weights(mask, self.patches.grid, self.foreground_weight, errors)
        loss = ((errors * weights).sum(dim=1) / weights.sum(dim=1)).mean()
        if mask is not None and self.decoded_loss_weight > 0:
            estimated = x_t + (1 - t[:, None, None, None]) * predicted
            decoded = self.codec.decode(estimated)
            loss = loss + self.decoded_loss_weight * (t.square() * reconstruction_errors(decoded, target, mask)).mean()
        return loss

    @torch.no_grad()
    def sample(self, prompt, source=None, *, steps=40, solver="midpoint", guidance=2.0, generator=None):
        if steps < 1 or solver not in ("euler", "midpoint"):
            raise ValueError("sampling needs positive steps and an euler or midpoint solver")
        if not math.isfinite(guidance) or guidance < 0:
            raise ValueError("guidance must be finite and nonnegative")
        prefix = self._prefix(prompt, source, self.vocab.image_start)
        null = self._prefix(unconditional_prompt(prompt, self.mask_tokens), source, self.vocab.image_start)
        mask = self._mask(prefix.shape[1], self.patches.count, source is not None)
        shape = (len(prompt), self.patches.grid, self.patches.grid, self.patches.dim)
        x = torch.randn(shape, device=prompt.device, generator=generator)

        def velocity(state, time):
            conditional = self._velocity(state, time, prefix, mask)
            if guidance == 1:
                return conditional
            unconditioned = self._velocity(state, time, null, mask)
            return unconditioned + guidance * (conditional - unconditioned)

        dt = 1 / steps
        for i in range(steps):
            t = x.new_full((len(prompt),), i * dt)
            v = velocity(x, t)
            if solver == "midpoint":
                v = velocity(x + 0.5 * dt * v, t + 0.5 * dt)
            x = x + dt * v
        return self.codec.decode(x).clamp(-1, 1)
