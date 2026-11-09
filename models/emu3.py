"""Emu3-style next-token prediction for discrete instructions and images."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .sequence import DecoderExpert, ImagePatches, StructuredVocabulary, marker


class PatchCodebook(nn.Module):
    """A frozen RGB patch tokenizer fitted by Lloyd iterations on training images."""

    def __init__(self, size: int, dim: int):
        super().__init__()
        if size < 2:
            raise ValueError("codebook_size must be at least two")
        self.register_buffer("centers", torch.zeros(size, dim))

    @staticmethod
    def _nearest(patches: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=patches.device.type, enabled=False):
            scores = 2 * (patches.float() @ centers.T) - centers.square().sum(dim=1)
        return scores.argmax(dim=-1)

    @torch.no_grad()
    def fit(self, patches: torch.Tensor, steps: int, generator: torch.Generator) -> None:
        if not len(patches) or steps < 1:
            raise ValueError("fitting the patch codebook requires samples and positive kmeans_steps")
        size = len(self.centers)
        order = torch.randperm(len(patches), device=patches.device, generator=generator)
        centers = patches[order[torch.arange(size, device=patches.device) % len(patches)]]
        for _ in range(steps):
            ids = torch.cat([self._nearest(part, centers) for part in patches.split(8192)])
            sums = torch.zeros_like(centers).index_add_(0, ids, patches)
            counts = torch.bincount(ids, minlength=size)[:, None]
            replacement = torch.randint(len(patches), (size,), device=patches.device, generator=generator)
            centers = torch.where(counts > 0, sums / counts.clamp_min(1), patches[replacement])
        self.centers.copy_(centers)

    @torch.no_grad()
    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        return self._nearest(patches, self.centers)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.centers[tokens]


class TinyEmu3(nn.Module):
    """One decoder and vocabulary for answers, image generation, and image edits."""

    image_objective = "ce"

    def __init__(self, cfg: dict, settings: dict):
        super().__init__()
        self.vocab = StructuredVocabulary(cfg)
        self.patches = ImagePatches(cfg["image_size"], settings["patch_size"])
        width, depth, heads = settings["width"], settings["layers"], settings["heads"]
        if depth < 1:
            raise ValueError("layers must be positive")
        codes = settings["codebook_size"]
        self.codebook = PatchCodebook(codes, self.patches.dim)
        self.embedding = nn.Embedding(self.vocab.size + codes, width)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.position = nn.Parameter(torch.randn(2 * self.patches.count + 9, width) * 0.02)
        self.layers = nn.ModuleList(DecoderExpert(width, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(width)

    def _hidden(self, ids: torch.Tensor, cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None) -> torch.Tensor:
        offset = cache[0][0].shape[-2] if cache else 0
        x = self.embedding(ids) + self.position[offset:offset + ids.shape[1]]
        for i, layer in enumerate(self.layers):
            q, k, v = layer.project(x)
            if offset:
                old_k, old_v = cache[i]
                k = torch.cat((old_k, k), dim=-2)
                v = torch.cat((old_v, v), dim=-2)
                cache[i] = (k, v)
            elif cache is not None:
                cache.append((k, v))
            # After prefilling, each call appends exactly one token to the cache.
            x = layer.update(x, F.scaled_dot_product_attention(q, k, v, is_causal=(offset == 0)))
        return x

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(self.norm(hidden), self.embedding.weight)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self._logits(self._hidden(ids))

    def image_ids(self, image: torch.Tensor) -> torch.Tensor:
        return self.vocab.size + self.codebook.encode(self.patches.pack(image))

    def _prefix(self, prompt: torch.Tensor, source: torch.Tensor | None, start: int) -> torch.Tensor:
        ids = [self.vocab.prompt_ids(prompt)]
        if source is not None:
            ids.extend((marker(prompt, self.vocab.image_start), self.image_ids(source), marker(prompt, self.vocab.image_end)))
        return torch.cat((*ids, marker(prompt, start)), dim=1)

    def _response_logits(self, prefix: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        # The final prefix token predicts the first response token.
        hidden = self._hidden(torch.cat((prefix, response[:, :-1]), dim=1))
        return self._logits(hidden[:, prefix.shape[1] - 1:])

    def answer_logits(self, prompt: torch.Tensor, source: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        prefix = self._prefix(prompt, source, self.vocab.answer_start)
        return self._response_logits(prefix, self.vocab.answer_ids(labels))

    def image_loss(self, prompt: torch.Tensor, target: torch.Tensor, source: torch.Tensor | None = None) -> torch.Tensor:
        prefix = self._prefix(prompt, source, self.vocab.image_start)
        response = torch.cat((self.image_ids(target), marker(prompt, self.vocab.image_end)), dim=1)
        logits = self._response_logits(prefix, response)
        return F.cross_entropy(logits.flatten(0, 1), response.flatten())

    @torch.no_grad()
    def predict(self, prompt: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        ids = self._prefix(prompt, source, self.vocab.answer_start)
        cache = []
        labels = []
        for allowed in self.vocab.answer_slices:
            logits = self._logits(self._hidden(ids, cache)[:, -1])
            label = logits[:, allowed].argmax(dim=1, keepdim=True)
            labels.append(label)
            ids = label + allowed.start
        return torch.cat(labels, dim=1)

    @torch.no_grad()
    def sample(
        self, prompt: torch.Tensor, source: torch.Tensor | None = None, *,
        temperature: float = 1.0, top_k: int | None = None, generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if not math.isfinite(temperature) or temperature < 0 or (top_k is not None and top_k < 1):
            raise ValueError("use a nonnegative temperature and a positive top_k")
        ids = self._prefix(prompt, source, self.vocab.image_start)
        cache = []
        codes = []
        # The fixed raster determines the valid output length and image boundary.
        for _ in range(self.patches.count):
            logits = self._logits(self._hidden(ids, cache)[:, -1])[:, self.vocab.size:].float()
            if temperature == 0:
                code = logits.argmax(dim=1, keepdim=True)
            elif top_k is None:
                probs = ((logits - logits.amax(dim=1, keepdim=True)) / temperature).softmax(dim=1)
                code = torch.multinomial(probs, 1, generator=generator)
            else:
                values, indices = logits.topk(min(top_k, logits.shape[1]), dim=1)
                probs = ((values - values[:, :1]) / temperature).softmax(dim=1)
                code = indices.gather(1, torch.multinomial(probs, 1, generator=generator))
            codes.append(code)
            ids = code + self.vocab.size
        return self.patches.unpack(self.codebook.decode(torch.cat(codes, dim=1)))
