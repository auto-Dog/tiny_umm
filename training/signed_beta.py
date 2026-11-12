"""Signed information-theoretic rate weights beta(A, C) predicted by a bounded MLP.

Objective (per context c = task type x instruction subcategory, Z the four queries):

    min  D  +  E_c[ beta(A, c) * I(A; Z | C = c) ]       beta may be negative.

D is the task NLL (Flow Matching + 0.08 understanding CE), which bounds -I(Y; Z | C) up to H(Y | C).
beta(a, c) = b(c) * m(a, c), with b(c) = beta_max * tanh(u(c)) carrying the sign and a positive modulation
m whose mean within each context of the batch is one. The rate weighted by m equals b(c) times the rate
under the tilted data distribution p~(a | c) proportional to m p(a | c), so the A dependence reweights
examples without changing the sign inside a context.

Variational bounds, chosen per sign so that the surrogate always upper-bounds the signed objective:
  b(c) > 0: KL(q(z | a, c) || r(z | y, c)) per coordinate, the conditional residual upper bound already
            used by the ib objective (r is the frozen target-conditioned prior).
  b(c) < 0: InfoNCE over same-context examples in the batch, I_NCE <= I(A; Z | C). Its critic is a fixed
            temperature Gaussian, f(z_i, a_j) = -|z_i - mu_j|^2 / (2 tau_c^2). The bound is translation
            invariant, so it cannot be raised by a drift shared by all queries, and it saturates at log K.

beta receives no gradient from the training loss. The MLP follows a one-step lookahead hypergradient:
after theta' = theta - eta grad(D + sum_i beta_i S_i), dL_meta/dbeta_i ~= -eta <grad L_meta, grad S_i>,
where L_meta is the task loss of an independent training batch and S_i the active branch surrogate.
The inner product is the directional derivative of S_i along grad L_meta over the query-producing
parameters, computed by a central finite difference in FP32. Each context contributes its own mean
alignment; a quadratic anchor on the context logits and momentum SGD keep the step proportional to the signal.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

CONTEXTS = ("edit/color", "edit/shape", "edit/both", "understand")
TEXT_CONTEXT = 3


def context_codes(context: torch.Tensor) -> torch.Tensor:
    """Task one-hot followed by instruction subcategory one-hot."""
    task = F.one_hot((context == TEXT_CONTEXT).long(), 2)
    sub = F.one_hot(context.clamp_min(0), len(CONTEXTS))
    return torch.cat((task, sub), dim=1).float()


def grouped_nce(z: torch.Tensor, mu: torch.Tensor, context: torch.Tensor, tau_sq: torch.Tensor):
    """Per-example InfoNCE in nats with same-context negatives; rows with context < 0 are ignored."""
    z, mu = z.flatten(1).float(), mu.flatten(1).float()
    with torch.autocast(device_type=z.device.type, enabled=False):
        distance = torch.cdist(z, mu).square()
    same = (context[:, None] == context[None]) & (context[:, None] >= 0)
    scale = tau_sq[context.clamp_min(0)]
    logits = (-distance / (2 * scale[:, None])).masked_fill(~same, float("-inf"))
    k = same.sum(dim=1)
    valid = k >= 2
    nce = logits.diagonal() - torch.logsumexp(logits, dim=1) + k.clamp_min(1).float().log()
    return torch.where(valid, nce, torch.zeros_like(nce)), valid


def group_center(values: torch.Tensor, context: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    groups = len(CONTEXTS)
    index = context.clamp_min(0)
    weight = active.float()
    sums = values.new_zeros(groups).index_add(0, index, values * weight)
    counts = values.new_zeros(groups).index_add(0, index, weight)
    return values - (sums / counts.clamp_min(1))[index]


class SignedBeta(nn.Module):
    """b(c) from a context MLP, positive m(a, c) from an input MLP, both bounded."""

    def __init__(self, cfg: dict, beta_max: float, initial: tuple[float, ...], span: float):
        super().__init__()
        if not 0 < beta_max or not 0 <= span < 0.5 or len(initial) != len(CONTEXTS):
            raise ValueError("signed beta needs beta_max > 0, span in [0, 0.5) and one initial value per context")
        if any(abs(value) >= beta_max for value in initial):
            raise ValueError("initial beta values must lie strictly inside (-beta_max, beta_max)")
        code = 2 + len(CONTEXTS)
        width = 16 * (cfg["model"]["flow"]["width"] + 3) + cfg["model"]["instruction_dim"] + code
        self.beta_max, self.span = beta_max, span
        self.sign = nn.Sequential(nn.Linear(code, 32), nn.GELU(), nn.Linear(32, 1))
        self.modulation = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))
        for head in (self.sign[-1], self.modulation[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.register_buffer("offset", torch.tensor([math.atanh(v / beta_max) for v in initial]))

    def context_logits(self, context: torch.Tensor) -> torch.Tensor:
        return self.sign(context_codes(context)).squeeze(1).float() + self.offset[context.clamp_min(0)]

    def context_beta(self, context: torch.Tensor) -> torch.Tensor:
        return self.beta_max * torch.tanh(self.context_logits(context))

    def anchor(self, device: torch.device) -> torch.Tensor:
        """Squared distance of every context logit from its initial value."""
        return (self.context_logits(torch.arange(len(CONTEXTS), device=device)) - self.offset).square().sum()

    def hyper_loss(self, unit: torch.Tensor, beta: torch.Tensor, context: torch.Tensor, active: torch.Tensor,
                   anchor: float) -> torch.Tensor:
        """Per-context mean of -dL_meta/dbeta * beta, plus a quadratic trust region on the context logits.

        Stationary points satisfy a_c (1 - tanh^2 u_c) = anchor (u_c - u0_c), so the logit shift is proportional
        to the mean alignment a_c instead of saturating whenever a_c keeps one sign.
        """
        index = context.clamp_min(0)
        weight = active.float()
        terms = torch.zeros(len(CONTEXTS), device=beta.device).index_add(0, index, unit * beta / self.beta_max * weight)
        counts = torch.zeros(len(CONTEXTS), device=beta.device).index_add(0, index, weight)
        return -(terms / counts.clamp_min(1)).sum() + 0.5 * anchor * self.anchor(beta.device)

    def forward(
        self, features: torch.Tensor, instruction: torch.Tensor, context: torch.Tensor, active: torch.Tensor,
        span_scale: float = 1.0,
    ) -> torch.Tensor:
        code = context_codes(context)
        raw = self.modulation(torch.cat((features.detach().float(), instruction.detach().float(), code), dim=1))
        centered = group_center(torch.tanh(raw.squeeze(1).float()), context, active)
        # |centered| < 2 and span < 0.5 keep m positive; its active context mean is exactly one.
        modulation = 1 + self.span * span_scale * centered
        return torch.where(active, self.context_beta(context) * modulation, torch.zeros_like(modulation))

    def table(self, device: torch.device) -> list[float]:
        return self.context_beta(torch.arange(len(CONTEXTS), device=device)).tolist()


class AlignmentScale:
    """Per-context running RMS of the hypergradient alignment, so each context steps on a unit scale."""

    def __init__(self, device: torch.device, decay: float = 0.98):
        self.decay = decay
        self.second = torch.zeros(len(CONTEXTS), device=device)
        self.seen = torch.zeros(len(CONTEXTS), device=device, dtype=torch.bool)

    def normalize(self, values: torch.Tensor, context: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        index = context.clamp_min(0)
        weight = active.float()
        counts = torch.zeros_like(self.second).index_add(0, index, weight)
        batch = torch.zeros_like(self.second).index_add(0, index, values.square() * weight) / counts.clamp_min(1)
        present = counts > 0
        update = torch.where(self.seen, self.decay * self.second + (1 - self.decay) * batch, batch)
        self.second = torch.where(present, update, self.second)
        self.seen |= present
        return torch.where(active, values / self.second[index].sqrt().clamp_min(1e-12), torch.zeros_like(values))


@torch.no_grad()
def directional_derivatives(params: list[torch.Tensor], direction: list[torch.Tensor], relative_step: float, evaluate):
    """Central finite difference of evaluate() along a unit direction; parameters are restored exactly."""
    norm = torch.sqrt(sum(d.float().square().sum() for d in direction)).clamp_min(1e-30)
    scale = relative_step * torch.sqrt(sum(p.float().square().sum() for p in params))
    saved = [p.detach().clone() for p in params]
    outputs = []
    try:
        for sign in (1.0, -1.0):
            for p, base, d in zip(params, saved, direction):
                p.copy_(base + (sign * scale / norm) * d.to(p.dtype))
            outputs.append(evaluate())
    finally:
        for p, base in zip(params, saved):
            p.copy_(base)
    return [(plus - minus) / (2 * scale) for plus, minus in zip(*outputs)], scale.item()
