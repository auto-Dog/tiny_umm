import math

import torch

from data.schema import load_config
from training.signed_beta import AlignmentScale, SignedBeta, directional_derivatives, grouped_nce


def _controller(span=0.4):
    cfg = load_config(smoke=True)
    torch.manual_seed(3)
    width = 16 * (cfg["model"]["flow"]["width"] + 3)
    return cfg, width, SignedBeta(cfg, 0.002, (0.0006, 0.0006, 0.0006, -0.001), span)


def test_signed_beta_keeps_one_sign_per_context_and_unit_mean_modulation():
    cfg, width, controller = _controller()
    torch.nn.init.normal_(controller.modulation[-1].weight, std=3.0)
    context = torch.tensor([0, 0, 1, 2, 2, -1, 3, 3])
    active = context >= 0
    features = torch.randn(8, width)
    instruction = torch.randn(8, cfg["model"]["instruction_dim"])
    beta = controller(features, instruction, context, active)
    base = controller.context_beta(context)
    assert torch.allclose(base[:5], torch.tensor(0.0006), atol=1e-9) and torch.allclose(base[6:], torch.tensor(-0.001))
    assert beta[5] == 0
    for c in range(4):
        rows = context == c
        ratio = beta[rows] / base[rows]
        assert (ratio > 0).all() and (ratio < 2).all()
        assert torch.isclose(ratio.mean(), torch.tensor(1.0), atol=1e-6)
    assert (beta.abs() < 0.002 * 2).all()


def test_controller_is_trained_only_by_its_own_loss():
    cfg, width, controller = _controller()
    features = torch.randn(4, width, requires_grad=True)
    instruction = torch.randn(4, cfg["model"]["instruction_dim"], requires_grad=True)
    beta = controller(features, instruction, torch.tensor([0, 1, 2, 3]), torch.ones(4, dtype=torch.bool))
    rate = (features.square().mean(dim=1) * beta.detach()).sum()
    rate.backward()
    assert all(p.grad is None for p in controller.parameters())
    assert instruction.grad is None


def test_grouped_nce_is_bounded_translation_invariant_and_masks_contexts():
    torch.manual_seed(0)
    mu = torch.randn(9, 4, 8)
    z = mu + 0.01 * torch.randn_like(mu)
    context = torch.tensor([0, 0, 0, 1, 1, 2, -1, 3, 3])
    tau = torch.full((4,), 2.0)
    nce, valid = grouped_nce(z, mu, context, tau)
    assert valid.tolist() == [True] * 5 + [False, False, True, True]
    k = torch.tensor([3, 3, 3, 2, 2, 1, 1, 2, 2]).float()
    assert (nce[valid] <= k[valid].log() + 1e-6).all() and (nce[~valid] == 0).all()
    shifted, _ = grouped_nce(z + 5.0, mu + 5.0, context, tau)
    assert torch.allclose(nce, shifted, atol=1e-4)


def test_finite_difference_matches_autograd_and_restores_parameters():
    torch.manual_seed(1)
    layer = torch.nn.Linear(5, 3)
    x = torch.randn(6, 5)
    params = list(layer.parameters())
    before = [p.detach().clone() for p in params]
    direction = [torch.randn_like(p) for p in params]
    (derivative,), _ = directional_derivatives(params, direction, 1e-3, lambda: (layer(x).square().sum(dim=1),))
    assert all(torch.equal(p, b) for p, b in zip(params, before))
    norm = math.sqrt(sum(d.square().sum().item() for d in direction))
    for i in range(6):
        grads = torch.autograd.grad(layer(x[i:i + 1]).square().sum(), params)
        exact = sum((g * d).sum() for g, d in zip(grads, direction)) / norm
        assert torch.isclose(derivative[i], exact, rtol=1e-3, atol=1e-5)


def test_alignment_scale_normalizes_each_context_separately():
    scale = AlignmentScale(torch.device("cpu"))
    values = torch.tensor([10.0, -10.0, 0.1, -0.1, 3.0])
    context = torch.tensor([0, 0, 3, 3, -1])
    unit = scale.normalize(values, context, context >= 0)
    assert torch.allclose(unit[:4], torch.tensor([1.0, -1.0, 1.0, -1.0])) and unit[4] == 0


def test_anchored_hyper_loss_settles_in_proportion_to_alignment():
    _, _, controller = _controller()
    optimizer = torch.optim.SGD(controller.sign.parameters(), lr=0.05, momentum=0.9)
    context = torch.tensor([0, 1, 2, 3])
    active = torch.ones(4, dtype=torch.bool)
    alignment = torch.tensor([0.05, -0.05, 0.0, 0.2])
    for _ in range(3000):
        beta = controller.context_beta(context)
        loss = controller.hyper_loss(alignment, beta, context, active, 0.25)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    u = controller.context_logits(context).detach()
    expected = alignment * (1 - torch.tanh(u).square()) / 0.25
    assert torch.allclose(u - controller.offset, expected, atol=2e-3)
    assert controller.table(torch.device("cpu"))[1] < 0.0006 < controller.table(torch.device("cpu"))[0]
