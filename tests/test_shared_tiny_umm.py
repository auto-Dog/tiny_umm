import numpy as np
import pytest
import torch

from data.dataset import normalize_images
from data.schema import edit_prompt, generate_prompt, load_config
from inference import predict_understanding
from models.bottleneck import SharedQueryBottleneck
from models.checkpoints import flow_state, load_understanding, save_checkpoint
from models.flow import FlowCondition, flow_matching_errors
from models.tiny_umm import TinyUMM


def test_shared_model_uses_the_same_queries_and_flow_on_all_three_tasks():
    cfg = load_config(smoke=True)
    torch.manual_seed(19)
    model = TinyUMM(cfg, "connector", shared_understanding=True).eval()
    source = torch.rand(2, cfg["image_size"], cfg["image_size"], 3) * 2 - 1
    gen = torch.as_tensor(generate_prompt(cfg, 2, 1, 0)).long()[None].expand(2, -1)
    edit = torch.as_tensor(edit_prompt(cfg, 1, 0)).long()[None].expand(2, -1)
    watched = (
        model.understanding.backbone[1].weight, model.instruction.fusion[0].weight, model.bridge.queries,
        model.flow.visual_attention.v_proj_weight, model.flow.output[-1].weight, model.text_aux.heads["count"].weight,
        model.source_encoder.project.weight, model.understanding.heads["count"].weight,
    )

    with pytest.raises(ValueError, match="direct source"):
        model.condition(edit, source, direct=True)
    torch.testing.assert_close(model.condition(gen).visual, model.condition(gen, source).visual, atol=0, rtol=0)
    assert model.condition(edit, source).source is None
    assert model.bridge.queries.shape[0] == 4

    def reaches(loss):
        gradients = torch.autograd.grad(loss, watched, allow_unused=True)
        return tuple(g is not None and g.isfinite().all().item() and g.abs().sum().item() > 0 for g in gradients)

    assert reaches(model.classify(source)["count"].square().mean()) == (True,) * 6 + (False,) * 2
    assert reaches(flow_matching_errors(model.flow, source, model.condition(edit, source)).mean()) == (
        (True,) * 5 + (False,) * 3
    )
    assert reaches(flow_matching_errors(model.flow, source, model.condition(gen)).mean()) == (
        (False,) + (True,) * 4 + (False,) * 3
    )


def test_shared_checkpoint_dispatches_to_the_sampled_text_path(artifact_dir):
    cfg = load_config(smoke=True)
    cfg["paths"]["checkpoint_dir"] = str(artifact_dir)
    cfg["evaluation"]["batch_size"] = 2
    model = TinyUMM(cfg, "connector", shared_understanding=True).eval()
    model.bottleneck = SharedQueryBottleneck(0.03)
    path = save_checkpoint(
        cfg, "shared", flow_state(model, True), 1, 0.0, direct_source=False, bottleneck_kind="shared_query",
        posttrain={"posterior_std": 0.03},
    )
    device = torch.device("cpu")
    loaded = load_understanding(cfg, device, path)
    images = np.full((2, cfg["image_size"], cfg["image_size"], 3), 128, dtype=np.uint8)
    source = torch.from_numpy(normalize_images(images))
    actual = predict_understanding(loaded, images, cfg, device, seed=52)
    reference = model.classify(source, generator=torch.Generator().manual_seed(52))
    for key, scores in reference.items():
        np.testing.assert_array_equal(actual[key], scores.detach().argmax(dim=1).numpy())

    query = model.condition(model.understand_tokens[None].expand(2, -1), source).visual
    prior_std = 0.35
    target = torch.full_like(query, 0.2)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(query, model.bottleneck.posterior_std),
        torch.distributions.Normal(target, prior_std),
    ).flatten(1).mean(dim=1)
    torch.testing.assert_close(model.bottleneck.rate(query, target, prior_std), expected)


def test_pixel_weights_apply_to_velocity_error_without_changing_the_flow_path():
    class ZeroVelocity(torch.nn.Module):
        def forward(self, state, time, condition):
            return torch.zeros_like(state)

    target = torch.zeros(2, 4, 4, 3)
    weights = torch.linspace(0.5, 1.5, 32).reshape(2, 4, 4)
    noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(3))
    condition = FlowCondition(torch.zeros(2, 1), torch.zeros(2, 1, 1))
    actual = flow_matching_errors(
        ZeroVelocity(), target, condition, torch.Generator().manual_seed(3), pixel_weight=weights,
    )
    torch.testing.assert_close(actual, (noise.square() * weights[..., None]).mean(dim=(1, 2, 3)))


def test_relaxed_query_rate_matches_full_covariance_and_mean_gradient():
    bottleneck = SharedQueryBottleneck(0.03)
    visual = torch.tensor([[[0.2, 0.4], [-0.3, 0.1]]], requires_grad=True)
    reference = torch.zeros_like(visual)
    basis = torch.tensor([[0.5], [0.5], [-0.5], [-0.5]])
    prior_std, relaxed_std = 0.35, 1.4
    rate = bottleneck.rate(visual, reference, prior_std, relaxed_basis=basis, relaxed_std=relaxed_std)
    covariance = prior_std ** 2 * torch.eye(4) + (relaxed_std ** 2 - prior_std ** 2) * (basis @ basis.T)
    prior = torch.distributions.MultivariateNormal(reference.flatten(1), covariance)
    posterior = torch.distributions.MultivariateNormal(visual.flatten(1), 0.03 ** 2 * torch.eye(4))
    torch.testing.assert_close(rate, torch.distributions.kl_divergence(posterior, prior) / 4)
    gradient, = torch.autograd.grad(rate.sum(), visual)
    expected = torch.linalg.solve(covariance, visual.detach().flatten()[:, None]).reshape_as(visual) / 4
    torch.testing.assert_close(gradient, expected)
