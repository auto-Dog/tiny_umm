import numpy as np
import torch

from data.schema import edit_prompt, generate_prompt, load_config
from inference import Pipeline
from models.bottleneck import ConditionalVisualBottleneck, GaussianBottleneck, latent_mean
from models.checkpoints import flow_state, save_checkpoint
from models.flow import FlowCondition, flow_matching_errors
from models.tiny_umm import TinyUMM


def test_bottleneck_samples_all_decoder_inputs_and_sums_kl():
    cfg = load_config(smoke=True)
    condition = FlowCondition(
        torch.ones(2, cfg["model"]["instruction_dim"]),
        torch.full((2, cfg["model"]["visual_tokens"], cfg["model"]["visual_dim"]), 0.5),
    )
    mean = latent_mean(condition)
    generator = torch.Generator().manual_seed(42)
    sampled, kl, std = GaussianBottleneck(cfg)(condition, generator)
    expected_std = torch.full_like(mean, 0.1)
    expected_kl = torch.distributions.kl_divergence(
        torch.distributions.Normal(mean, expected_std),
        torch.distributions.Normal(0.0, 1.0),
    )
    torch.testing.assert_close(kl, expected_kl.sum(dim=1))
    torch.testing.assert_close(std, expected_std.mean(dim=1))
    noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(latent_mean(sampled), mean + expected_std * noise)
    assert sampled.source is None


def test_generation_gradients_and_posttrained_checkpoint_inference(artifact_dir):
    cfg = load_config(smoke=True)
    cfg["paths"]["checkpoint_dir"] = str(artifact_dir)
    torch.manual_seed(42)
    model = TinyUMM(cfg, "connector", bottleneck=True)
    model.train_bridge(joint_instruction=True)
    model.bottleneck.requires_grad_(True)
    prompts = np.stack([generate_prompt(cfg, 2, 1, 1), generate_prompt(cfg, 3, 2, 0)])
    mean = model.condition(torch.tensor(prompts, dtype=torch.long))
    sampled, kl, _ = model.bottleneck(mean, torch.Generator().manual_seed(43))
    target = torch.zeros(2, cfg["image_size"], cfg["image_size"], 3)
    loss = flow_matching_errors(model.flow, target, sampled).mean() + 3e-6 * kl.mean()
    loss.backward()
    assert model.instruction.slots[0].weight.grad.abs().sum() > 0
    assert model.bottleneck.instruction_logvar.weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.understanding.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.flow.parameters())

    path = save_checkpoint(cfg, "posttrain", flow_state(model, True), 1, loss.item(), direct_source=False)
    pipeline = Pipeline(cfg, torch.device("cpu"), path)
    assert pipeline.model.bottleneck is not None
    images = pipeline.generate(prompts, steps=2, seed=44)
    np.testing.assert_array_equal(images, pipeline.generate(prompts, steps=2, seed=44))


@torch.no_grad()
def test_conditional_bottleneck_keeps_instruction_and_generation_intact(artifact_dir):
    cfg = load_config(smoke=True)
    cfg["paths"]["checkpoint_dir"] = str(artifact_dir)
    model = TinyUMM(cfg, "connector").eval()
    model.bottleneck = ConditionalVisualBottleneck(cfg, max_std_ratio=0.06)
    prompts = np.stack([generate_prompt(cfg, 2, 1, 1), edit_prompt(cfg, 2, 0)])
    tokens = torch.as_tensor(prompts, dtype=torch.long)
    group = model.bottleneck.prior_index(tokens)[1]
    counts = torch.zeros(len(model.bottleneck.prior_mean))
    first = torch.zeros_like(model.bottleneck.prior_mean)
    second = torch.zeros_like(first)
    counts[group] = 2
    first[group] = 2 * 0.2
    second[group] = 2 * (0.2 ** 2 + 0.5 ** 2)
    model.bottleneck.fit_prior(counts, first, second)
    source = torch.zeros(2, cfg["image_size"], cfg["image_size"], 3)
    mean = model.condition(tokens, source)
    sampled, kl, std = model.bottleneck(mean, torch.Generator().manual_seed(12), prompt=tokens)
    torch.testing.assert_close(sampled.instruction, mean.instruction, rtol=0, atol=0)
    torch.testing.assert_close(sampled.visual[0], mean.visual[0], rtol=0, atol=0)
    assert not torch.equal(sampled.visual[1], mean.visual[1])
    torch.testing.assert_close(std, torch.tensor([0.0, 0.03]))
    expected_kl = torch.distributions.kl_divergence(
        torch.distributions.Normal(mean.visual[1], 0.03), torch.distributions.Normal(0.2, 0.5),
    ).sum()
    torch.testing.assert_close(kl, torch.stack((torch.zeros_like(expected_kl), expected_kl)))
    other, target_kl, _ = model.bottleneck(
        mean, torch.Generator().manual_seed(12), prompt=tokens, reference=torch.ones_like(mean.visual),
    )
    torch.testing.assert_close(other.visual, sampled.visual, rtol=0, atol=0)
    assert target_kl[1] != kl[1]
    inactive, inactive_kl, _ = model.bottleneck(
        mean, torch.Generator().manual_seed(12), prompt=tokens, active=torch.zeros(2, dtype=torch.bool),
    )
    torch.testing.assert_close(inactive.visual, mean.visual, rtol=0, atol=0)
    torch.testing.assert_close(inactive_kl, torch.zeros_like(kl), rtol=0, atol=0)

    path = save_checkpoint(
        cfg, "posttrain", flow_state(model, True), 1, 0.0, direct_source=False, bottleneck_kind="conditional",
        posttrain={"max_std_ratio": 0.06},
    )
    pipeline = Pipeline(cfg, torch.device("cpu"), path)
    assert pipeline.model.bottleneck.max_log_ratio == model.bottleneck.max_log_ratio
    torch.testing.assert_close(pipeline.model.bottleneck.prior_scale, model.bottleneck.prior_scale)
    images = np.full((1, cfg["image_size"], cfg["image_size"], 3), 128, dtype=np.uint8)
    region = np.ones(images.shape[:3], dtype=bool)
    first = pipeline.edit(images, prompts[1:], region=region, steps=2, seed=44).raw
    second = pipeline.edit(images, prompts[1:], region=region, steps=2, seed=44).raw
    np.testing.assert_array_equal(first, second)


@torch.no_grad()
def test_appearance_bottleneck_preserves_orthogonal_query_coordinates():
    cfg = load_config(smoke=True)
    cfg["model"].update(visual_tokens=2, visual_dim=4, instruction_dim=8)
    bottleneck = ConditionalVisualBottleneck(cfg, appearance_rank=2)
    tokens = torch.as_tensor(edit_prompt(cfg, 2, 0), dtype=torch.long)[None].expand(4, -1)
    differences = torch.zeros(4, 8)
    differences[0, 2], differences[1, 2] = 2, -2
    differences[2, 6], differences[3, 6] = 1, -1
    bottleneck.fit_appearance_prior(differences.view(4, 2, 4), bottleneck.prior_index(tokens))
    mean = FlowCondition(torch.zeros(4, 8), torch.zeros(4, 2, 4))
    sampled, kl, _ = bottleneck(mean, torch.Generator().manual_seed(12), prompt=tokens)
    delta = sampled.visual.flatten(1)
    projected = delta @ bottleneck.basis @ bottleneck.basis.T
    torch.testing.assert_close(delta, projected)
    assert delta.abs().sum() > 0 and torch.isfinite(kl).all()


@torch.no_grad()
def test_protected_query_is_excluded_from_noise_and_rate():
    cfg = load_config(smoke=True)
    cfg["model"].update(visual_tokens=2, visual_dim=4, instruction_dim=8)
    bottleneck = ConditionalVisualBottleneck(cfg, protected_tokens=1)
    prompt = torch.as_tensor(edit_prompt(cfg, 2, 0), dtype=torch.long)[None]
    mean = FlowCondition(torch.zeros(1, 8), torch.zeros(1, 2, 4))
    sampled, kl, _ = bottleneck(mean, torch.Generator().manual_seed(12), prompt=prompt)
    mean.visual[:, 0] = 5.0
    changed, changed_kl, _ = bottleneck(mean, torch.Generator().manual_seed(12), prompt=prompt)
    torch.testing.assert_close(changed.visual[:, 0], mean.visual[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(changed.visual[:, 1], sampled.visual[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(changed_kl, kl, rtol=0, atol=0)
