import numpy as np
import torch

from data.schema import generate_prompt, load_config
from inference import Pipeline
from models.bottleneck import GaussianBottleneck, latent_mean
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
