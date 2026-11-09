import numpy as np
import pytest
import torch
from torch import nn

from data.schema import edit_prompt, generate_prompt, load_config
from inference import Pipeline
from models.checkpoints import flow_state, save_checkpoint
from models.flow import FlowCondition, flow_matching_errors, linear_path, sample_flow
from models.tiny_umm import TinyUMM


def test_linear_flow_path_and_midpoint_integration():
    start = torch.full((2, 28, 28, 3), -0.3)
    end = torch.full_like(start, 0.7)
    x, v = linear_path(start, end, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(x[0], start[0])
    torch.testing.assert_close(x[1], end[1])
    torch.testing.assert_close(v, torch.ones_like(v))

    class Velocity(nn.Module):
        image_size = 28

        def forward(self, x, t, condition):
            return (0.2 + 0.4 * t[:, None, None, None]).expand_as(x)

    condition = FlowCondition(torch.zeros((2, 8)), torch.zeros((2, 4, 8)))
    result = sample_flow(Velocity(), condition, steps=3, noise=start, solver="midpoint")
    torch.testing.assert_close(result, start + 0.4)


def test_velocity_distillation_reuses_flow_path_and_stops_teacher_gradients():
    class ConstantVelocity(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = nn.Parameter(torch.tensor(value))

        def forward(self, x, t, condition):
            self.t = t
            return self.value.expand_as(x)

    student, teacher = ConstantVelocity(1.0), ConstantVelocity(2.0)
    target = torch.zeros(2, 4, 4, 3)
    condition = FlowCondition(torch.zeros(2, 1), torch.zeros(2, 1, 1))
    weights = torch.tensor([0.0, 3.0])
    ordinary = flow_matching_errors(student, target, condition, torch.Generator().manual_seed(11))
    distilled = flow_matching_errors(
        student, target, condition, torch.Generator().manual_seed(11), teacher=(teacher, condition, weights),
    )
    torch.testing.assert_close(student.t, teacher.t)
    torch.testing.assert_close(distilled - ordinary, weights * (1 - teacher.t).square())
    distilled.sum().backward()
    assert student.value.grad is not None and teacher.value.grad is None


@pytest.mark.parametrize("bridge", ["connector", "discrete"])
def test_frozen_flow_passes_gradients_to_visual_bridge(bridge):
    torch.manual_seed(10)
    cfg = load_config(smoke=True)
    model = TinyUMM(cfg, bridge)
    model.train_bridge()
    source = torch.rand((2, 28, 28, 3)) * 2 - 1
    tokens = torch.tensor(np.stack([edit_prompt(cfg, 3, 0), generate_prompt(cfg, 2, 1, 1)]), dtype=torch.long)
    condition = model.condition(tokens, source)
    assert condition.source is None
    assert condition.visual.shape == (2, 4, cfg["model"]["visual_dim"])
    torch.testing.assert_close(condition.visual[1], model.flow.null_visual)
    loss = flow_matching_errors(model.flow, source.flip(1), condition).mean()
    loss.backward()
    assert all(p.grad is None and not p.requires_grad for name, p in model.named_parameters() if not name.startswith("bridge."))
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.bridge.parameters())
    assert not model.understanding.training and not model.flow.training


def test_edit_pipeline_keeps_source_bytes_outside_region_over_multiple_rounds(artifact_dir):
    cfg = load_config(smoke=True)
    cfg["paths"]["checkpoint_dir"] = str(artifact_dir)
    torch.manual_seed(21)
    model = TinyUMM(cfg, "connector")
    path = save_checkpoint(cfg, "connector", flow_state(model, True), 1, 1.0, direct_source=False)
    pipeline = Pipeline(cfg, torch.device("cpu"), path)
    source = np.random.default_rng(20).integers(0, 256, size=(1, 28, 28, 3), dtype=np.uint8)
    region = np.zeros((28, 28), dtype=bool)
    region[6:18, 9:20] = True
    prompts = [edit_prompt(cfg, 3, 1)[None], edit_prompt(cfg, 1, 2)[None]]
    outputs = pipeline.chain(source, prompts, region=region, steps=2, seed=11)
    assert len(outputs) == 2 and np.any(outputs[0].raw[0, ~region] != source[0, ~region])
    for result in outputs:
        np.testing.assert_array_equal(result.constrained[0, ~region], source[0, ~region])
        np.testing.assert_array_equal(result.constrained[0, region], result.raw[0, region])
