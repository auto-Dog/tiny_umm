import numpy as np
import pytest
import torch
from torch.nn import functional as F

from data.schema import load_config
from data.build_dataset import build
from models.checkpoints import flow_state, load_flow, save_checkpoint
from models.flow import FlowCondition
from models.tiny_umm import TinyUMM
from training.query_objectives import DynamicBeta, MatchingPairs, TaskMatcher, negative_answers
from training.train_shared import _options, configure_objective, train


def test_edit_negatives_preserve_target_attributes_and_reject_duplicate_images():
    images = np.arange(6, dtype=np.uint8)[:, None, None, None] * np.ones((6, 2, 2, 3), dtype=np.uint8)
    images[3] = images[2]
    data = {
        "images": images, "edit_targets": images, "scene_ids": np.arange(6),
        "counts": np.array([1, 1, 2, 2, 3, 4]), "edit_colors": np.zeros(6), "edit_shapes": np.zeros(6),
    }
    pairs = MatchingPairs(data, 23)
    np.testing.assert_array_equal(pairs.negative, [1, 0, -1, -1, -1, -1])
    assert pairs.fingerprint() == MatchingPairs(data, 23).fingerprint()
    assert 0 < MatchingPairs(data, 23, 0.01).labeled.sum() < 6


def test_text_negatives_change_exactly_one_slot_without_false_negatives():
    labels = torch.zeros(20, 3, dtype=torch.long)
    wrong = negative_answers(labels, (1, 2, 3), torch.Generator().manual_seed(3))
    assert ((wrong != labels).sum(dim=1) == 1).all()
    assert (wrong[:, 0] == 0).all()
    assert (wrong < torch.tensor([1, 2, 3])).all()
    assert torch.equal(wrong, negative_answers(labels, (1, 2, 3), torch.Generator().manual_seed(3)))


def test_matching_gradient_reaches_shared_flow_queries_and_image_backbone():
    cfg = load_config(smoke=True)
    torch.manual_seed(19)
    model = TinyUMM(cfg, "connector", shared_understanding=True).eval()
    matcher = TaskMatcher(cfg)
    source = torch.rand(2, 28, 28, 3) * 2 - 1
    prompt = model.understand_tokens[None].expand(2, -1)
    condition = model.condition(prompt, source)
    sampled = FlowCondition(condition.instruction, condition.visual + 0.003 * torch.randn_like(condition.visual))
    labels = torch.zeros(2, 3, dtype=torch.long)
    wrong = negative_answers(labels, (5, 11, 3), torch.Generator().manual_seed(2))
    errors, _ = matcher.errors(model.query_features(sampled), matcher.candidates(labels), matcher.candidates(wrong))
    watched = (
        model.understanding.backbone[1].weight, model.bridge.queries,
        model.flow.visual_attention.v_proj_weight, model.flow.output[-1].weight,
        model.source_encoder.project.weight,
    )
    gradients = torch.autograd.grad(errors.mean(), watched, allow_unused=True)
    assert all(g is not None and g.isfinite().all() and g.abs().sum() > 0 for g in gradients[:-1])
    assert gradients[-1] is None


def test_dynamic_beta_has_a_fixed_mean_and_cannot_minimize_its_own_rate_penalty():
    cfg = load_config(smoke=True)
    controller = DynamicBeta(cfg)
    features = torch.randn(4, 16 * (cfg["model"]["flow"]["width"] + 3), requires_grad=True)
    instruction = torch.randn(4, cfg["model"]["instruction_dim"], requires_grad=True)
    logits = controller(features, instruction)
    active = torch.tensor([True, True, False, True])
    multipliers = controller.multipliers(logits, active, 0.9)
    torch.testing.assert_close(multipliers[active].mean(), torch.tensor(1.0))
    assert (multipliers[active] > 0.1).all() and (multipliers[active] < 1.9).all()
    assert multipliers[2] == 1 and not multipliers.requires_grad
    rates = torch.arange(1.0, 5.0, requires_grad=True)
    (rates * multipliers).mean().backward()
    assert all(parameter.grad is None for parameter in controller.parameters())
    F.binary_cross_entropy_with_logits(logits, torch.tensor([0.1, 0.9, 0.1, 0.9])).backward()
    assert all(parameter.grad is not None for parameter in controller.parameters())
    assert features.grad is None and instruction.grad is None
    ordered = controller.multipliers(torch.tensor([-5.0, 0.0, 5.0]), torch.ones(3, dtype=torch.bool), 0.9)
    assert ordered[0] < ordered[1] < ordered[2]


@pytest.mark.parametrize("objective", ["baseline", "ib", "itm", "ib_itm", "dynamic_itm"])
def test_named_objectives_share_the_channel_and_preserve_loss_ablation(objective):
    args = _options().parse_args(["--run-dir", "unused", "--steps", "4", "--objective", objective])
    configure_objective(args)
    assert args.posterior_std == 0.003 and args.target_prior == "geometry"
    assert args.edit_beta == (0.0006 if objective in ("ib", "ib_itm", "dynamic_itm") else 0.0)
    assert args.text_beta == 0.0


def test_legacy_signed_kl_arguments_retain_their_defaults():
    args = _options().parse_args(["--run-dir", "unused", "--steps", "4", "--edit-beta", "-0.006"])
    configure_objective(args)
    assert args.edit_beta == -0.006 and args.posterior_std == 0.0 and args.target_prior is None


def test_partial_matching_training_saves_auxiliary_heads_and_loads_normal_inference(artifact_dir):
    cfg = load_config(smoke=True)
    cfg["paths"]["data_dir"] = str(artifact_dir / "data")
    cfg["paths"]["checkpoint_dir"] = str(artifact_dir / "base")
    cfg["data"]["scene_counts"] = {"train": 16, "val": 8, "test_random": 1, "test_comp": 1}
    build(cfg)
    base = TinyUMM(cfg, "connector", shared_understanding=True).eval()
    checkpoint = save_checkpoint(cfg, "shared", flow_state(base, True), 1, 0.0, direct_source=False)
    args = _options().parse_args([
        "--objective", "dynamic_itm", "--steps", "3", "--batch-size", "4", "--val-samples", "4",
        "--itm-label-fraction", "0.5", "--target-prior", "query", "--base-checkpoint", str(checkpoint),
        "--run-dir", str(artifact_dir / "trained"),
    ])
    path = train(cfg, torch.device("cpu"), args)
    restored, saved = load_flow(cfg, path, torch.device("cpu"))
    assert set(saved["matching_state"]) == {"matcher", "controller"}
    assert 0 < saved["posttrain"]["itm_labeled_scenes"] < 16
    assert saved["posttrain"]["parameter_delta_l2"]["beta_controller"] > 0
    source = torch.zeros(2, 28, 28, 3)
    predictions = restored.classify(source)
    assert all(value.isfinite().all() and len(value) == 2 for value in predictions.values())
