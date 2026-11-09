import pytest
import torch

from configs.base_models import base_recipe
from data.schema import edit_prompt, load_config, understand_prompt
from models.base_checkpoints import BASE_TYPES


def model_for(kind):
    cfg = load_config(smoke=True)
    settings = {**base_recipe(cfg, kind)["model"], "width": 32}
    model = BASE_TYPES[kind](cfg, settings)
    image = torch.randn(1, cfg["image_size"], cfg["image_size"], 3).tanh()
    if kind == "emu3":
        patches = model.patches.pack(image).flatten(0, 1)
        model.codebook.fit(patches, 2, torch.Generator().manual_seed(10))
    return cfg, model, image


@pytest.mark.parametrize("kind", ("emu3", "bagel"))
@torch.no_grad()
def test_answer_logits_use_only_previous_labels(kind):
    torch.manual_seed(5)
    cfg, model, image = model_for(kind)
    prompt = torch.as_tensor(understand_prompt(cfg), dtype=torch.long)[None]
    labels = torch.tensor([[0, 1, 2]])
    logits = model.answer_logits(prompt, image, labels)
    different_count = model.answer_logits(prompt, image, torch.tensor([[2, 1, 2]]))
    different_color = model.answer_logits(prompt, image, torch.tensor([[0, 2, 2]]))
    torch.testing.assert_close(logits[:, :1], different_count[:, :1])
    torch.testing.assert_close(logits[:, :2], different_color[:, :2])
    predicted = model.predict(prompt, image)
    rolled_logits = model.answer_logits(prompt, image, predicted)
    torch.testing.assert_close(predicted, model.vocab.greedy_answers(rolled_logits))


@torch.no_grad()
def test_emu3_raster_and_cached_decoding_match_full_prefix():
    torch.manual_seed(6)
    cfg, model, image = model_for("emu3")
    torch.testing.assert_close(model.patches.unpack(model.patches.pack(image)), image)
    prompt = torch.as_tensor(edit_prompt(cfg, 1, 1), dtype=torch.long)[None]
    prefix = model._prefix(prompt, image, model.vocab.image_start)
    next_tokens = model.image_ids(image)[:, :4]
    expected = model(torch.cat((prefix, next_tokens), dim=1))[:, prefix.shape[1] - 1:]
    cache = []
    outputs = [model._logits(model._hidden(prefix, cache)[:, -1:])]
    for token in next_tokens.split(1, dim=1):
        outputs.append(model._logits(model._hidden(token, cache)))
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected)
    assert model.sample(prompt, image, temperature=0).shape == image.shape


def test_bagel_target_patches_share_context_and_train_both_experts():
    torch.manual_seed(7)
    cfg, model, image = model_for("bagel")
    prompt = torch.as_tensor(edit_prompt(cfg, 1, 1), dtype=torch.long)[None]
    prefix = model._prefix(prompt, image, model.vocab.image_start)
    context_length = prefix.shape[1]
    mask = model._mask(context_length, model.patches.count, has_source=True)
    assert mask[5:5 + model.patches.count, 5:5 + model.patches.count].all()
    assert mask[context_length:].all()
    assert not mask[:context_length, context_length:].any()
    assert not mask[0, 1]
    loss = model.image_loss(prompt, image.flip(1), image, generator=torch.Generator().manual_seed(3))
    loss.backward()
    for expert in (model.layers[0].context, model.layers[0].generation):
        grad = expert.qkv.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
