import pytest
import torch

from configs.latent_models import latent_recipe
from data.schema import edit_prompt, load_config
from models.latent_baselines import LatentBagel, LatentEmu3
from models.latent_checkpoints import load_latent_model, save_latent_checkpoint
from models.spatial_codec import SpatialCodec


@pytest.mark.parametrize("kind", ("emu3", "bagel"))
def test_frozen_codec_stays_fixed_and_checkpoint_reproduces_sampling(kind, artifact_dir):
    torch.manual_seed(31)
    cfg = load_config(smoke=True)
    recipe = latent_recipe(cfg)
    recipe["model"]["width"] = 32
    codec = SpatialCodec(cfg["image_size"], **recipe["codec"])
    codec.latent_mean.fill_(0.1)
    codec.latent_std.fill_(0.8)
    image = torch.rand(2, 28, 28, 3) * 2 - 1
    with torch.no_grad():
        vectors = codec.encode_raw(image).flatten(0, 2)
        codec.codebook.initialize(vectors, torch.Generator().manual_seed(32))
    model_type = LatentEmu3 if kind == "emu3" else LatentBagel
    model = model_type(cfg, recipe, codec).train()
    before = codec.codebook.centers.clone()
    prompt = torch.as_tensor(edit_prompt(cfg, 3, 1), dtype=torch.long).expand(2, -1)
    loss = model.image_loss(prompt, image.flip(1), image, mask=torch.ones(2, 28, 28), generator=torch.Generator().manual_seed(33))
    loss.backward()
    assert not codec.training
    assert all(p.grad is None and not p.requires_grad for p in codec.parameters())
    torch.testing.assert_close(codec.codebook.centers, before)
    if kind == "bagel":
        for expert in (model.layers[0].context, model.layers[0].generation):
            grad = expert.qkv.weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    else:
        assert model.embedding.weight.grad.abs().sum() > 0
    path = artifact_dir / f"{kind}.pt"
    save_latent_checkpoint(path, model, kind=kind, config=cfg, recipe=recipe, epoch=1)
    restored, _ = load_latent_model(path, torch.device("cpu"))
    model.eval()
    options = {"temperature": 0} if kind == "emu3" else {"steps": 2}
    with torch.no_grad():
        expected = model.sample(prompt, image, guidance=1, generator=torch.Generator().manual_seed(34), **options)
        actual = restored.sample(prompt, image, guidance=1, generator=torch.Generator().manual_seed(34), **options)
    assert actual.shape == image.shape
    torch.testing.assert_close(actual, expected)
