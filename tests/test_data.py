import math

import numpy as np
import pytest

from data.build_dataset import build, heldout_set
from data.dataset import load_group
from data.renderer import make_background, make_noise_field, render_objects, shape_coverage, shape_mask
from data.schema import load_config, palette, understand_prompt
from evaluation.metrics import SceneOracle


def test_dataset_splits_and_edit_invariants(artifact_dir):
    cfg = load_config(smoke=True)
    cfg["paths"]["data_dir"] = str(artifact_dir / "shards")
    cfg["data"]["scene_counts"] = {"train": 24, "val": 16, "test_random": 16, "test_comp": 12}
    cfg["data"]["shard_size"] = 11
    manifest = build(cfg)
    assert manifest["total_tasks"] == 3 * 68
    heldout = heldout_set(cfg)
    scene_ids = set()
    for name, expected in cfg["data"]["scene_counts"].items():
        data = load_group(cfg, name)
        assert len(data["images"]) == expected
        assert len(manifest["groups"][name]["shards"]) == math.ceil(expected / 11)
        assert all(n == expected for n in manifest["groups"][name]["tasks"].values())
        ids = set(data["scene_ids"].tolist())
        assert len(ids) == expected and ids.isdisjoint(scene_ids)
        scene_ids |= ids
        source = set(zip(data["counts"], data["colors"], data["shapes"]))
        target = set(zip(data["counts"], data["edit_colors"], data["edit_shapes"]))
        if name == "test_comp":
            assert source == heldout
        else:
            assert source.isdisjoint(heldout) and target.isdisjoint(heldout)
        assert np.all(data["prompts_und"] == understand_prompt(cfg))
        assert np.all(data["prompts_edit"][:, 1] == cfg["tokens"]["count_keep"])
        foreground = (data["src_masks"] | data["tgt_masks"]).astype(bool)
        region = data["region_masks"].astype(bool)
        assert np.all(~foreground | region)
        np.testing.assert_array_equal(data["images"][~foreground], data["edit_targets"][~foreground])
        angles = data["angles"]
        assert angles.dtype == np.float32 and angles.shape == (expected, 5)
        padded = np.arange(5)[None] >= data["counts"][:, None]
        assert np.all(angles[padded] == 0)
        angle_id = angles[~padded] * 12 / (2 * np.pi)
        np.testing.assert_allclose(angle_id, np.rint(angle_id), atol=1e-5)


def test_rotated_triangle_stays_inside_fixed_square_and_has_partial_pixels():
    cx, cy, side, angle = 14.2, 13.6, 10, math.pi / 6
    mask = shape_mask(2, cx, cy, side, 28, angle)
    y, x = np.nonzero(mask)
    assert np.max(np.abs(x + 0.5 - cx)) <= side / 2
    assert np.max(np.abs(y + 0.5 - cy)) <= side / 2
    np.testing.assert_array_equal(shape_coverage(2, cx, cy, side, 28, 1, angle) > 0, mask)
    coverage = shape_coverage(2, cx, cy, side, 28, 8, angle)
    assert np.any((coverage > 0) & (coverage < 1))


@pytest.fixture(scope="module")
def oracle():
    return SceneOracle(load_config(smoke=True))


@pytest.mark.parametrize("shape", [0, 1, 2])
def test_oracle_counts_and_distinguishes_rotated_noisy_shapes(oracle, shape):
    cfg = oracle.cfg
    pal = palette(cfg)
    background = make_background(28, pal[9], 2, 0, 1, 1, 0.035, 3, 1)
    noise = make_noise_field(28, 1, 0.018, 0.0, 200)
    centers = np.array([[8.25, 8.25], [18.5, 18.75]], dtype=np.float32)
    sides = np.array([9, 11], dtype=np.float32)
    rgb, _ = render_objects(
        shape, 5, centers, sides, 2, 28, background, noise, pal,
        supersample=8, angles=np.array([math.pi / 6, math.pi / 3]),
    )
    parsed = oracle.parse(rgb)
    assert (parsed.count, parsed.color, parsed.shape) == (2, 5, shape)
    np.testing.assert_allclose([(p.x, p.y) for p in parsed.objects], centers, atol=1)
    assert oracle.region(parsed).any()


def test_oracle_abstains_on_a_plain_background(oracle):
    image = np.full((28, 28, 3), 140, dtype=np.uint8)
    parsed = oracle.parse(image)
    assert not parsed.parsed and parsed.count is None
    assert not oracle.region(parsed).any()
