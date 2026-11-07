import numpy as np

from data.schema import edit_prompt, load_config
from evaluation.evaluate import editing_report
from evaluation.metrics import ParsedScene
from inference import EditResult


def test_edit_sampler_never_receives_evaluation_targets():
    cfg = load_config(smoke=True)
    region = np.zeros((28, 28), dtype=bool)
    region[6:12, 7:13] = True
    source = np.zeros((1, 28, 28, 3), dtype=np.uint8)
    called = False

    class Split(dict):
        def __getitem__(self, name):
            if name in {"edit_targets", "region_masks", "src_masks", "tgt_masks"}:
                assert called, f"read {name} before calling the sampler"
            return super().__getitem__(name)

    class PipelineSpy:
        metadata = {"direct_source": False}

        def edit(self, images, prompts, *, region, steps, seed, gain, direct):
            nonlocal called
            np.testing.assert_array_equal(images, source)
            np.testing.assert_array_equal(prompts, data["prompts_edit"])
            np.testing.assert_array_equal(region[0], selected)
            called = True
            return EditResult(images.copy(), images.copy(), region)

    class OracleSpy:
        def parse(self, image):
            return ParsedScene(None, None, None, (), np.zeros((28, 28), dtype=bool))

        def region(self, parsed):
            return selected

    selected = region.copy()
    selected[12, 10] = True
    data = Split(
        images=source, prompts_edit=edit_prompt(cfg, 0, 0)[None], counts=np.array([1]),
        edit_targets=np.ones_like(source), region_masks=region[None],
        src_masks=region[None], tgt_masks=region[None],
        centers=np.array([[[9.5, 8.5]]]), sizes=np.array([[6]]),
    )
    pipeline = PipelineSpy()
    pipeline.cfg = cfg
    report = editing_report(pipeline, data, np.array([0]), OracleSpy(), steps=2, seed=42)
    assert called
    assert report["constrained"]["outside_selected_region"]["exact_match"] == 1.0
    assert report["localization"]["coverage"] == 1.0
    assert report["localization"]["precision"] < 1.0
