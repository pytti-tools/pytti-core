"""
Broad strokes, end-to-end testing because something is better than nothing,
which is what we have right now.
"""

from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
import pytest

CONFIG_BASE_PATH = "config"
CONFIG_DEFAULTS = "default.yaml"


def test_import():
    from pytti.workhorse import _main as render_frames

    assert True


# to do: E2E tests generate files, and setting the seed makes that process deterministic.
# should compare the outputs of these tests with "ground-truth" generated images
# to ensure consistency
class _E2e_FromYaml:
    # def do_the_thing(self, cfg_fpath, **kwargs):
    def do_the_thing(self, **overrides):
        # kwargs.update({'conf':cfg_path})
        from pytti.workhorse import _main as render_frames

        with initialize(config_path=CONFIG_BASE_PATH):
            cfg = compose(
                config_name=CONFIG_DEFAULTS,
                # overrides=[f"conf={cfg_fpath}"],
                overrides=[f"{k}={v}" for k, v in overrides.items()],
            )
            render_frames(cfg)

    def test_limited(self, **kwargs):
        # self.do_the_thing(cfg_fpath="_test_limited_palette.yaml")
        self.do_the_thing(conf="_test_limited_palette.yaml", **kwargs)
        assert True

    def test_unlimited(self, **kwargs):
        self.do_the_thing(conf="_test_unlimited_palette.yaml", **kwargs)
        assert True

    def test_vqgan(self, **kwargs):
        self.do_the_thing(conf="_test_vqgan.yaml", **kwargs)
        assert True


class TestE2e_ImageModels_FromYaml(_E2e_FromYaml):
    pass


@pytest.mark.parametrize(
    # "animation_mode", ["off","2D","3D","Video Source",
    # ("foobar", pytest.mark.fail), ("", pytest.mark.fail), (None, pytest.mark.fail)]
    "kwargs",
    [{"animation_mode": v} for v in ("off", "2D", "3D", "Video Source")],
)
class TestE2e_AnimationModes_FromYaml(_E2e_FromYaml):
    def _add_video_path_to_kwargs(self, kwargs):
        if kwargs["animation_mode"] == "Video Source":
            kwargs["video_path"] = "./src/pytti/assets/HebyMorgongava_512kb.mp4"
        return kwargs

    def test_limited(self, kwargs):
        kwargs = self._add_video_path_to_kwargs(kwargs)
        super().test_limited(**kwargs)

    def test_unlimited(self, kwargs):
        kwargs = self._add_video_path_to_kwargs(kwargs)
        super().test_unlimited(**kwargs)

    def test_vqgan(self, kwargs):
        kwargs = self._add_video_path_to_kwargs(kwargs)
        super().test_vqgan(**kwargs)


@pytest.mark.parametrize(
    # "animation_mode", ["off","2D","3D","Video Source",
    # ("foobar", pytest.mark.fail), ("", pytest.mark.fail), (None, pytest.mark.fail)]
    "kwargs",
    [
        {"depth_stabilization_weight": 1},
        {"flow_stabilization_weight": 1},
        {"direct_stabilization_weight": 1},
        {"semantic_stabilization_weight": 1},
        {"edge_stabilization_weight": 1},
    ],
)
class TestE2e_StabilizationModes_FromYaml(_E2e_FromYaml):
    def _add_video_path_to_kwargs(self, kwargs):
        if kwargs.get("animation_mode") == "Video Source":
            kwargs["video_path"] = "./src/pytti/assets/HebyMorgongava_512kb.mp4"
        return kwargs

    def test_limited(self, kwargs):
        kwargs = self._add_video_path_to_kwargs(kwargs)
        super().test_limited(**kwargs)

    def test_unlimited(self, kwargs):
        kwargs = self._add_video_path_to_kwargs(kwargs)
        super().test_unlimited(**kwargs)

    def test_vqgan(self, kwargs):
        kwargs = self._add_video_path_to_kwargs(kwargs)
        super().test_vqgan(**kwargs)


from pytti.Image.VQGANImage import VQGAN_MODEL_NAMES


@pytest.mark.parametrize(
    # "animation_mode", ["off","2D","3D","Video Source",
    # ("foobar", pytest.mark.fail), ("", pytest.mark.fail), (None, pytest.mark.fail)]
    "kwargs",
    [{"vqgan_model": v} for v in VQGAN_MODEL_NAMES],
)
class TestE2e_vqgan_models_FromYaml(_E2e_FromYaml):
    def _add_video_path_to_kwargs(self, kwargs):
        if kwargs.get("animation_mode") == "Video Source":
            kwargs["video_path"] = "./src/pytti/assets/HebyMorgongava_512kb.mp4"
        return kwargs

    def test_limited(self, kwargs):
        pass

    def test_unlimited(self, kwargs):
        pass

    def test_vqgan(self, kwargs):
        kwargs = self._add_video_path_to_kwargs(kwargs)
        super().test_vqgan(**kwargs)
