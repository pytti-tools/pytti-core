"""
Broad strokes, end-to-end testing because something is better than nothing,
which is what we have right now.
"""

from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf

CONFIG_BASE_PATH = "config"
CONFIG_DEFAULTS = "default.yaml"


def test_import():
    from pytti.workhorse import _main as render_frames

    assert True


class TestE2e_ImageModels_FromYaml:
    def do_the_thing(self, cfg_fpath):
        from pytti.workhorse import _main as render_frames

        with initialize(config_path=CONFIG_BASE_PATH):
            cfg = compose(config_name=CONFIG_DEFAULTS, overrides=[f"conf={cfg_fpath}"])
            render_frames(cfg)

    def test_limited(self):
        self.do_the_thing(cfg_fpath="_test_limited_palette.yaml")
        assert True

    def test_unlimited(self):
        self.do_the_thing(cfg_fpath="_test_unlimited_palette.yaml")
        assert True

    def test_vqgan(self):
        self.do_the_thing(cfg_fpath="_test_vqgan.yaml")
        assert True
