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


def test_simple():
    from pytti.workhorse import _main as render_frames

    c_o = "_test.yaml"
    with initialize(config_path=CONFIG_BASE_PATH):
        cfg = compose(config_name=CONFIG_DEFAULTS, overrides=[f"conf={c_o}"])
        render_frames(cfg)
