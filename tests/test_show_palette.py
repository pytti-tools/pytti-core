import pytest

from hydra import initialize, compose
from loguru import logger
from pytti.workhorse import _main as render_frames
from omegaconf import OmegaConf, open_dict


CONFIG_BASE_PATH = "config"
CONFIG_DEFAULTS = "default.yaml"


def run_cfg(cfg_str):
    with initialize(config_path=CONFIG_BASE_PATH):
        cfg_base = compose(
            config_name=CONFIG_DEFAULTS,
            overrides=[f"conf=_empty"],
        )
        cfg_this = OmegaConf.create(cfg_str)

        with open_dict(cfg_base) as cfg:
            cfg = OmegaConf.merge(cfg_base, cfg_this)
        render_frames(cfg)


def test_show_palette():
    cfg_str = f"""# @package _global_
scenes: a photograph of an apple
image_model: Limited Palette
show_palette: true
"""
    run_cfg(cfg_str)
