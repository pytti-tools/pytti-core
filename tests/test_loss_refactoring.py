import pytest

from hydra import initialize, compose
from loguru import logger
from pytti.workhorse import _main as render_frames
from omegaconf import OmegaConf, open_dict
import torch
from pathlib import Path

CONFIG_BASE_PATH = "config"
CONFIG_DEFAULTS = "default.yaml"

TEST_DEVICE = "cuda:0"  # "cuda:1"


# video_fpath = str(next(Path(".").glob("**/assets/*.mp4")))
img_fpath = str(next(Path(".").glob("**/src/pytti/assets/*.jpg")))


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


# @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multiple-GPUs")
def test_direct_init_weight():
    cfg_str = f"""# @package _global_
scenes: a photograph of an apple
direct_image_prompts: '{img_fpath}:-1:-.5'
direct_init_weight: 1
device: '{TEST_DEVICE}'
"""
    run_cfg(cfg_str)
