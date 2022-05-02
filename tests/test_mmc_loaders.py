import pytest

from hydra import initialize, compose
from loguru import logger
from pytti.workhorse import _main as render_frames
from omegaconf import OmegaConf, open_dict


CONFIG_BASE_PATH = "config"
CONFIG_DEFAULTS = "default.yaml"


cfg_yaml_openai = """# @package _global_
scenes: a photograph of an apple
use_mmc: true
mmc_models:
- architecture: clip
  publisher: openai
  id: RN50
- architecture: clip
  publisher: openai
  id: ViT-B/32
"""


def test_mmc_openai_models():

    with initialize(config_path=CONFIG_BASE_PATH):
        cfg_base = compose(
            config_name=CONFIG_DEFAULTS,
            overrides=[f"conf=_empty"],
        )
        cfg_mmc = OmegaConf.create(cfg_yaml_openai)

        with open_dict(cfg_base) as cfg:
            cfg = OmegaConf.merge(cfg_base, cfg_mmc)
        render_frames(cfg)


cfg_yaml_mlf = """# @package _global_
scenes: a photograph of an apple
use_mmc: true
mmc_models:
- architecture: clip
  publisher: mlfoundations
  id: RN50--yfcc15m
- architecture: clip
  publisher: mlfoundations
  id: ViT-B-32--laion400m_avg
"""


def test_mmc_mlf_models():

    with initialize(config_path=CONFIG_BASE_PATH):
        cfg_base = compose(
            config_name=CONFIG_DEFAULTS,
            overrides=[f"conf=_empty"],
        )
        cfg_mmc = OmegaConf.create(cfg_yaml_mlf)

        with open_dict(cfg_base) as cfg:
            cfg = OmegaConf.merge(cfg_base, cfg_mmc)
        render_frames(cfg)
