import pytest

from hydra import initialize, compose
from loguru import logger
from pytti.workhorse import _main as render_frames
from omegaconf import OmegaConf, open_dict
import torch

CONFIG_BASE_PATH = "config"
CONFIG_DEFAULTS = "default.yaml"

TEST_DEVICE = "cuda:1"


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


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multiple-GPUs")
def test_mmc_device():
    cfg_str = f"""# @package _global_
scenes: a photograph of an apple
use_mmc: true
mmc_models:
- architecture: clip
  publisher: openai
  id: RN50
  #device: '{TEST_DEVICE}'
device: '{TEST_DEVICE}'
"""
    run_cfg(cfg_str)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multiple-GPUs")
def test_vqgan_device():
    cfg_str = f"""# @package _global_
scenes: a photograph of an apple
image_model: VQGAN
device: '{TEST_DEVICE}'
"""
    run_cfg(cfg_str)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multiple-GPUs")
def test_depth_device():
    cfg_str = f"""# @package _global_
scenes: a photograph of an apple
depth_stabilization_weight: 1
device: '{TEST_DEVICE}'
"""
    run_cfg(cfg_str)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multiple-GPUs")
def test_flow_device():
    cfg_str = f"""# @package _global_
scenes: a photograph of an apple
flow_stabilization_weight: 1
device: '{TEST_DEVICE}'
"""
    run_cfg(cfg_str)
