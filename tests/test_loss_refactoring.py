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


video_fpath = str(next(Path(".").glob("**/assets/*.mp4")))
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
semantic_iniit_weight: 1
device: '{TEST_DEVICE}'
"""
    run_cfg(cfg_str)


def test_stabilization_weights():
    cfg_str = f"""# @package _global_
scenes: a photograph of an apple
depth_stabilization_weight: 1
edge_stabilization_weight: 1
direct_stabilization_weight: 1
semantic_stabilization_weight: 1
flow_stabilization_weight: 1
steps_per_frame: 10
steps_per_scene: 150
#flow_long_term_samples: 1
device: '{TEST_DEVICE}'
"""
    run_cfg(cfg_str)


def test_limited_palette_image_encode():
    cfg_str = f"""# @package _global_
scenes: a photograph of an apple
direct_image_prompts: '{img_fpath}:-1:-.5'
direct_init_weight: 1
semantic_iniit_weight: 1
image_model: Limited Palette
device: '{TEST_DEVICE}'
"""
    run_cfg(cfg_str)


def test_video_optical_flow():
    cfg_str = f"""# @package _global_
scenes: a photograph of an apple
animation_mode: Video Source
video_path: {video_fpath}
flow_stabilization_weight: 1
steps_per_frame: 10
steps_per_scene: 150
flow_long_term_samples: 3
device: '{TEST_DEVICE}'
height: 512
width: 512
pixel_size: 1
"""
    run_cfg(cfg_str)


def test_3D_optical_flow():
    cfg_str = f"""# @package _global_
scenes: a photograph of an apple
animation_mode: 3D
video_path: {video_fpath}
flow_stabilization_weight: 1
steps_per_frame: 10
steps_per_scene: 150
device: '{TEST_DEVICE}'
height: 512
width: 512
pixel_size: 1
translate_z_3d: 10
"""
    run_cfg(cfg_str)


# RuntimeError: CUDA error: device-side assert triggered
@pytest.mark.xfail
def test_3d_null_transform_bug():
    cfg_str = f"""# @package _global_
scenes: a photograph of an apple
animation_mode: 3D
video_path: {video_fpath}
flow_stabilization_weight: 1
steps_per_frame: 10
steps_per_scene: 150
device: '{TEST_DEVICE}'
height: 512
width: 512
pixel_size: 1
translate_z_3d: 0
rotate_3d: [1,0,0,0]
"""
    run_cfg(cfg_str)
