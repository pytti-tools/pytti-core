"""
Trying to construct test case to reproduce
  https://github.com/pytti-tools/pytti-core/issues/82
"""

from omegaconf import OmegaConf
from pytti.workhorse import _main as render_frames
from pathlib import Path

# video_fpath = str(next(Path(".").glob("**/assets/*.mp4")))
video_fpath = str(Path(".").glob("**/src/assets/HebyMorgongava_512kb.mp4"))
print(video_fpath)
params = {
    # "scenes": "sunlight:3_testmasktest.mp4 | midnight:3_-testmasktest.mp4",
    "scenes": "sunlight",
    "scene_prefix": "",
    "scene_suffix": "",
    "interpolation_steps": 0,
    "steps_per_scene": 100,  # 4530,
    "direct_image_prompts": "",
    "init_image": "",
    "direct_init_weight": "",
    "semantic_init_weight": "",
    "image_model": "Limited Palette",
    "width": 360,
    "height": 640,
    "pixel_size": 1,
    "smoothing_weight": 0.02,
    "vqgan_model": "sflckr",
    "random_initial_palette": False,
    "palette_size": 6,
    "palettes": 9,
    "gamma": 1,
    "hdr_weight": 0.01,
    "palette_normalization_weight": 0.2,
    "show_palette": False,
    "target_palette": "",
    "lock_palette": False,
    "animation_mode": "Video Source",
    "sampling_mode": "bilinear",
    "infill_mode": "smear",
    "pre_animation_steps": 0,
    "steps_per_frame": 10,
    "frames_per_second": 12,
    "direct_stabilization_weight": "",  # "testmasktest.mp4",
    "semantic_stabilization_weight": "",
    "depth_stabilization_weight": "",
    "edge_stabilization_weight": "",
    "flow_stabilization_weight": "",  # "testmasktest.mp4",
    "video_path": video_fpath,  # "testmasktest.mp4",
    "frame_stride": 1,
    "reencode_each_frame": False,
    "flow_long_term_samples": 1,
    "translate_x": "0",
    "translate_y": "0",
    "translate_z_3d": "0",
    "rotate_3d": "[1,0,0,0]",
    "rotate_2d": "0",
    "zoom_x_2d": "0",
    "zoom_y_2d": "0",
    "lock_camera": True,
    "field_of_view": 60,
    "near_plane": 1,
    "far_plane": 10000,
    "file_namespace": "default",
    "allow_overwrite": False,
    "display_every": 10,
    "clear_every": 0,
    "display_scale": 1,
    "save_every": 10,
    "backups": 5,
    "show_graphs": False,
    "approximate_vram_usage": False,
    "ViTB32": True,
    "ViTB16": False,
    "RN50": False,
    "RN50x4": False,
    "ViTL14": False,
    "RN101": False,
    "RN50x16": False,
    "RN50x64": False,
    "learning_rate": None,
    "reset_lr_each_frame": True,
    "seed": 15291079827822783929,
    "cutouts": 40,
    "cut_pow": 2,
    "cutout_border": 0.25,
    "gradient_accumulation_steps": 1,
    "border_mode": "clamp",
    "models_parent_dir": ".",
    ##########################
    # adding new config items for backwards compatibility
    "use_tensorboard": True,  # This should actually default to False. Prior to April2022, tb was non-optional
    # Default null audio input parameters
    "input_audio": "",
    "input_audio_offset": 0,
    "input_audio_filters": [],
}


def test_issue83():
    """
    Reproduce https://github.com/pytti-tools/pytti-core/issues/82
    """
    cfg = OmegaConf.create(params)
    render_frames(cfg)
