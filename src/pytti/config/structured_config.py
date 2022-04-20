from dataclasses import MISSING
from functools import partial
from typing import Optional

import hydra
from attrs import define, field
from hydra.core.config_store import ConfigStore

from pytti.Image.VQGANImage import VQGAN_MODEL_NAMES


def check_input_against_list(attribute, value, valid_values):
    if value not in valid_values:
        raise ValueError(
            f"{value} is not a valid input for {attribute.name} Valid inputs are {valid_values}"
        )


@define(auto_attribs=True)
class AudioFilterConfig:
    variable_name: str = ""
    f_center: int = -1
    f_width: int = -1
    order: int = 5


@define(auto_attribs=True)
class ConfigSchema:
    #############
    ## Prompts ##
    #############
    scenes: str = ""
    scene_prefix: str = ""
    scene_suffix: str = ""

    direct_image_prompts: str = ""
    init_image: str = ""
    direct_init_weight: str = ""
    semantic_init_weight: str = ""

    ##################################

    image_model: str = field(default="Unlimited Palette")
    vqgan_model: str = field(default="sflckr")
    animation_mode: str = field(default="off")

    @image_model.validator
    def check(self, attribute, value):
        check_input_against_list(
            attribute,
            value,
            valid_values=[
                "Unlimited Palette",
                "Limimted Palette",
                "VQGAN",
            ],
        )

    # I feel like there should be a better way to do this...
    @vqgan_model.validator
    def check(self, attribute, value):
        check_input_against_list(
            attribute,
            value,
            valid_values=VQGAN_MODEL_NAMES,
        )

    @animation_mode.validator
    def check(self, attribute, value):
        check_input_against_list(
            attribute, value, valid_values=["off", "2D", "3D", "Video Source"]
        )

    ##################################

    width: int = 180
    height: int = 112

    steps_per_scene: int = 100
    steps_per_frame: int = 50
    interpolation_steps: int = 0

    learning_rate: Optional[float] = None  # based on pytti.Image.DifferentiableImage
    reset_lr_each_frame: bool = True
    seed: str = "${now:%f}"  # microsecond component of timestamp. Basically random.
    cutouts: int = 40
    cut_pow: int = 2
    cutout_border: float = 0.25
    border_mode: str = field(default="clamp")

    @border_mode.validator
    def check(self, attribute, value):
        check_input_against_list(
            attribute, value, valid_values=["clamp", "mirror", "wrap", "black", "smear"]
        )

    ##################################

    ##########
    # Camera #
    ##########

    field_of_view: int = 60
    near_plane: int = 1
    far_plane: int = 10000

    ######################
    ### Induced Motion ###
    ######################

    input_audio: str = ""
    input_audio_offset: float = 0
    input_audio_filters: Optional[AudioFilterConfig] = None

    #  _2d and _3d only apply to those animation modes

    translate_x: str = "0"
    translate_y: str = "0"
    translate_z_3d: str = "0"
    rotate_3d: str = "[1, 0, 0, 0]"
    rotate_2d: str = "0"
    zoom_x_2d: str = "0"
    zoom_y_2d: str = "0"

    sampling_mode: str = field(default="bicubic")

    @sampling_mode.validator
    def check(self, attribute, value):
        check_input_against_list(
            attribute, value, valid_values=["nearest", "bilinear", "bicubic"]
        )

    infill_mode: str = field(default="wrap")

    @infill_mode.validator
    def check(self, attribute, value):
        check_input_against_list(
            attribute, value, valid_values=["mirror", "wrap", "black", "smear"]
        )

    pre_animation_steps: int = 100
    lock_camera: bool = True

    ##################################

    #######################
    ### Limited Palette ###
    #######################

    pixel_size: int = 4
    smoothing_weight: float = 0.02
    random_initial_palette: bool = False
    palette_size: int = 6
    palettes: int = 9
    gamma: int = 1
    hdr_weight: float = 0.01
    palette_normalization_weight: float = 0.2
    show_palette: bool = False
    target_palette: str = ""
    lock_palette: bool = False

    ##############
    ### ffmpeg ###
    ##############

    frames_per_second: int = 12

    direct_stabilization_weight: str = ""
    semantic_stabilization_weight: str = ""
    depth_stabilization_weight: str = ""
    edge_stabilization_weight: str = ""
    flow_stabilization_weight: str = ""

    #####################################
    ### animation_mode = Video Source ###
    #####################################

    video_path: str = ""
    frame_stride: int = 1
    reencode_each_frame: bool = True
    flow_long_term_samples: int = 1

    ############
    ### CLIP ###
    ############

    ViTB32: bool = True
    ViTB16: bool = False
    ViTL14: bool = False
    RN50: bool = False
    RN101: bool = False
    RN50x4: bool = False
    RN50x16: bool = False
    RN50x64: bool = False

    ###############
    ### Outputs ###
    ###############

    file_namespace: str = "default"
    allow_overwrite: bool = False
    display_every: int = 50
    clear_every: int = 0
    display_scale: int = 1
    save_every: int = 50

    backups: int = 0
    show_graphs: bool = False
    approximate_vram_usage: bool = False
    use_tensorboard: Optional[bool] = False

    #####################################

    #################
    ### Model I/O ###
    #################

    # This is where pytti will expect to find model weights.
    # Each model will be assigned a separate subdirectory within this folder
    # If the expected model artifacts are not present, pytti will attempt to download them.
    models_parent_dir: str = "${user_cache:}"

    ######################################

    ##########################
    ### Performance tuning ###
    ##########################

    gradient_accumulation_steps: int = 1


def register():
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=ConfigSchema)


register()
