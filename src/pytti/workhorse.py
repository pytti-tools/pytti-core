"""
This is the rendering logic that used to live in the notebook.
It's sort of a mess in here. I'm working on it.
Thank you for your patience.-- The Management
"""

import gc
import json
from pathlib import Path
import os
import re
from sqlite3 import NotSupportedError
import sys
import subprocess

import hydra
from loguru import logger
from omegaconf import OmegaConf, DictConfig

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter

# Deprecate or move functionality somewhere less general
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display

logger.info("Loading pytti...")
from pytti.Notebook import (
    change_tqdm_color,  # why though?
    get_last_file,
    get_next_file,
    load_settings,  # hydra should handle this stuff
    save_settings,
    save_batch,
    CLIP_MODEL_NAMES,
    load_clip,
    # get_frames,
    # build_loss,
    format_params,
    # clear_rotoscopers,
)
from pytti.rotoscoper import clear_rotoscopers, get_frames
from pytti.LossAug import build_loss

from pytti.Image import PixelImage, RGBImage, VQGANImage
from pytti.ImageGuide import DirectImageGuide
from pytti.Perceptor.Embedder import HDMultiClipEmbedder
from pytti.Perceptor.Prompt import parse_prompt

# HSVLoss no longer available to users?
# ... nm, gets used in PixelImage
from pytti.LossAug import TVLoss, HSVLoss, OpticalFlowLoss, TargetFlowLoss

from pytti import (
    fetch,
    vram_usage_mode,
    print_vram_usage,
    reset_vram_usage,
    vram_profiling,
)
from pytti.LossAug.DepthLossClass import init_AdaBins

logger.info("pytti loaded.")

change_tqdm_color()

logger.debug(sys.path)


sns.set()
plt.style.use("bmh")
pd.options.display.max_columns = None
pd.options.display.width = 175


TB_LOGDIR = "logs"  # to do: make this more easily configurable
# writer = SummaryWriter(TB_LOGDIR)
OUTPATH = f"{os.getcwd()}/images_out/"

# To do: ove remaining gunk into this...
# class Renderer:
#    """
#    High-level orchestrator for pytti rendering procedure.
#    """
#
#    def __init__(self, params):
#        pass


# this is the only place `parse_prompt` is invoked.
# combine load_scenes, parse_prompt, and parse into a unified, generic parser.
# generic here means the output of the parsing process shouldn't be bound to
# modules yet, just a collection of settings.
def parse_scenes(
    embedder,
    scenes,
    scene_prefix,
    scene_suffix,
):
    """
    Parses scenes separated by || and applies provided prefixes and suffixes to each scene.

    :param embedder: The embedder object
    :param params: The experiment parameters
    :return: The embedder and the prompts.
    """
    logger.info("Loading prompts...")
    prompts = [
        [
            parse_prompt(embedder, p.strip())
            for p in (scene_prefix + stage + scene_suffix).strip().split("|")
            if p.strip()
        ]
        for stage in scenes.split("||")
        if stage
    ]
    logger.info("Prompts loaded.")
    return embedder, prompts


def load_init_image(
    init_image_path=None,
    height: int = -1,  # why '-1'? should be None. Or btter yet, assert that it's greater than zero
    width: int = -1,
):
    """
    If the user has specified an image to use as the initial image, load it. Otherwise, if the
    user has specified a width or height, create a blank image of the specified size

    :param init_image_path: A local path or URL describing where to load the image from
    :param height: height of the image to be generated.
    :return: the initial image and the size of the initial image.
    """
    if init_image_path:
        init_image_pil = Image.open(fetch(init_image_path)).convert("RGB")
        init_size = init_image_pil.size
        # automatic aspect ratio matching
        if width == -1:
            width = int(height * init_size[0] / init_size[1])
        if height == -1:
            height = int(width * init_size[1] / init_size[0])
    else:
        init_image_pil = None
    return init_image_pil, height, width


def load_video_source(
    video_path: str,
    pre_animation_steps: int,
    steps_per_frame: int,
    height: int,
    width: int,
    init_image_pil: Image.Image,
):
    """
    Loads a video file and returns a PIL image of the first frame

    :param video_path: The path to the video file
    :param pre_animation_steps: The number of frames to skip at the beginning of the video
    :param steps_per_frame: How many steps to take per frame
    :param height: the height of the output image
    :param width: the width of the output image
    :return: The video frames, the initial image, the height and width of the image.
    """
    logger.info(f"loading {video_path}...")
    video_frames = get_frames(video_path)
    pre_animation_steps = max(steps_per_frame, pre_animation_steps)
    if init_image_pil is None:
        init_image_pil = Image.fromarray(video_frames.get_data(0)).convert("RGB")
        # enhancer = ImageEnhance.Contrast(init_image_pil)
        # init_image_pil = enhancer.enhance(2)
        init_size = init_image_pil.size
        if width == -1:
            width = int(height * init_size[0] / init_size[1])
        if height == -1:
            height = int(width * init_size[1] / init_size[0])
    return video_frames, init_image_pil, height, width


@hydra.main(config_path="config", config_name="default")
def _main(cfg: DictConfig):
    # params = OmegaConf.to_container(cfg, resolve=True)
    params = cfg
    logger.debug(params)
    logger.debug(OmegaConf.to_container(cfg, resolve=True))
    latest = -1

    writer = None
    if params.use_tensorboard:
        writer = SummaryWriter(TB_LOGDIR)

    batch_mode = False  # @param{type:"boolean"}

    ### Move these into default.yaml
    # @markdown check `restore` to restore from a previous run
    restore = False  # @param{type:"boolean"}
    # @markdown check `reencode` if you are restoring with a modified image or modified image settings
    reencode = False  # @param{type:"boolean"}
    # @markdown which run to restore
    restore_run = latest  # @param{type:"raw"}

    # NB: `backup/` dir probably not working at present
    if restore and restore_run == latest:
        _, restore_run = get_last_file(
            f"backup/{params.file_namespace}",
            f"^(?P<pre>{re.escape(params.file_namespace)}\\(?)(?P<index>\\d*)(?P<post>\\)?_\\d+\\.bak)$",
        )

    # I feel like there's probably no reason this is defined inside of _main()
    def do_run():

        # Phase 1 - reset state
        ########################
        clear_rotoscopers()  # what a silly name
        vram_profiling(params.approximate_vram_usage)
        reset_vram_usage()
        global CLIP_MODEL_NAMES  # We're gonna do something about these globals
        # @markdown which frame to restore from
        restore_frame = latest  # @param{type:"raw"}

        # set up seed for deterministic RNG
        if params.seed is not None:
            torch.manual_seed(params.seed)

        # Phase 2 - load and parse
        ###########################

        # load CLIP
        load_clip(params)

        cutn = params.cutouts
        if params.gradient_accumulation_steps > 1:
            try:
                assert cutn % params.gradient_accumulation_steps == 0
            except:
                logger.warning(
                    "To use GRADIENT_ACCUMULATION_STEPS > 1, "
                    "the CUTOUTS parameter must be a scalar multiple of "
                    "GRADIENT_ACCUMULATION_STEPS. I.e `STEPS/CUTS` must have no remainder."
                )
                raise
            cutn //= params.gradient_accumulation_steps
        logger.debug(cutn)

        embedder = HDMultiClipEmbedder(
            # cutn=params.cutouts,
            cutn=cutn,
            cut_pow=params.cut_pow,
            padding=params.cutout_border,
            border_mode=params.border_mode,
        )

        # load scenes

        with vram_usage_mode("Text Prompts"):
            embedder, prompts = parse_scenes(
                embedder,
                scenes=params.scenes,
                scene_prefix=params.scene_prefix,
                scene_suffix=params.scene_suffix,
            )

        # load init image

        init_image_pil, height, width = load_init_image(
            init_image_path=params.init_image,
            height=params.height,
            width=params.width,
        )

        # video source
        video_frames = None
        if params.animation_mode == "Video Source":

            video_frames, init_image_pil, height, width = load_video_source(
                video_path=params.video_path,
                pre_animation_steps=params.pre_animation_steps,
                steps_per_frame=params.steps_per_frame,
                height=params.height,
                width=params.width,
                init_image_pil=init_image_pil,
            )

        # not a fan of modifying the params object like this, but may as well be consistent for now...
        params.height, params.width = height, width

        # Phase 3 - Setup Optimization
        ###############################

        assert params.image_model in (
            "Limited Palette",
            "Unlimited Palette",
            "VQGAN",
        )
        # set up image
        if params.image_model == "Limited Palette":
            img = PixelImage(
                *format_params(
                    params,
                    "width",
                    "height",
                    "pixel_size",
                    "palette_size",
                    "palettes",
                    "gamma",
                    "hdr_weight",
                    "palette_normalization_weight",
                )
            )
            img.encode_random(random_pallet=params.random_initial_palette)
            if params.target_palette.strip() != "":
                img.set_pallet_target(
                    Image.open(fetch(params.target_palette)).convert("RGB")
                )
            else:
                img.lock_pallet(params.lock_palette)
        elif params.image_model == "Unlimited Palette":
            img = RGBImage(params.width, params.height, params.pixel_size)
            img.encode_random()
        elif params.image_model == "VQGAN":
            model_artifacts_path = Path(params.models_parent_dir) / "vqgan"
            VQGANImage.init_vqgan(params.vqgan_model, model_artifacts_path)
            img = VQGANImage(params.width, params.height, params.pixel_size)
            img.encode_random()
        else:
            logger.critical(
                "You should never see this message."
                "Please document the circumstances under which you observed this "
                "message here: https://github.com/pytti-tools/pytti-core/issues/new"
            )
            raise NotSupportedError

        #######################################

        # set up losses

        loss_augs = []

        if init_image_pil is not None:
            if not restore:
                logger.info("Encoding image...")
                # logger.debug(type(img)) # pytti.Image.PixelImage.PixelImage
                # logger.debug(type(init_image_pil)) # PIL.Image.Image
                img.encode_image(init_image_pil)
                logger.info("Encoded Image:")
                # pretty sure this assumes we're in a notebook
                display.display(img.decode_image())
            # set up init image prompt
            init_augs = ["direct_init_weight"]
            init_augs = [
                build_loss(
                    x,
                    params[x],
                    f"init image ({params.init_image})",
                    img,
                    init_image_pil,
                )
                for x in init_augs
                if params[x] not in ["", "0"]
            ]
            loss_augs.extend(init_augs)
            if params.semantic_init_weight not in ["", "0"]:
                semantic_init_prompt = parse_prompt(
                    embedder,
                    f"init image [{params.init_image}]:{params.semantic_init_weight}",
                    init_image_pil,
                )
                prompts[0].append(semantic_init_prompt)
            else:
                semantic_init_prompt = None
        else:
            init_augs, semantic_init_prompt = [], None

        # other image prompts

        loss_augs.extend(
            type(img)
            .get_preferred_loss()
            .TargetImage(p.strip(), img.image_shape, is_path=True)
            for p in params.direct_image_prompts.split("|")
            if p.strip()
        )

        # stabilization

        stabilization_augs = [
            "direct_stabilization_weight",
            "depth_stabilization_weight",
            "edge_stabilization_weight",
        ]
        stabilization_augs = [
            build_loss(x, params[x], "stabilization", img, init_image_pil)
            for x in stabilization_augs
            if params[x] not in ["", "0"]
        ]
        loss_augs.extend(stabilization_augs)

        if params.semantic_stabilization_weight not in ["0", ""]:
            last_frame_semantic = parse_prompt(
                embedder,
                f"stabilization:{params.semantic_stabilization_weight}",
                init_image_pil if init_image_pil else img.decode_image(),
            )
            last_frame_semantic.set_enabled(init_image_pil is not None)
            for scene in prompts:
                scene.append(last_frame_semantic)
        else:
            last_frame_semantic = None

        # optical flow
        if params.animation_mode == "Video Source":
            if params.flow_stabilization_weight == "":
                params.flow_stabilization_weight = "0"
            optical_flows = [
                OpticalFlowLoss.TargetImage(
                    f"optical flow stabilization (frame {-2**i}):{params.flow_stabilization_weight}",
                    img.image_shape,
                )
                for i in range(params.flow_long_term_samples + 1)
            ]
            for optical_flow in optical_flows:
                optical_flow.set_enabled(False)
            loss_augs.extend(optical_flows)
        elif params.animation_mode == "3D" and params.flow_stabilization_weight not in [
            "0",
            "",
        ]:
            optical_flows = [
                TargetFlowLoss.TargetImage(
                    f"optical flow stabilization:{params.flow_stabilization_weight}",
                    img.image_shape,
                )
            ]
            for optical_flow in optical_flows:
                optical_flow.set_enabled(False)
            loss_augs.extend(optical_flows)
        else:
            optical_flows = []
        # other loss augs
        if params.smoothing_weight != 0:
            loss_augs.append(TVLoss(weight=params.smoothing_weight))

        # Phase 4 - setup outputs
        ##########################

        # Transition as much of this as possible to hydra

        # set up filespace
        Path(f"{OUTPATH}/{params.file_namespace}").mkdir(parents=True, exist_ok=True)
        Path(f"backup/{params.file_namespace}").mkdir(parents=True, exist_ok=True)
        if restore:
            base_name = (
                params.file_namespace
                if restore_run == 0
                else f"{params.file_namespace}({restore_run})"
            )
        elif not params.allow_overwrite:
            # finds the next available base_name to save files with. Why did I do this with regex?
            _, i = get_next_file(
                f"{OUTPATH}/{params.file_namespace}",
                f"^(?P<pre>{re.escape(params.file_namespace)}\\(?)(?P<index>\\d*)(?P<post>\\)?_1\\.png)$",
                [f"{params.file_namespace}_1.png", f"{params.file_namespace}(1)_1.png"],
            )
            base_name = (
                params.file_namespace if i == 0 else f"{params.file_namespace}({i})"
            )
        else:
            base_name = params.file_namespace

        # restore
        if restore:
            if not reencode:
                if restore_frame == latest:
                    filename, restore_frame = get_last_file(
                        f"backup/{params.file_namespace}",
                        f"^(?P<pre>{re.escape(base_name)}_)(?P<index>\\d*)(?P<post>\\.bak)$",
                    )
                else:
                    filename = f"{base_name}_{restore_frame}.bak"
                logger.info("restoring from", filename)
                img.load_state_dict(
                    torch.load(f"backup/{params.file_namespace}/{filename}")
                )
            else:  # reencode
                if restore_frame == latest:
                    filename, restore_frame = get_last_file(
                        f"{OUTPATH}/{params.file_namespace}",
                        f"^(?P<pre>{re.escape(base_name)}_)(?P<index>\\d*)(?P<post>\\.png)$",
                    )
                else:
                    filename = f"{base_name}_{restore_frame}.png"
                logger.info("restoring from", filename)
                img.encode_image(
                    Image.open(f"{OUTPATH}/{params.file_namespace}/{filename}").convert(
                        "RGB"
                    )
                )
            i = restore_frame * params.save_every
        else:
            i = 0

        ## tensorboard should handle this stuff.

        # graphs
        fig, axs = None, None
        if params.show_graphs:
            fig, axs = plt.subplots(4, 1, figsize=(21, 13))
            axs = np.asarray(axs).flatten()

        # Phase 5 - setup optimizer
        ############################

        # make the main model object
        model = DirectImageGuide(
            img,
            embedder,
            lr=params.learning_rate,
            params=params,
            writer=writer,
            OUTPATH=OUTPATH,
            base_name=base_name,
            fig=fig,
            axs=axs,
            video_frames=video_frames,
            optical_flows=optical_flows,
            last_frame_semantic=last_frame_semantic,  # fml...
            semantic_init_prompt=semantic_init_prompt,
            init_augs=init_augs,
            null_update=False,  # uh... we can do better.
        )

        # Pretty sure this isn't necessary, Hydra should take care of saving
        # the run settings now
        settings_path = f"{OUTPATH}/{params.file_namespace}/{base_name}_settings.txt"
        logger.info(f"Settings saved to {settings_path}")
        save_settings(params, settings_path)

        # Run the training loop
        ########################

        # `i`: current iteration
        # `skip_X`: number of _X that have already been processed to completion (per the current iteration)
        # `last_scene`: previously processed scene/prompt (or current prompt if on first/only scene)
        skip_prompts = i // params.steps_per_scene
        skip_steps = i % params.steps_per_scene
        last_scene = prompts[0] if skip_prompts == 0 else prompts[skip_prompts - 1]
        for scene in prompts[skip_prompts:]:
            logger.info("Running prompt:", " | ".join(map(str, scene)))
            i += model.run_steps(
                params.steps_per_scene - skip_steps,
                scene,
                last_scene,
                loss_augs,
                interp_steps=params.interpolation_steps,
                i_offset=i,
                skipped_steps=skip_steps,
                gradient_accumulation_steps=params.gradient_accumulation_steps,
            )
            skip_steps = 0
            model.clear_dataframe()
            last_scene = scene

        # tensorboard summarywriter should supplant all our graph stuff
        if fig:
            del fig, axs
        ############################# DMARX
        if writer is not None:
            writer.close()
        #############################

    ## Work on getting rid of this batch mode garbage. Hydra's got this.
    try:
        gc.collect()
        torch.cuda.empty_cache()
        if batch_mode:
            if restore:
                settings_list = batch_list[restore_run:]
            else:
                settings_list = batch_list
                namespace = batch_list[0]["file_namespace"]
                subprocess.run(["mkdir", "-p", f"{OUTPATH}/{namespace}"])
                save_batch(
                    batch_list, f"{OUTPATH}/{namespace}/{namespace}_batch settings.txt"
                )
                logger.info(
                    f"Batch settings saved to {OUTPATH}/{namespace}/{namespace}_batch settings.txt"
                )
            for settings in settings_list:
                setting_string = json.dumps(settings)
                logger.debug("SETTINGS:")
                logger.debug(setting_string)
                params = load_settings(setting_string)
                if params.animation_mode == "3D":
                    init_AdaBins()
                params.allow_overwrite = False
                do_run()
                restore = False
                reencode = False
                gc.collect()
                torch.cuda.empty_cache()
        else:
            do_run()
            logger.info("Complete.")
            gc.collect()
            torch.cuda.empty_cache()
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        print_vram_usage()
        raise


if __name__ == "__main__":
    _main()
