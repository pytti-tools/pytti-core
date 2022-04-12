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

logger.info("Loading pytti...")
from pytti.Notebook import (
    change_tqdm_color,  # why though?
    get_last_file,
    get_next_file,
    load_settings,  # hydra should handle this stuff
    save_settings,
    save_batch,
    load_clip,
)

from pytti.rotoscoper import ROTOSCOPERS, get_frames
from pytti.Image import PixelImage, RGBImage, VQGANImage
from pytti.ImageGuide import DirectImageGuide
from pytti.Perceptor.Embedder import HDMultiClipEmbedder
from pytti.Perceptor.Prompt import parse_prompt

from pytti import (
    fetch,
    vram_usage_mode,
    print_vram_usage,
    reset_vram_usage,
    vram_profiling,
)
from pytti.LossAug.DepthLossClass import init_AdaBins
from pytti.LossAug.LossOrchestratorClass import LossConfigurator

from IPython import display

from collections import Counter

from pytti import (
    # format_input,
    set_t,
    print_vram_usage,
    freeze_vram_usage,
    vram_usage_mode,
)
from pytti.Image.differentiable_image import DifferentiableImage
from pytti.Image.PixelImage import PixelImage
from pytti.Notebook import tqdm, make_hbox

# from pytti.rotoscoper import update_rotoscopers
from pytti.rotoscoper import ROTOSCOPERS
from pytti.Transforms import (
    animate_2d,
    zoom_3d,
    animate_video_source,
)

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
#
# ...actually, parse_prompt is invoked in loss orchestration
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

    # literal "off" in yaml interpreted as False
    if params.animation_mode == False:
        params.animation_mode = "off"

    logger.debug(params)
    logger.debug(OmegaConf.to_container(cfg, resolve=True))
    latest = -1

    writer = None
    if params.use_tensorboard:
        writer = SummaryWriter(TB_LOGDIR)

    batch_mode = False  # @param{type:"boolean"}

    ### Move these into default.yaml
    # @markdown check `restore` to restore from a previous run
    restore = params.get("restore") or False  # @param{type:"boolean"}
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
        # clear_rotoscopers()  # what a silly name
        ROTOSCOPERS.clear_rotoscopers()
        vram_profiling(params.approximate_vram_usage)
        reset_vram_usage()
        # global CLIP_MODEL_NAMES  # we don't do anything with this...
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
                width=params.width,
                height=params.height,
                scale=params.pixel_size,  # NB: inconsistent naming
                pallet_size=params.palette_size,  # NB: inconsistent naming
                n_pallets=params.palettes,  # NB: inconsistent naming
                gamma=params.gamma,
                hdr_weight=params.hdr_weight,
                norm_weight=params.palette_normalization_weight,
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
        loss_orch = LossConfigurator(
            init_image_pil=init_image_pil,
            restore=restore,
            img=img,
            embedder=embedder,
            prompts=prompts,
            # params=params,
            ########
            # To do: group arguments into param groups
            animation_mode=params.animation_mode,
            init_image=params.init_image,
            direct_image_prompts=params.direct_image_prompts,
            semantic_init_weight=params.semantic_init_weight,
            semantic_stabilization_weight=params.semantic_stabilization_weight,
            flow_stabilization_weight=params.flow_stabilization_weight,
            flow_long_term_samples=params.flow_long_term_samples,
            smoothing_weight=params.smoothing_weight,
            ###########
            direct_init_weight=params.direct_init_weight,
            direct_stabilization_weight=params.direct_stabilization_weight,
            depth_stabilization_weight=params.depth_stabilization_weight,
            edge_stabilization_weight=params.edge_stabilization_weight,
        )

        (
            loss_augs,
            init_augs,
            stabilization_augs,
            optical_flows,
            semantic_init_prompt,
            last_frame_semantic,
            img,
        ) = loss_orch.configure_losses()

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
            # params=params,
            # writer=writer,
            # OUTPATH=OUTPATH,
            # base_name=base_name,
            # fig=fig,
            # axs=axs,
            # video_frames=video_frames,
            # # these can be passed in together as the loss orchestrator
            # optical_flows=optical_flows,
            # stabilization_augs=stabilization_augs,
            # last_frame_semantic=last_frame_semantic,  # fml...
            # semantic_init_prompt=semantic_init_prompt,
            # init_augs=init_augs,
            null_update=False,  # uh... we can do better.
        )

        ##################################################################

        update_rotoscopers = ROTOSCOPERS.update_rotoscopers

        # Update is called each step.
        def update(i, stage_i):
            # display

            #
            #
            #
            #

            # DM: I bet this could be abstracted out into a report_out() function or whatever
            if params.clear_every > 0 and i > 0 and i % params.clear_every == 0:
                display.clear_output()
            if params.display_every > 0 and i % params.display_every == 0:
                logger.debug(f"Step {i} losses:")
                if model.dataframe:
                    rec = model.dataframe[0].iloc[-1]
                    logger.debug(rec)
                    for k, v in rec.iteritems():
                        writer.add_scalar(
                            tag=f"losses/{k}", scalar_value=v, global_step=i
                        )
                # does this VRAM stuff even do anything?
                if params.approximate_vram_usage:
                    logger.debug("VRAM Usage:")
                    print_vram_usage()  # update this function to use logger
                display_width = int(img.image_shape[0] * params.display_scale)
                display_height = int(img.image_shape[1] * params.display_scale)
                if stage_i > 0 and params.show_graphs:
                    model.plot_losses(axs)
                    im = img.decode_image()
                    sidebyside = make_hbox(
                        im.resize((display_width, display_height), Image.LANCZOS), fig
                    )
                    display.display(sidebyside)
                else:
                    im = img.decode_image()
                    display.display(
                        im.resize((display_width, display_height), Image.LANCZOS)
                    )
                if params.show_palette and isinstance(img, PixelImage):
                    logger.debug("Palette:")
                    display.display(img.render_pallet())
            # save
            if i > 0 and params.save_every > 0 and i % params.save_every == 0:
                try:
                    im
                except NameError:
                    im = img.decode_image()
                n = i // params.save_every
                filename = f"{OUTPATH}/{params.file_namespace}/{base_name}_{n}.png"
                im.save(filename)

                im_np = np.array(im)
                writer.add_image(
                    tag="pytti output",
                    # img_tensor=filename, # thought this would work?
                    img_tensor=im_np,
                    global_step=i,
                    dataformats="HWC",  # this was the key
                )

                if params.backups > 0:
                    filename = f"backup/{params.file_namespace}/{base_name}_{n}.bak"
                    torch.save(img.state_dict(), filename)
                    if n > params.backups:

                        # YOOOOOOO let's not start shell processes unnecessarily
                        # and then execute commands using string interpolation.
                        # Replace this with a pythonic folder removal, then see
                        # if we can't deprecate the folder removal entirely. What
                        # is the purpose of "backups" here? Just use the frames that
                        # are being written to disk.
                        subprocess.run(
                            [
                                "rm",
                                f"backup/{params.file_namespace}/{base_name}_{n-params.backups}.bak",
                            ]
                        )

            ### DM: report_out() would probably end down here

            #
            #
            #
            #

            # animate
            ################
            t = (i - params.pre_animation_steps) / (
                params.steps_per_frame * params.frames_per_second
            )
            set_t(t)
            if i >= params.pre_animation_steps:
                if (i - params.pre_animation_steps) % params.steps_per_frame == 0:
                    logger.debug(f"Time: {t:.4f} seconds")
                    update_rotoscopers(
                        ((i - params.pre_animation_steps) // params.steps_per_frame + 1)
                        * params.frame_stride
                    )
                    if params.reset_lr_each_frame:
                        model.set_optim(None)
                    if params.animation_mode == "2D":
                        tx, ty = parametric_eval(params.translate_x), parametric_eval(
                            params.translate_y
                        )
                        theta = parametric_eval(params.rotate_2d)
                        zx, zy = parametric_eval(params.zoom_x_2d), parametric_eval(
                            params.zoom_y_2d
                        )
                        next_step_pil = zoom_2d(
                            img,
                            (tx, ty),
                            (zx, zy),
                            theta,
                            border_mode=params.infill_mode,
                            sampling_mode=params.sampling_mode,
                        )
                        ################
                        for k, v in {
                            "tx": tx,
                            "ty": ty,
                            "theta": theta,
                            "zx": zx,
                            "zy": zy,
                            "t": t,
                        }.items():

                            writer.add_scalar(
                                tag=f"translation_2d/{k}", scalar_value=v, global_step=i
                            )

                        ###########################
                    elif params.animation_mode == "3D":
                        try:
                            im
                        except NameError:
                            im = img.decode_image()
                        with vram_usage_mode("Optical Flow Loss"):
                            flow, next_step_pil = zoom_3d(
                                img,
                                (
                                    params.translate_x,
                                    params.translate_y,
                                    params.translate_z_3d,
                                ),
                                params.rotate_3d,
                                params.field_of_view,
                                params.near_plane,
                                params.far_plane,
                                border_mode=params.infill_mode,
                                sampling_mode=params.sampling_mode,
                                stabilize=params.lock_camera,
                            )
                            freeze_vram_usage()

                        for optical_flow in optical_flows:
                            optical_flow.set_last_step(im)
                            optical_flow.set_target_flow(flow)
                            optical_flow.set_enabled(True)
                    elif params.animation_mode == "Video Source":
                        frame_n = min(
                            (i - params.pre_animation_steps)
                            * params.frame_stride
                            // params.steps_per_frame,
                            len(video_frames) - 1,
                        )
                        next_frame_n = min(
                            frame_n + params.frame_stride, len(video_frames) - 1
                        )
                        next_step_pil = (
                            Image.fromarray(video_frames.get_data(next_frame_n))
                            .convert("RGB")
                            .resize(img.image_shape, Image.LANCZOS)
                        )
                        for j, optical_flow in enumerate(optical_flows):
                            old_frame_n = frame_n - (2 ** j - 1) * params.frame_stride
                            save_n = i // params.save_every - (2 ** j - 1)
                            if old_frame_n < 0 or save_n < 1:
                                break
                            current_step_pil = (
                                Image.fromarray(video_frames.get_data(old_frame_n))
                                .convert("RGB")
                                .resize(img.image_shape, Image.LANCZOS)
                            )
                            filename = f"backup/{params.file_namespace}/{base_name}_{save_n}.bak"
                            filename = None if j == 0 else filename
                            flow_im, mask_tensor = optical_flow.set_flow(
                                current_step_pil,
                                next_step_pil,
                                img,
                                filename,
                                params.infill_mode,
                                params.sampling_mode,
                            )
                            optical_flow.set_enabled(True)
                            # first flow is previous frame
                            if j == 0:
                                mask_accum = mask_tensor.detach()
                                valid = mask_tensor.mean()
                                logger.debug("valid pixels:", valid)
                                if params.reencode_each_frame or valid < 0.03:
                                    if isinstance(img, PixelImage) and valid >= 0.03:
                                        img.lock_pallet()
                                        img.encode_image(
                                            next_step_pil, smart_encode=False
                                        )
                                        img.lock_pallet(params.lock_palette)
                                    else:
                                        img.encode_image(next_step_pil)
                                    reencoded = True
                                else:
                                    reencoded = False
                            else:
                                with torch.no_grad():
                                    optical_flow.set_mask(
                                        (mask_tensor - mask_accum).clamp(0, 1)
                                    )
                                    mask_accum.add_(mask_tensor)
                    if params.animation_mode != "off":
                        for aug in stabilization_augs:
                            aug.set_comp(next_step_pil)
                            aug.set_enabled(True)
                        if last_frame_semantic is not None:
                            last_frame_semantic.set_image(embedder, next_step_pil)
                            last_frame_semantic.set_enabled(True)
                        for aug in init_augs:
                            aug.set_enabled(False)
                        if semantic_init_prompt is not None:
                            semantic_init_prompt.set_enabled(False)

        ###############################################################
        ###

        # Wait.... we literally instantiated the model just before
        # defining update here.
        # I bet all of this can go in the DirectImageGuide class and then
        # we can just instantiate that class with the config object.

        model.update = update

        ##################################################################

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
