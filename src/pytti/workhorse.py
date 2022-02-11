"""
This is the rendering logic that used to live in the notebook. 
It's sort of a mess in here. I'm working on it. 
Thank you for your patience.-- The Management
"""
from pathlib import Path
import os
from os.path import exists as path_exists
import sys
import gc, glob, subprocess, warnings, re, math, json

import hydra
from IPython import display
from omegaconf import OmegaConf, DictConfig
from loguru import logger
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional as TF

# Libraries to deprecate
from bunch import Bunch
import matplotlib.pyplot as plt
import seaborn as sns


logger.info("Loading pytti...")

from pytti.Notebook import (
    is_notebook,
    change_tqdm_color, # who cares tho.
    get_last_file,
    get_next_file,
    make_hbox,
    load_settings,
    write_settings,
    save_settings,
    save_batch,
    CLIP_MODEL_NAMES,
    load_clip,
    get_frames,
    build_loss,
    format_params,
    rotoscopers,
    clear_rotoscopers,
    update_rotoscopers,
    Rotoscoper,
)

from pytti import Perceptor
from pytti.Image import PixelImage, RGBImage, VQGANImage
from pytti.ImageGuide import DirectImageGuide
from pytti.Perceptor.Embedder import HDMultiClipEmbedder
from pytti.Perceptor.Prompt import parse_prompt
from pytti.LossAug import TVLoss, HSVLoss, OpticalFlowLoss, TargetFlowLoss
from pytti.Transforms import zoom_2d, zoom_3d, apply_flow
from pytti import (
    DEVICE,
    fetch,
    parametric_eval,
    set_t,
    vram_usage_mode,
    print_vram_usage,
    reset_vram_usage,
    freeze_vram_usage,
    vram_profiling,
)
from pytti.LossAug.DepthLoss import init_AdaBins

logger.info("pytti loaded.")

sns.set()
plt.style.use("bmh")
pd.options.display.max_columns = None
pd.options.display.width = 175


TB_LOGDIR = "logs"  # to do: make this more easily configurable
writer = SummaryWriter(TB_LOGDIR)
OUTPATH = f"{os.getcwd()}/images_out/" # to do: transition to relying fully on hydra outputs/

# TO DO: populate this from... params? globals?
# what is this even for? Can I eliminate this variable?
drive_mounted = False

# This should probably be configured in a hydra config file 
local_path = Path(os.getcwd()) / "config"
logger.debug(local_path)

@hydra.main(config_path=local_path, config_name="default")
def _main(cfg: DictConfig):
    default_params = OmegaConf.to_container(cfg, resolve=True)
    logger.debug(os.getcwd())
    logger.debug(default_params)

    # this is probably some version checking hack I should remove
    latest = -1
    
    batch_mode = False
    restore = False
    reencode = False

    restore_run = latest # I think this is only useful inside the notebook

    # This is why notebooks are bad. Right here.
    if batch_mode:
        try:
            batch_list 
        except NameError:
            raise RuntimeError(
                "ERROR: no batch settings. Please run 'batch settings' cell at the bottom of the page to use batch mode."
            )
    else:
        try:
            params = default_params
            params = Bunch(
                params
            )  # maybe replace Bunch () with  OmegaConf.create(params)?
        except NameError:
            raise RuntimeError(
                "ERROR: no parameters. Please run parameters (step 2.1)."
            )

    # This will never evaluate to true right now because `latest=-1` above
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
        global CLIP_MODEL_NAMES # We're gonna do something about these globals

        restore_frame = latest

        # set up seed for deterministic RNG
        if params.seed is not None:
            torch.manual_seed(params.seed)

        # Phase 2 - load and parse
        ###########################

        # load CLIP
        load_clip(params)
        embedder = HDMultiClipEmbedder(
            cutn=params.cutouts,
            cut_pow=params.cut_pow,
            padding=params.cutout_border,
            border_mode=params.border_mode,
        )

        # load scenes
        with vram_usage_mode("Text Prompts"):
            logger.info("Loading prompts...")
            prompts = [
                [
                    parse_prompt(embedder, p.strip())
                    for p in (params.scene_prefix + stage + params.scene_suffix)
                    .strip()
                    .split("|")
                    if p.strip()
                ]
                for stage in params.scenes.split("||")
                if stage
            ]
            logger.info("Prompts loaded.")

        # load init image
        if params.init_image != "":
            init_image_pil = Image.open(fetch(params.init_image)).convert("RGB")
            init_size = init_image_pil.size
            # automatic aspect ratio matching
            if params.width == -1:
                params.width = int(params.height * init_size[0] / init_size[1])
            if params.height == -1:
                params.height = int(params.width * init_size[1] / init_size[0])
        else:
            init_image_pil = None

        # video source
        if params.animation_mode == "Video Source":
            logger.info(f"loading {params.video_path}...")
            video_frames = get_frames(params.video_path)
            params.pre_animation_steps = max(
                params.steps_per_frame, params.pre_animation_steps
            )
            if init_image_pil is None:
                init_image_pil = Image.fromarray(video_frames.get_data(0)).convert(
                    "RGB"
                )
                # enhancer = ImageEnhance.Contrast(init_image_pil)
                # init_image_pil = enhancer.enhance(2)
                init_size = init_image_pil.size
                if params.width == -1:
                    params.width = int(params.height * init_size[0] / init_size[1])
                if params.height == -1:
                    params.height = int(params.width * init_size[1] / init_size[0])


        # Phase 3 - Setup Optimization
        ###############################

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
            VQGANImage.init_vqgan(params.vqgan_model)
            img = VQGANImage(params.width, params.height, params.pixel_size)
            img.encode_random()

        loss_augs = []

        if init_image_pil is not None:
            if not restore:
                logger.info("Encoding image...")
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

        # graphs
        if params.show_graphs:
            fig, axs = plt.subplots(4, 1, figsize=(21, 13))
            axs = np.asarray(axs).flatten()
            # fig.facecolor = (0,0,0)
        else:
            fig, axs = None, None


        # Phase 5 - setup optimizer 
        ############################

        # make the main model object
        model = DirectImageGuide(img, embedder, lr=params.learning_rate)

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


        # Pretty sure this isn't necessary, Hydra should take care of saving 
        # the run settings now
        logger.info(
            f"Settings saved to {OUTPATH}/{params.file_namespace}/{base_name}_settings.txt"
        )
        save_settings(
            params, f"{OUTPATH}/{params.file_namespace}/{base_name}_settings.txt"
        )

        # Run the training loop
        ########################

        # what are these skip variables doing?
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
            )
            skip_steps = 0
            model.clear_dataframe()
            last_scene = scene

        # tensorboard summarywriter should supplnat all our graph stuff
        if fig:
            del fig, axs
        ############################# DMARX
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
