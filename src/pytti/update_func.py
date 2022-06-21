from pathlib import Path
import os
import subprocess

from PIL import Image
import numpy as np
import torch
from IPython import display
from loguru import logger

from pytti import (
    parametric_eval,
    set_t,
    vram_usage_mode,
    print_vram_usage,
    freeze_vram_usage,
)

from pytti.image_models.pixel import PixelImage

from pytti.Transforms import (
    animate_2d,
    zoom_2d,
    zoom_3d,
    animate_video_source,
)

from pytti.rotoscoper import (
    # clear_rotoscopers,
    update_rotoscopers,
)

# OUTPATH = f"{os.getcwd()}/images_out/"
OUTPATH = f"{os.getcwd()}/images_out"


# Update is called each step.
def update(
    model,
    img,
    i,
    stage_i,
    params=None,
    writer=None,
    fig=None,
    axs=None,
    base_name=None,
    optical_flows=None,
    video_frames=None,
    stabilization_augs=None,
    last_frame_semantic=None,
    embedder=None,
    init_augs=None,
    semantic_init_prompt=None,
):
    def report_out(
        img,
        i,
        stage_i,
        model,
        writer,
        fig,  # default to None...
        axs,  # default to None...
        clear_every,
        display_every,
        approximate_vram_usage,
        display_scale,
        show_graphs,
        show_palette,
    ):

        logger.debug(f"Step {i} losses:")
        if model.dataframe:
            rec = model.dataframe[0].iloc[-1]
            logger.debug(rec)
            if writer is not None:
                for k, v in rec.iteritems():
                    writer.add_scalar(tag=f"losses/{k}", scalar_value=v, global_step=i)

        # does this VRAM stuff even do anything?
        if approximate_vram_usage:
            logger.debug("VRAM Usage:")
            print_vram_usage()  # update this function to use logger
        # update this stuff to use/rely on tensorboard
        display_width = int(img.image_shape[0] * display_scale)
        display_height = int(img.image_shape[1] * display_scale)
        if stage_i > 0 and show_graphs:
            model.plot_losses(axs)
            im = img.decode_image()
            sidebyside = make_hbox(
                im.resize((display_width, display_height), Image.LANCZOS),
                fig,
            )
            display.display(sidebyside)
        else:
            im = img.decode_image()
            display.display(im.resize((display_width, display_height), Image.LANCZOS))
        if show_palette and isinstance(img, PixelImage):
            logger.debug("Palette:")
            display.display(img.render_pallet())

    def save_out(
        i,
        img,
        writer,
        OUTPATH,
        base_name,
        save_every,
        file_namespace,
        backups,
    ):
        try:
            im
        except NameError:
            im = img.decode_image()
        n = j // save_every
        Path(f"{OUTPATH}/{file_namespace}").mkdir(
            parents=True,
            exist_ok=True,
        )
        filename = f"{OUTPATH}/{file_namespace}/{base_name}_{n}.png"
        im.save(filename)

        if writer is not None:
            im_np = np.array(im)
            writer.add_image(
                tag="pytti output",
                # img_tensor=filename, # thought this would work?
                img_tensor=im_np,
                global_step=i,
                dataformats="HWC",  # this was the key
            )

        if backups > 0:
            filename = f"backup/{file_namespace}/{base_name}_{n}.bak"
            torch.save(img.state_dict(), filename)
            if n > backups:
                fname = f"{base_name}_{n-backups}.bak"
                fpath = Path("backup") / file_namespace / fname
                # delete the file. if file not found, nothing happens.
                if fpath.exists():
                    fpath.unlink()
                #fpath.unlink(
                #    missing_ok=True
                #)

    j = i + 1

    if (params.clear_every > 0) and (i > 0) and (j % params.clear_every == 0):
        display.clear_output()

    if (params.display_every > 0) and (j % params.display_every == 0):
        report_out(
            img=img,
            i=i,
            stage_i=stage_i,
            model=model,
            writer=writer,
            fig=fig,  # default to None...
            axs=axs,  # default to None...
            clear_every=params.clear_every,
            display_every=params.display_every,
            approximate_vram_usage=params.approximate_vram_usage,
            display_scale=params.display_scale,
            show_graphs=params.show_graphs,
            show_palette=params.show_palette,
        )

    if (i > 0) and (params.save_every > 0) and (j % params.save_every == 0):
        save_out(
            i=i,
            img=img,
            writer=writer,
            OUTPATH=OUTPATH,
            base_name=base_name,
            save_every=params.save_every,
            file_namespace=params.file_namespace,
            backups=params.backups,
        )

    # animate
    ################
    t = (i - params.pre_animation_steps) / (
        params.steps_per_frame * params.frames_per_second
    )
    set_t(t, {})
    if i >= params.pre_animation_steps:
        if (i - params.pre_animation_steps) % params.steps_per_frame == 0:
            logger.debug(f"Time: {t:.4f} seconds")

            # Audio Reactivity ############
            if model.audio_parser is None:
                set_t(t, {})
            # set_t(t)  # this won't need to be a thing with `t`` attached to the class
            if i >= params.pre_animation_steps:
                # next_step_pil = None
                if (i - params.pre_animation_steps) % params.steps_per_frame == 0:
                    if model.audio_parser is not None:
                        band_dict = model.audio_parser.get_params(t)
                        logger.debug(
                            f"Time: {t:.4f} seconds, audio params: {band_dict}"
                        )
                        set_t(t, band_dict)
                    else:
                        logger.debug(f"Time: {t:.4f} seconds")
            ###############################

            update_rotoscopers(
                ((i - params.pre_animation_steps) // params.steps_per_frame + 1)
                * params.frame_stride
            )
            if params.reset_lr_each_frame:
                model.set_optim(None)
            if params.animation_mode == "2D":
                logger.debug(params.translate_x)
                logger.debug(params.translate_y)
                logger.debug(params.rotate_2d)
                logger.debug(params.zoom_x_2d)
                logger.debug(params.zoom_y_2d)
                from pytti.eval_tools import global_bands

                logger.debug(global_bands)
                tx, ty = parametric_eval(params.translate_x), parametric_eval(
                    params.translate_y
                )
                theta = parametric_eval(params.rotate_2d)
                zx, zy = parametric_eval(params.zoom_x_2d), parametric_eval(
                    params.zoom_y_2d
                )
                logger.debug(f"Translate: {tx}, {ty}")
                logger.debug(f"Rotate: {theta}")
                logger.debug(f"Zoom: {zx}, {zy}")

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
                    if writer is not None:
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
                        # device=None,
                        device=params.device,
                    )
                    freeze_vram_usage()

                for optical_flow in optical_flows:
                    optical_flow.set_last_step(im)
                    optical_flow.set_target_flow(flow)
                    optical_flow.set_enabled(True)
            elif params.animation_mode == "Video Source":
                flow_im, next_step_pil = animate_video_source(
                    i=i,
                    img=img,
                    video_frames=video_frames,
                    optical_flows=optical_flows,
                    base_name=base_name,
                    pre_animation_steps=params.pre_animation_steps,
                    frame_stride=params.frame_stride,
                    steps_per_frame=params.steps_per_frame,
                    file_namespace=params.file_namespace,
                    reencode_each_frame=params.reencode_each_frame,
                    lock_palette=params.lock_palette,
                    save_every=params.save_every,
                    infill_mode=params.infill_mode,
                    sampling_mode=params.sampling_mode,
                    device=params.device,
                )

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
