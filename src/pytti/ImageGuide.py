import math
import subprocess

import einops as eo
from loguru import logger
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import savgol_filter
import torch
from torch import optim, nn

from collections import Counter

from pytti import (
    format_input,
    set_t,
    print_vram_usage,
    freeze_vram_usage,
    vram_usage_mode,
)
from pytti.AudioParse import SpectralAudioParser
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


# deprecate this
from labellines import labelLines
from IPython import display


def unpack_dict(D, n=2):
    """
    Given a dictionary D and a number n, return a tuple of n dictionaries,
    each containing the same keys as D and values corresponding to those
    values of D at the corresponding index

    :param D: a dictionary
    :param n: number of samples to draw, defaults to 2 (optional)
    :return: A tuple of dictionaries.
    """
    ds = [{k: V[i] for k, V in D.items()} for i in range(n)]
    return tuple(ds)


# this only gets used in the plot_losses method below.
# deprecate (tensorboard)
def smooth_dataframe(df, window_size):
    """applies a moving average filter to the columns of df"""
    smoothed_df = pd.DataFrame().reindex_like(df)
    for key in df.columns:
        smoothed_df[key] = savgol_filter(df[key], window_size, 2, mode="nearest")
    return smoothed_df


class DirectImageGuide:
    """
    Image guide that uses an optimizer and torch autograd to optimize an image representation
    Based on the BigGan+CLIP algorithm by advadnoun (https://twitter.com/advadnoun)
    image_rep: (DifferentiableImage) image representation
    embedder: (Module)               image embedder
    optimizer: (Class)               optimizer class to use. Defaults to Adam
    all other arguments are passed as kwargs to the optimizer.
    """

    def __init__(
        self,
        image_rep: DifferentiableImage,
        embedder: nn.Module,
        optimizer: optim.Optimizer = None,
        lr: float = None,
        # null_update=True,
        params=None,
        writer=None,
        fig=None,
        axs=None,
        base_name=None,
        OUTPATH=None,  # <<<<<<<<<<<<<<
        #####################
        video_frames=None,  # # only need this to pass to animate_video_source
        optical_flows=None,
        stabilization_augs=None,
        last_frame_semantic=None,
        semantic_init_prompt=None,
        init_augs=None,
        **optimizer_params,
        # pretty sure passing in optimizer_params isn't being used anywhere
        # We pass in the optimizer object itself anyway... why not just give it
        # initialize it with `**optimizer_params`` before passing it to this?
    ):
        self.image_rep = image_rep
        self.embedder = embedder
        if lr is None:
            lr = image_rep.lr
        optimizer_params["lr"] = lr
        self.optimizer_params = optimizer_params
        if optimizer is None:
            self.optimizer = optim.Adam(image_rep.parameters(), **optimizer_params)
        else:
            self.optimizer = optimizer
        self.dataframe = []

        if params.input_audio:
            self.audio_parser = SpectralAudioParser(params.input_audio, params.offset, params.frames_per_second, params.filters)
        else:
            self.audio_parser = None

        # self.null_update = null_update
        self.params = params
        self.writer = writer
        self.OUTPATH = OUTPATH
        self.base_name = base_name
        self.fig = fig
        self.axs = axs
        self.video_frames = video_frames
        self.optical_flows = optical_flows
        # if stabilization_augs is None:
        #    stabilization_augs = []
        self.stabilization_augs = stabilization_augs
        self.last_frame_semantic = last_frame_semantic
        self.semantic_init_prompt = semantic_init_prompt
        # if init_augs is None:
        #    init_augs = []
        self.init_augs = init_augs

    def run_steps(
        self,
        n_steps,
        prompts,
        interp_prompts,
        loss_augs,
        stop=-math.inf,
        interp_steps=0,
        i_offset=0,
        skipped_steps=0,
        gradient_accumulation_steps: int = 1,
    ):
        """
        runs the optimizer
        prompts: (ClipPrompt list) list of prompts
        n_steps: (positive integer) steps to run
        returns: the number of steps run
        """
        for i in tqdm(range(n_steps)):
            # not a huge fan of this.
            # currently need it for PixelImage.encode_image
            # TO DO: all that stuff we just moved around:
            #        let's attach it to a "Renderer" class,
            #        and here we can check if the DirectImageGuide was
            #        initialized with a renderer or not, and call self.renderer.update()
            #        if appropriate
            # if not self.null_update:
            #    self.update(i + i_offset, i + skipped_steps)
            self.update(
                model=self,
                img=self.image_rep,
                i=i + i_offset,
                stage_i=i + skipped_steps,
                params=self.params,
                writer=self.writer,
                fig=self.fig,
                axs=self.axs,
                base_name=self.base_name,
                optical_flows=self.optical_flows,
                video_frames=self.video_frames,
                stabilization_augs=self.stabilization_augs,
                last_frame_semantic=self.last_frame_semantic,
                embedder=self.embedder,
                init_augs=self.init_augs,
                semantic_init_prompt=self.semantic_init_prompt,
            )
            losses = self.train(
                i + skipped_steps,
                prompts,
                interp_prompts,
                loss_augs,
                interp_steps=interp_steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            if losses["TOTAL"] <= stop:
                break
        return i + 1

    def set_optim(self, opt=None):
        if opt is not None:
            self.optimizer = opt
        else:
            self.optimizer = optim.Adam(
                self.image_rep.parameters(), **self.optimizer_params
            )

    def clear_dataframe(self):
        """
        The .dataframe attribute is just a list of pd.DataFrames that
        are tracking losses for the current scene. I wanna say one
        for each prompt. To do: flush all that out and let tensorboard handle it.
        """
        self.dataframe = []

    # deprecate (tensorboard)
    def plot_losses(self, axs):
        def plot_dataframe(df, ax, legend=False):
            keys = list(df)
            keys.sort(reverse=True, key=lambda k: df[k].iloc[-1])
            ax.clear()
            df[keys].plot(ax=ax, legend=legend)
            if legend:
                ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            ax.tick_params(
                labelbottom=True,
                labeltop=False,
                labelleft=True,
                labelright=False,
                bottom=True,
                top=False,
                left=True,
                right=False,
            )
            last_x = df.last_valid_index()
            lines = ax.get_lines()

            colors = [l.get_color() for l in lines]
            labels = [l.get_label() for l in lines]
            ax.relim()
            ax.autoscale_view()

            labelLines(ax.get_lines(), align=False)
            return dict(zip(labels, colors))

        dfs = self.dataframe[:]
        if dfs != []:
            dfs[0] = smooth_dataframe(dfs[0], 17)
        for i, (df, ax) in enumerate(zip(dfs, axs)):
            if len(df.index) < 2:
                return False
            # m = df.apply(lambda col: col.first_valid_index())
            # print(m)
            # print(df.lookup(m, m.index))
            # rel_loss = (df-df.lookup(m, m.index))
            if not df.empty:
                plot_dataframe(df, ax, legend=i == 0)
            ax.set_ylabel("Loss")
            ax.set_xlabel("Step")
        return True

    def train(
        self,
        i,
        prompts,
        interp_prompts,
        loss_augs,
        interp_steps=0,
        save_loss=True,
        gradient_accumulation_steps: int = 1,
    ):
        """
        steps the optimizer
        promts: (ClipPrompt list) list of prompts
        """
        self.optimizer.zero_grad()
        z = self.image_rep.decode_training_tensor()
        # logger.debug(z.shape)  # [1, 3, height, width]
        losses = []

        aug_losses = {
            aug: aug(format_input(z, self.image_rep, aug), self.image_rep)
            for aug in loss_augs
        }

        image_augs = self.image_rep.image_loss()
        image_losses = {aug: aug(self.image_rep) for aug in image_augs}

        # losses_accumulator, losses_raw_accumulator = Counter(), Counter()
        losses, losses_raw = [], []  # just... don't care
        total_loss = 0
        if self.embedder is not None:
            for mb_i in range(gradient_accumulation_steps):
                # logger.debug(mb_i)
                image_embeds, offsets, sizes = self.embedder(self.image_rep, input=z)

                t = 1
                interp_losses = [0]
                if i < interp_steps:
                    t = i / interp_steps
                    interp_losses = [
                        prompt(
                            format_input(image_embeds, self.embedder, prompt),
                            format_input(offsets, self.embedder, prompt),
                            format_input(sizes, self.embedder, prompt),
                        )[0]
                        * (1 - t)
                        for prompt in interp_prompts
                    ]

                prompt_losses = {
                    prompt: prompt(
                        format_input(image_embeds, self.embedder, prompt),
                        format_input(offsets, self.embedder, prompt),
                        format_input(sizes, self.embedder, prompt),
                    )
                    for prompt in prompts
                }

                losses, losses_raw = zip(
                    *map(unpack_dict, [prompt_losses, aug_losses, image_losses])
                    # *map(unpack_dict, [prompt_losses])
                )
                # logger.debug(losses)
                losses = list(losses)
                # logger.debug(losses)
                # losses = Counter(losses)
                # logger.debug(losses)
                losses_raw = list(losses_raw)
                # losses_raw = Counter(losses_raw)
                # losses_accumulator += losses
                # losses_raw_accumulator += losses_raw

                for v in prompt_losses.values():
                    v[0].mul_(t)

                total_loss_mb = sum(map(lambda x: sum(x.values()), losses)) + sum(
                    interp_losses
                )

                total_loss_mb /= gradient_accumulation_steps

                # total_loss_mb.backward()
                total_loss_mb.backward(retain_graph=True)
                # total_loss += total_loss_mb # this is causing it to break
                # total_loss = total_loss_mb

        # losses = [{k:v} for k,v in losses_accumulator.items()]
        # losses_raw = [{k:v} for k,v in losses_raw_accumulator.items()]
        losses_raw.append({"TOTAL": total_loss})  # this needs to be fixed
        self.optimizer.step()
        self.image_rep.update()
        self.optimizer.zero_grad()
        # if t != 0:
        #  for v in prompt_losses.values():
        #    v[0].div_(t)
        if save_loss:
            if not self.dataframe:
                self.dataframe = [
                    pd.DataFrame({str(k): float(v) for k, v in loss.items()}, index=[i])
                    for loss in losses_raw
                ]
                for df in self.dataframe:
                    df.index.name = "Step"
            else:
                for j, (df, loss) in enumerate(zip(self.dataframe, losses_raw)):
                    frames = [df] + [
                        pd.DataFrame(
                            {str(k): float(v) for k, v in loss.items()}, index=[i]
                        )
                    ]
                    self.dataframe[j] = pd.concat(frames, ignore_index=False)
                    self.dataframe[j].name = "Step"

        return {"TOTAL": float(total_loss)}

    def update(self, model, img, i, stage_i, *args, **kwargs):
        """
        update hook called ever step
        """
<<<<<<< HEAD
        pass
=======
        # logger.debug("model.update called")

        # ... I have regrets.
        params = self.params
        writer = self.writer
        OUTPATH = self.OUTPATH
        base_name = self.base_name
        fig = self.fig
        axs = self.axs
        video_frames = self.video_frames
        optical_flows = self.optical_flows
        stabilization_augs = self.stabilization_augs
        last_frame_semantic = self.last_frame_semantic
        semantic_init_prompt = self.semantic_init_prompt
        init_augs = self.init_augs

        model = self
        img = self.image_rep
        embedder = self.embedder

        model.report_out(
            i=i,
            stage_i=stage_i,
            # model=model,
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

        model.save_out(
            i=i,
            # img=img,
            writer=writer,
            OUTPATH=OUTPATH,
            base_name=base_name,
            save_every=params.save_every,
            file_namespace=params.file_namespace,
            backups=params.backups,
        )

        # animate
        ################
        ## TO DO: attach T as a class attribute
        t = (i - params.pre_animation_steps) / (
            params.steps_per_frame * params.frames_per_second
        )
        if self.audio_parser is None:
            set_t(t, 0, 0, 0)
        # set_t(t)  # this won't need to be a thing with `t`` attached to the class
        if i >= params.pre_animation_steps:
            # next_step_pil = None
            if (i - params.pre_animation_steps) % params.steps_per_frame == 0:
                if self.audio_parser is not None:
                    band_dict = self.audio_parser.get_params(t)
                    logger.debug(f"Time: {t:.4f} seconds, audio params: lo: {lo:.4f}, mid: {mid:.4f}, hi: {hi:.4f}")
                    set_t(t, band_dict)
                else:
                    logger.debug(f"Time: {t:.4f} seconds")
                # update_rotoscopers(
                ROTOSCOPERS.update_rotoscopers(
                    ((i - params.pre_animation_steps) // params.steps_per_frame + 1)
                    * params.frame_stride
                )
                if params.reset_lr_each_frame:
                    model.set_optim(None)

                if params.animation_mode == "2D":

                    next_step_pil = animate_2d(
                        translate_y=params.translate_y,
                        translate_x=params.translate_x,
                        rotate_2d=params.rotate_2d,
                        zoom_x_2d=params.zoom_x_2d,
                        zoom_y_2d=params.zoom_y_2d,
                        infill_mode=params.infill_mode,
                        sampling_mode=params.sampling_mode,
                        writer=writer,
                        i=i,
                        img=img,
                        t=t,  # just here for logging
                    )

                    ###########################
                elif params.animation_mode == "3D":
                    try:
                        im
                    except NameError:
                        im = img.decode_image()
                    with vram_usage_mode("Optical Flow Loss"):
                        # zoom_3d -> rename to animate_3d or transform_3d
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
                    )

                if params.animation_mode != "off":
                    try:
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
                    except UnboundLocalError:
                        logger.critical(
                            "\n\n-----< PYTTI-TOOLS > ------"
                            "If you are seeing this error, it might mean "
                            "you are using an option that expects you have "
                            "provided an init_image or video_file.\n\nIf you "
                            "think you are seeing this message in error, please "
                            "file an issue here: "
                            "https://github.com/pytti-tools/pytti-core/issues/new"
                            "-----< PYTTI-TOOLS > ------\n\n"
                        )
                        raise
>>>>>>> 57553a3 (feat: initial rough audio parsing logic)
