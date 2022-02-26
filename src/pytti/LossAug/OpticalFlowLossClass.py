# ton of unused imports in here...
# from pytti.LossAug import MSELoss, LatentLoss
from pytti.LossAug.MSELossClass import MSELoss
from pytti.LossAug.LatentLossClass import LatentLoss
import sys, os, gc
import argparse
import os
import cv2
import glob
import math, copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import imageio
import matplotlib.pyplot as plt

# from pytti.Notebook import Rotoscoper
from pytti.rotoscoper import Rotoscoper
from torchvision.transforms import functional as TF

import gma
from gma.core.network import RAFTGMA
from gma.core.utils import flow_viz
from gma.core.utils.utils import InputPadder

from pytti.Transforms import apply_flow
from pytti import fetch, to_pil, DEVICE, vram_usage_mode
from pytti.Image.RGBImage import RGBImage

GMA = None

try:
    from importlib.resources import files as ir_files
except:
    # Patch for colab using old importlib version
    import pkg_resources

    def ir_files(module):
        if pkg_resources.resource_exists(
            gma.__name__, "data/checkpoints/gma-sintel.pth"
        ):
            return pkg_resources.resource_stringsource_filename(
                gma.__name__, "data/checkpoints/gma-sintel.pth"
            )
        else:
            raise ValueError("Unable to locate GMA checkpoint.")


def init_GMA(checkpoint_path=None):
    if checkpoint_path is None:
        root = ir_files(gma)
        checkpoint_path = str(next(root.glob("**/*sintel.pth")))
    global GMA
    if GMA is None:
        with vram_usage_mode("GMA"):
            # migrate this to a hydra initialize/compose operation
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "--model", help="restore checkpoint", default=checkpoint_path
            )
            parser.add_argument("--model_name", help="define model name", default="GMA")
            parser.add_argument("--path", help="dataset for evaluation")
            parser.add_argument(
                "--num_heads",
                default=1,
                type=int,
                help="number of heads in attention and aggregation",
            )
            parser.add_argument(
                "--position_only",
                default=False,
                action="store_true",
                help="only use position-wise attention",
            )
            parser.add_argument(
                "--position_and_content",
                default=False,
                action="store_true",
                help="use position and content-wise attention",
            )
            parser.add_argument(
                "--mixed_precision", action="store_true", help="use mixed precision"
            )
            args = parser.parse_args([])
            GMA = torch.nn.DataParallel(RAFTGMA(args))
            GMA.load_state_dict(torch.load(checkpoint_path))
            GMA.to(DEVICE)
            GMA.eval()


def sample(tensor, uv, device=DEVICE):
    height, width = tensor.shape[-2:]
    max_pos = torch.tensor([width - 1, height - 1], device=device).view(2, 1, 1)
    grid = uv.div(max_pos / 2).sub(1).movedim(0, -1).unsqueeze(0)
    return F.grid_sample(tensor.unsqueeze(0), grid, align_corners=True).squeeze(0)


class TargetFlowLoss(MSELoss):
    def __init__(
        self,
        comp,
        weight=0.5,
        stop=-math.inf,
        name="direct target loss",
        image_shape=None,
    ):
        super().__init__(comp, weight, stop, name, image_shape)
        with torch.no_grad():
            self.register_buffer("last_step", comp.clone())
            self.mag = 1

    @torch.no_grad()
    def set_target_flow(self, flow, device=DEVICE):
        """
        Set the target flow to the given flow field

        :param flow: the flow to be set as the target flow
        :param device: the device to run the training on
        """
        self.comp.set_(
            flow.movedim(-1, 1).to(device, memory_format=torch.channels_last)
        )
        self.mag = float(torch.linalg.norm(self.comp, dim=1).square().mean())

    @torch.no_grad()
    def set_last_step(self, last_step_pil, device=DEVICE):
        """
        It sets the last_step class attribute to the provided last_step_pil image.

        :param last_step_pil: The last step of the sequence
        :param device: The device to use for training
        """
        last_step = (
            TF.to_tensor(last_step_pil)
            .unsqueeze(0)
            .to(device, memory_format=torch.channels_last)
        )
        self.last_step.set_(last_step)

    def get_loss(self, input, img, device=DEVICE):
        """
        Uses the pretrained flow model (GMA) to compute the loss.

        :param input: The flow target image
        :param img: the DifferentiableImage we are fitting
        :param device: the device to run the model on
        :return: The loss function.
        """
        init_GMA(
            # "GMA/checkpoints/gma-sintel.pth"
        )  # update this to use model dir from config
        image1 = self.last_step
        image2 = input
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        _, flow = GMA(image1, image2, iters=3, test_mode=True)
        flow = flow.to(device, memory_format=torch.channels_last)
        return super().get_loss(TF.resize(flow, self.comp.shape[-2:]), img) / self.mag


class OpticalFlowLoss(MSELoss):
    @staticmethod
    @torch.no_grad()
    def motion_edge_map(
        flow_forward,
        flow_backward,
        img,
        border_mode="smear",
        sampling_mode="bilinear",
        device=DEVICE,
    ):
        """
        Given a forward flow, a backward flow, and an image,
        it returns a mask that is 1 where the backward flow is valid,
        and 0 where the backward flow is invalid.

        :param flow_forward: The forward flow map
        :param flow_backward: The backward flow, which is the flow from the future frame to the current
        frame
        :param img: The image to be warped
        :param border_mode: "smear" or "nearest", defaults to smear (optional)
        :param sampling_mode: "bilinear" or "nearest", defaults to bilinear (optional)
        :param device: the device to run the algorithm on
        :return: a mask that is used to mask out the unreliable pixels.
        """
        # algorithm based on https://github.com/manuelruder/artistic-videos/blob/master/consistencyChecker/consistencyChecker.cpp
        # reimplemented in pytorch by Henry Rachootin
        # // consistencyChecker
        # // Check consistency of forward flow via backward flow.
        # //
        # // (c) Manuel Ruder, Alexey Dosovitskiy, Thomas Brox 2016
        dx_ker = (
            torch.tensor([[[[0, 0, 0], [1, 0, -1], [0, 0, 0]]]], device=device)
            .float()
            .div(2)
            .repeat(2, 2, 1, 1)
        )
        dy_ker = (
            torch.tensor([[[[0, 1, 0], [0, 0, 0], [0, -1, 0]]]], device=device)
            .float()
            .div(2)
            .repeat(2, 2, 1, 1)
        )
        f_x = nn.functional.conv2d(flow_backward, dx_ker, padding="same")
        f_y = nn.functional.conv2d(flow_backward, dy_ker, padding="same")
        motionedge = torch.cat([f_x, f_y]).square().sum(dim=(0, 1))

        height, width = flow_forward.shape[-2:]
        y, x = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
        x = x.to(device)
        y = y.to(device)

        p1 = torch.stack([x, y])
        v1 = flow_forward.squeeze(0)
        p0 = p1 + flow_backward.squeeze()
        v0 = sample(v1, p0)
        p1_back = p0 + v0
        v1_back = flow_backward.squeeze(0)

        r1 = torch.floor(p0)
        r2 = r1 + 1
        max_pos = torch.tensor([width - 1, height - 1], device=device).view(2, 1, 1)
        min_pos = torch.tensor([0, 0], device=device).view(2, 1, 1)
        overshoot = torch.logical_or(r1.lt(min_pos), r2.gt(max_pos))
        overshoot = torch.logical_or(overshoot[0], overshoot[1])

        missed = (
            (p1_back - p1)
            .square()
            .sum(dim=0)
            .ge(torch.stack([v1_back, v0]).square().sum(dim=(0, 1)).mul(0.01).add(0.5))
        )
        motion_boundary = motionedge.ge(
            v1_back.square().sum(dim=0).mul(0.01).add(0.002)
        )

        reliable = torch.ones((height, width), device=device)
        reliable[motion_boundary] = 0
        reliable[missed] = -1
        reliable[overshoot] = 0
        mask = TF.gaussian_blur(reliable.unsqueeze(0), 3).clip(0, 1)

        return mask

    @staticmethod
    @torch.no_grad()
    def get_flow(image1, image2, device=DEVICE):
        """
        Takes two images and returns the flow between them.

        :param image1: The first image in the sequence
        :param image2: the image that we want to transform towards
        :param device: The device to run the model on
        :return: the flow field.
        """
        # init_GMA("GMA/checkpoints/gma-sintel.pth")
        init_GMA()
        if isinstance(image1, Image.Image):
            image1 = TF.to_tensor(image1).unsqueeze(0).to(device)
        if isinstance(image2, Image.Image):
            image2 = TF.to_tensor(image2).unsqueeze(0).to(device)
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        flow_low, flow_up = GMA(image1, image2, iters=12, test_mode=True)
        return flow_up

    def __init__(
        self,
        comp,
        weight=0.5,
        stop=-math.inf,
        name="direct target loss",
        image_shape=None,
    ):
        super().__init__(comp, weight, stop, name, image_shape)
        with torch.no_grad():
            self.latent_loss = MSELoss(
                comp.new_zeros((1, 1, 1, 1)), weight, stop, name, image_shape
            )
            self.register_buffer("bg_mask", comp.new_zeros((1, 1, 1, 1)))

    @torch.no_grad()
    def set_flow(
        self,
        frame_prev,
        frame_next,
        img,
        path,
        border_mode="smear",
        sampling_mode="bilinear",
        device=DEVICE,
    ):
        """
        Given a frame, a path to a pretrained model, and a mask,
        returns a flow field that can be applied to the frame

        :param frame_prev: The previous frame of the video
        :param frame_next: The next frame in the video
        :param img: the image to be warped
        :param path: path to the pretrained model
        :param border_mode: "smear" or "crop", defaults to smear (optional)
        :param sampling_mode: "bilinear" or "nearest", defaults to bilinear (optional)
        :param device: the device to run the model on
        :return: The flow image and the mask.
        """
        if path is not None:
            img = img.clone()
            state_dict = torch.load(path)
            img.load_state_dict(state_dict)

        gc.collect()
        torch.cuda.empty_cache()

        image1 = TF.to_tensor(frame_prev).unsqueeze(0).to(device)
        image2 = TF.to_tensor(frame_next).unsqueeze(0).to(device)

        if self.bg_mask.shape[-2:] != image1.shape[-2:]:
            bg_mask = TF.resize(self.bg_mask, image1.shape[-2:])
            self.bg_mask.set_(bg_mask)
        noise = torch.empty_like(image2)
        noise.normal_(mean=0, std=0.05)
        noise.mul_(self.bg_mask)
        # adding the same noise vectors to both images forces
        # the flow model to match those parts of the frame, effectively
        # disabling the flow in those areas.
        image1.add_(noise)
        image2.add_(noise)

        flow_forward = OpticalFlowLoss.get_flow(image1, image2)
        flow_backward = OpticalFlowLoss.get_flow(image2, image1)
        unwarped_target_direct = img.decode_tensor()
        flow_target_direct = apply_flow(
            img, -flow_backward, border_mode=border_mode, sampling_mode=sampling_mode
        )

        fancy_mask = OpticalFlowLoss.motion_edge_map(
            flow_forward, flow_backward, img, border_mode, sampling_mode
        )

        target_direct = flow_target_direct
        target_latent = img.get_latent_tensor(detach=True)
        mask = fancy_mask.unsqueeze(0)

        self.comp.set_(target_direct)
        self.latent_loss.comp.set_(target_latent)
        self.set_flow_mask(mask)

        array = (
            flow_target_direct.squeeze(0)
            .movedim(0, -1)
            .mul(255)
            .clamp(0, 255)
            .cpu()
            .detach()
            .numpy()
            .astype(np.uint8)[:, :, :]
        )
        return Image.fromarray(array), fancy_mask

    @torch.no_grad()
    def set_flow_mask(self, mask):
        """
        If a mask is provided, resize it to the size of the component image and set it as the mask for
        the latent loss. Otherwise, set the mask to None

        :param mask: a binary mask of the same size as the input image. If None, no mask is used
        """
        super().set_mask(TF.resize(mask, self.comp.shape[-2:]))
        if mask is not None:
            self.latent_loss.set_mask(TF.resize(mask, self.latent_loss.comp.shape[-2:]))
        else:
            self.latent_loss.set_mask(None)

    @torch.no_grad()
    def set_mask(self, mask, inverted=False, device=DEVICE):
        """
        Sets the mask for the background.

        :param mask: The mask to use. Can be a string, a PIL Image, or a Tensor
        :param inverted: If True, the mask is inverted. This means that the mask will be applied to the
        background, defaults to False (optional)
        :param device: The device to run the mask on
        :return: Nothing.
        """
        if isinstance(mask, str) and mask != "":
            if mask[0] == "-":
                mask = mask[1:]
                inverted = True
            if mask.strip()[-4:] == ".mp4":
                r = Rotoscoper(mask, self)
                r.update(0)
                return
            mask = Image.open(fetch(mask)).convert("L")
        if isinstance(mask, Image.Image):
            with vram_usage_mode("Masks"):
                mask = (
                    TF.to_tensor(mask)
                    .unsqueeze(0)
                    .to(device, memory_format=torch.channels_last)
                )
        if mask not in ["", None]:
            # this is where the inversion is. This mask is naturally inverted :)
            # since it selects the background
            self.bg_mask.set_(mask if inverted else (1 - mask))

    def get_loss(self, input, img):
        l1 = super().get_loss(input, img)
        l2 = self.latent_loss.get_loss(img.get_latent_tensor(), img)
        return l1 + l2 * img.latent_strength
