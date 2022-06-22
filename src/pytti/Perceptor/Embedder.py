from typing import Tuple

from loguru import logger

import pytti
from pytti import format_input, cat_with_pad, format_module, normalize

# from pytti.ImageGuide import DirectImageGuide
from pytti.image_models import DifferentiableImage

import torch
from torch import nn
from torch.nn import functional as F


# import .cutouts
# import .cutouts as cutouts
# import cutouts

from .cutouts import augs as cutouts_augs
from .cutouts import samplers as cutouts_samplers

PADDING_MODES = {
    "mirror": "reflect",
    "smear": "replicate",
    "wrap": "circular",
    "black": "constant",
}


class HDMultiClipEmbedder(nn.Module):
    """
    Multi-CLIP embedder that uses cutouts to view images larger than 224x224.
    with code by Katherine Crowson (https://github.com/crowsonkb)
    and jbusted (https://twitter.com/jbusted1)
    and dribnet (https://github.com/dribnet)
    """

    def __init__(
        self,
        perceptors=None,
        cutn=40,
        cut_pow=1.5,
        padding=0.25,
        border_mode="clamp",
        noise_fac=0.1,
        device=None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        if perceptors is None:
            perceptors = pytti.Perceptor.CLIP_PERCEPTORS
        self.cut_sizes = [p.visual.input_resolution for p in perceptors]
        self.cutn = cutn
        self.noise_fac = noise_fac
        self.augs = cutouts_augs.pytti_classic()
        self.input_axes = ("n", "s", "y", "x")
        self.output_axes = ("c", "n", "i")
        self.perceptors = perceptors
        self.padding = padding
        self.cut_pow = cut_pow
        self.border_mode = border_mode

    def make_cutouts(
        self,
        input: torch.Tensor,
        side_x,
        side_y,
        cut_size,
        ####
        # padding,
        # cutn,
        # cut_pow,
        # border_mode,
        # augs,
        # noise_fac,
        ####
        device=None,
    ) -> Tuple[list, list, list]:
        if device is None:
            device = self.device
        cutouts, offsets, sizes = cutouts_samplers.pytti_classic(
            input=input,
            side_x=side_x,
            side_y=side_y,
            cut_size=cut_size,
            padding=self.padding,
            cutn=self.cutn,
            cut_pow=self.cut_pow,
            border_mode=self.border_mode,
            augs=self.augs,
            noise_fac=self.noise_fac,
            device=device,
        )
        return cutouts, offsets, sizes

    def forward(
        self,
        # diff_image: DirectImageGuide,
        diff_image: DifferentiableImage,
        input=None,
        device=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        diff_image: (DifferentiableImage) input image
        returns images embeds
        """
        if device is None:
            device = self.device
        perceptors = self.perceptors
        side_x, side_y = diff_image.image_shape
        if input is None:
            input = format_module(diff_image, self).to(
                device=device, memory_format=torch.channels_last
            )
        else:
            input = format_input(input, diff_image, self).to(
                device=device, memory_format=torch.channels_last
            )
        max_size = min(side_x, side_y)
        image_embeds = []
        all_offsets = []
        all_sizes = []

        paddingx = min(round(side_x * self.padding), side_x)
        paddingy = min(round(side_y * self.padding), side_y)
        if self.border_mode != "clamp":
            input = F.pad(
                input,
                (paddingx, paddingx, paddingy, paddingy),
                mode=PADDING_MODES[self.border_mode],
            )
        for cut_size, perceptor in zip(self.cut_sizes, perceptors):
            logger.debug(f"cut_size: {cut_size}")  # 224
            logger.debug(input.shape)  # 1, 3, 512, 512
            logger.debug(side_x)  # 2048
            logger.debug(side_y)  # 2048
            cutouts, offsets, sizes = self.make_cutouts(input, side_x, side_y, cut_size)
            clip_in = normalize(cutouts)
            image_embeds.append(perceptor.encode_image(clip_in).float().unsqueeze(0))
            all_offsets.append(offsets)
            all_sizes.append(sizes)
        return (
            cat_with_pad(image_embeds),
            torch.stack(all_offsets),
            torch.stack(all_sizes),
        )
