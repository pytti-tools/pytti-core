"""
Methods for obtaining cutouts, agnostic to augmentations.

Cutout choices have a significant impact on the performance of the perceptors and the
overall look of the image.

The objects defined here probably are only being used in pytti.Perceptor.cutouts.Embedder.HDMultiClipEmbedder, but they
should be sufficiently general for use in notebooks without pyttitools otherwise in use.
"""

import torch
from typing import Tuple
from torch.nn import functional as F

PADDING_MODES = {
    "mirror": "reflect",
    "smear": "replicate",
    "wrap": "circular",
    "black": "constant",
}

# (
# cut_size = 64
# cut_pow = 0.5
# noise_fac = 0.0
# cutn = 8
# border_mode = "clamp"
# augs = None
# return Cutout(
#     cut_size=cut_size,
#     cut_pow=cut_pow,
#     noise_fac=noise_fac,
#     cutn=cutn,
#     border_mode=border_mode,
#     augs=augs,
# )


def pytti_classic(
    # self,
    input: torch.Tensor,
    side_x,
    side_y,
    cut_size,
    padding,
    cutn,
    cut_pow,
    border_mode,
    augs,
    noise_fac,
    device,
) -> Tuple[list, list, list]:
    """
    This is the cutout method that was already in use in the original pytti.
    """
    min_size = min(side_x, side_y, cut_size)
    max_size = min(side_x, side_y)
    paddingx = min(round(side_x * padding), side_x)
    paddingy = min(round(side_y * padding), side_y)
    cutouts = []
    offsets = []
    sizes = []
    for _ in range(cutn):
        # mean is 0.8
        # varience is 0.3
        size = int(
            max_size
            * (
                torch.zeros(
                    1,
                )
                .normal_(mean=0.8, std=0.3)
                .clip(cut_size / max_size, 1.0)
                ** cut_pow
            )
        )
        offsetx_max = side_x - size + 1
        offsety_max = side_y - size + 1
        if border_mode == "clamp":
            offsetx = torch.clamp(
                (torch.rand([]) * (offsetx_max + 2 * paddingx) - paddingx)
                .floor()
                .int(),
                0,
                offsetx_max,
            )
            offsety = torch.clamp(
                (torch.rand([]) * (offsety_max + 2 * paddingy) - paddingy)
                .floor()
                .int(),
                0,
                offsety_max,
            )
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
        else:
            px = min(size, paddingx)
            py = min(size, paddingy)
            offsetx = (torch.rand([]) * (offsetx_max + 2 * px) - px).floor().int()
            offsety = (torch.rand([]) * (offsety_max + 2 * py) - py).floor().int()
            cutout = input[
                :,
                :,
                paddingy + offsety : paddingy + offsety + size,
                paddingx + offsetx : paddingx + offsetx + size,
            ]
        cutouts.append(F.adaptive_avg_pool2d(cutout, cut_size))
        offsets.append(
            torch.as_tensor([[offsetx / side_x, offsety / side_y]]).to(device)
        )
        sizes.append(torch.as_tensor([[size / side_x, size / side_y]]).to(device))
    cutouts = augs(torch.cat(cutouts))
    offsets = torch.cat(offsets)
    sizes = torch.cat(sizes)
    if noise_fac:
        facs = cutouts.new_empty([cutn, 1, 1, 1]).uniform_(0, noise_fac)
        cutouts.add_(facs * torch.randn_like(cutouts))
    return cutouts, offsets, sizes
