import math, re
from PIL import Image
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from pytti.LossAug.BaseLossClass import Loss

from pytti.rotoscoper import Rotoscoper
from pytti import fetch, vram_usage_mode
from pytti.eval_tools import parse_subprompt
import torch


class MSELoss(Loss):
    @torch.no_grad()
    def __init__(
        self,
        comp,
        weight=0.5,
        stop=-math.inf,
        name="direct target loss",
        image_shape=None,
        device=None,
    ):
        super().__init__(weight, stop, name, device)
        self.register_buffer("comp", comp)
        if image_shape is None:
            height, width = comp.shape[-2:]
            image_shape = (width, height)
        self.image_shape = image_shape
        self.register_buffer("mask", torch.ones(1, 1, 1, 1, device=self.device))
        self.use_mask = False

    @torch.no_grad()
    def set_mask(self, mask, inverted=False, device=None):
        if device is None:
            device = self.device
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
            self.mask.set_(mask if not inverted else (1 - mask))
        self.use_mask = mask not in ["", None]

    @classmethod
    def convert_input(cls, input, img):
        return input

    @classmethod
    def make_comp(cls, pil_image, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        out = (
            TF.to_tensor(pil_image)
            .unsqueeze(0)
            .to(device, memory_format=torch.channels_last)
        )
        return cls.convert_input(out, None)

    def set_comp(self, pil_image, device=None):
        if device is None:
            device = self.device
        self.comp.set_(type(self).make_comp(pil_image, device=device))

    def get_loss(self, input, img):
        input = type(self).convert_input(input, img)
        if self.use_mask:
            if self.mask.shape[-2:] != input.shape[-2:]:
                with torch.no_grad():
                    mask = TF.resize(self.mask, input.shape[-2:])
                    self.set_mask(mask)
            return F.mse_loss(input * self.mask, self.comp * self.mask)
        else:
            return F.mse_loss(input, self.comp)
