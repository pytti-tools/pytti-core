from pytti.LossAug.MSELossClass import MSELoss
import gc, torch, os, math
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from PIL import Image
import copy, re
from pytti import DEVICE, fetch, parse, vram_usage_mode

from loguru import logger


class LatentLoss(MSELoss):
    @torch.no_grad()
    def __init__(
        self,
        comp,
        weight=0.5,
        stop=-math.inf,
        name="direct target loss",
        image_shape=None,
    ):
        super().__init__(
            comp, weight, stop, name, image_shape
        )  # this really should link back to the image model...
        logger.debug(type(comp))  # inits to image tensor
        self.pil_image = None
        self.has_latent = False
        w, h = image_shape
        comp_adjusted = TF.resize(comp.clone(), (h, w))
        # try:
        #     comp_adjusted = TF.resize(comp.clone(), (h, w))
        # except:
        #     # comp_adjusted = comp.clone()
        #     # Need to convert the latent to its image form
        #     comp_adjusted = img_model.decode_tensor(comp.clone())
        self.direct_loss = MSELoss(comp_adjusted, weight, stop, name, image_shape)

    @torch.no_grad()
    def set_comp(self, pil_image, device=DEVICE):
        """
        sets the DIRECT loss anchor "comp" to the tensorized image.
        """
        logger.debug(type(pil_image))
        self.pil_image = pil_image
        self.has_latent = False
        im_resized = pil_image.resize(
            self.image_shape, Image.LANCZOS
        )  # to do: ResizeRight
        # self.direct_loss.set_comp(im_resized)
        self.direct_loss.set_comp(im_resized)

    @classmethod
    def convert_input(cls, input, img):
        """
        Converts the input image tensor to the image representation of the image model.
        E.g. if img is VQGAN, then the input tensor is converted to the latent representation.
        """
        logger.debug(type(input))  # pretty sure this is gonna be tensor
        # return input # this is the default MSE loss version
        return img.make_latent(input)

    @classmethod
    def default_comp(cls, img_model=None, *args, **kargs):
        logger.debug("default_comp")
        logger.debug(type(img_model))
        device = kargs.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        if img_model is None:
            return torch.zeros(1, 1, 1, 1, device=device)
        return img_model.default_comp(*args, **kargs)

    @classmethod
    @vram_usage_mode("Latent Image Loss")
    @torch.no_grad()
    def TargetImage(
        cls,
        prompt_string,
        image_shape,
        pil_image=None,
        is_path=False,
        device=DEVICE,
        img_model=None,
    ):
        logger.debug(
            type(pil_image)
        )  # None. emitted prior to do_run:559 but after parse_scenes:122. Why even use this constructor if no pil_image?
        text, weight, stop = parse(
            prompt_string, r"(?<!^http)(?<!s):|:(?!/)", ["", "1", "-inf"]
        )
        weight, mask = parse(weight, r"_", ["1", ""])
        text = text.strip()
        mask = mask.strip()
        if pil_image is None and text != "" and is_path:
            pil_image = Image.open(fetch(text)).convert("RGB")
        comp = (
            MSELoss.make_comp(pil_image)
            if pil_image is not None
            # else torch.zeros(1, 1, 1, 1, device=device)
            else cls.default_comp(img_model=img_model)
        )
        out = cls(comp, weight, stop, text + " (latent)", image_shape)
        if pil_image is not None:
            out.set_comp(pil_image)
        out.set_mask(mask)
        return out

    def set_mask(self, mask, inverted=False):
        self.direct_loss.set_mask(mask, inverted)
        super().set_mask(mask, inverted)

    def get_loss(self, input, img):
        logger.debug(type(input))  # Tensor
        logger.debug(input.shape)  # this is an image tensor
        logger.debug(type(img))  # DIPImage
        logger.debug(type(self.comp))  # Tensor
        logger.debug(
            self.comp.shape
        )  # [1 1 1 1] -> from target image constructor when no input image provided

        # why is the latent comp only set here? why not in the __init__ and set_comp?
        if not self.has_latent:
            # make_latent() encodes the image through a dummy class instance, returns the resulting fitted image representation
            # if get_image_tensor() is not implemented, then the returned 'latent' tensor is just the tensorized pil image
            latent = img.make_latent(self.pil_image)
            logger.debug(type(latent))  # EMAParametersDict
            logger.debug(type(self.comp))  # torch.Tensor
            with torch.no_grad():
                if type(latent) == type(self.comp):
                    self.comp.set_(latent.clone())
                # else:

            self.has_latent = True

        l1 = super().get_loss(img.get_latent_tensor(), img) / 2
        l2 = self.direct_loss.get_loss(input, img) / 10
        return l1 + l2


######################################################################

# fuck it, let's just make a dip latent loss from scratch.


# The issue we're resolving here is that by inheriting from the MSELoss,
# I can't easily set the comp to the parameters of the image model.

from pytti.LossAug.BaseLossClass import Loss
from pytti.image_models.ema import EMAImage, EMAParametersDict
from pytti.rotoscoper import Rotoscoper

import deep_image_prior
import deep_image_prior.models
from deep_image_prior.models import (
    get_hq_skip_net,
    get_non_offset_params,
    get_offset_params,
)


def load_dip(input_depth, num_scales, offset_type, offset_groups, device):
    dip_net = get_hq_skip_net(
        input_depth,
        skip_n33d=192,
        skip_n33u=192,
        skip_n11=4,
        num_scales=num_scales,
        offset_type=offset_type,
        offset_groups=offset_groups,
    ).to(device)

    return dip_net


class LatentLossDIP(Loss):
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
        ##################################################################
        super().__init__(weight, stop, name, device)
        if image_shape is None:
            raise
            # height, width = comp.shape[-2:]
            # image_shape = (width, height)
        self.image_shape = image_shape
        self.register_buffer("mask", torch.ones(1, 1, 1, 1, device=self.device))
        self.use_mask = False
        ##################################################################
        self.pil_image = None
        self.has_latent = False
        logger.debug(type(comp))  # inits to image tensor
        if comp is None:
            comp = self.default_comp()
        if isinstance(comp, EMAParametersDict):
            logger.debug("initializing loss from latent")
            self.register_module("comp", comp)
            self.has_latent = True
        else:
            w, h = image_shape
            comp_adjusted = TF.resize(comp.clone(), (h, w))
            # try:
            #     comp_adjusted = TF.resize(comp.clone(), (h, w))
            # except:
            #     # comp_adjusted = comp.clone()
            #     # Need to convert the latent to its image form
            #     comp_adjusted = img_model.decode_tensor(comp.clone())
            self.direct_loss = MSELoss(comp_adjusted, weight, stop, name, image_shape)

        ##################################################################

        logger.debug(type(comp))

    @classmethod
    def default_comp(*args, **kargs):
        logger.debug("default_comp")
        device = kargs.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        net = load_dip(
            input_depth=32,
            num_scales=7,
            offset_type="none",
            offset_groups=4,
            device=device,
        )
        return EMAParametersDict(z=net, decay=0.99, device=device)

    ###################################################################################

    @torch.no_grad()
    def set_comp(self, pil_image, device=DEVICE):
        """
        sets the DIRECT loss anchor "comp" to the tensorized image.
        """
        logger.debug(type(pil_image))
        self.pil_image = pil_image
        self.has_latent = False
        im_resized = pil_image.resize(
            self.image_shape, Image.LANCZOS
        )  # to do: ResizeRight
        # self.direct_loss.set_comp(im_resized)

        im_tensor = (
            TF.to_tensor(pil_image)
            .unsqueeze(0)
            .to(device, memory_format=torch.channels_last)
        )

        if hasattr(self, "direct_loss"):
            self.direct_loss.set_comp(im_tensor)
        else:
            self.direct_loss = MSELoss(
                im_tensor, self.weight, self.stop, self.name, self.image_shape
            )
        # self.direct_loss.set_comp(im_resized)

    @classmethod
    def convert_input(cls, input, img):
        """
        Converts the input image tensor to the image representation of the image model.
        E.g. if img is VQGAN, then the input tensor is converted to the latent representation.
        """
        logger.debug(type(input))  # pretty sure this is gonna be tensor
        # return input # this is the default MSE loss version
        return img.make_latent(input)

    @classmethod
    @vram_usage_mode("Latent Image Loss")
    @torch.no_grad()
    def TargetImage(
        cls,
        prompt_string,
        image_shape,
        pil_image=None,
        is_path=False,
        device=DEVICE,
        img_model=None,
    ):
        logger.debug(
            type(pil_image)
        )  # None. emitted prior to do_run:559 but after parse_scenes:122. Why even use this constructor if no pil_image?
        text, weight, stop = parse(
            prompt_string, r"(?<!^http)(?<!s):|:(?!/)", ["", "1", "-inf"]
        )
        weight, mask = parse(weight, r"_", ["1", ""])
        text = text.strip()
        mask = mask.strip()
        if pil_image is None and text != "" and is_path:
            pil_image = Image.open(fetch(text)).convert("RGB")
        comp = (
            MSELoss.make_comp(pil_image)
            if pil_image is not None
            # else torch.zeros(1, 1, 1, 1, device=device)
            else cls.default_comp(img_model=img_model)
        )
        out = cls(comp, weight, stop, text + " (latent)", image_shape)
        if pil_image is not None:
            out.set_comp(pil_image)
        if (
            mask
        ):  # this will break if there's no pil_image since the direct_loss won't be initialized
            out.set_mask(mask)
        return out

    def set_mask(self, mask, inverted=False):
        self.direct_loss.set_mask(mask, inverted)
        # super().set_mask(mask, inverted)
        # if device is None:
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

    def get_loss(self, input, img):
        logger.debug(type(input))  # Tensor
        logger.debug(input.shape)  # this is an image tensor
        logger.debug(type(img))  # DIPImage
        logger.debug(type(self.comp))  # EMAParametersDict
        # logger.debug(
        #    self.comp.shape
        # )  # [1 1 1 1] -> from target image constructor when no input image provided

        # why is the latent comp only set here? why not in the __init__ and set_comp?
        if not self.has_latent:
            raise
            # make_latent() encodes the image through a dummy class instance, returns the resulting fitted image representation
            # if get_image_tensor() is not implemented, then the returned 'latent' tensor is just the tensorized pil image
            latent = img.make_latent(self.pil_image)
            logger.debug(type(latent))  # EMAParametersDict
            logger.debug(type(self.comp))  # torch.Tensor
            with torch.no_grad():
                if type(latent) == type(self.comp):
                    self.comp.set_(latent.clone())
                # else:

            self.has_latent = True

        estimated_image = self.comp.get_image_tensor()

        l1 = super().get_loss(img.get_latent_tensor(), img) / 2
        l2 = self.direct_loss.get_loss(input, img) / 10
        return l1 + l2
