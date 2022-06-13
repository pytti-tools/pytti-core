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
        try:
            comp_adjusted = TF.resize(comp.clone(), (h, w))
        except:
            # comp_adjusted = comp.clone()
            # Need to convert the latent to its image form
            comp_adjusted = img_model.decode_tensor(comp.clone())
        self.direct_loss = MSELoss(comp_adjusted, weight, stop, name, image_shape)

    @torch.no_grad()
    def set_comp(self, pil_image, device=DEVICE):
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
        if not self.has_latent:
            # make_latent() encodes the image through a dummy class instance, returns the resulting fitted image representation
            # if get_image_tensor() is not implemented, then the returned 'latent' tensor is just the tensorized pil image
            latent = img.make_latent(self.pil_image)
            logger.debug(type(latent))  # EMAParametersDict
            logger.debug(type(self.comp))  # torch.Tensor
            with torch.no_grad():
                self.comp.set_(latent.clone())
            self.has_latent = True
        l1 = super().get_loss(img.get_latent_tensor(), img) / 2
        l2 = self.direct_loss.get_loss(input, img) / 10
        return l1 + l2


######################################################################


class LatentLossGeneric(LatentLoss):
    # class LatentLoss(MSELoss):
    @torch.no_grad()
    def __init__(
        self,
        comp,
        weight=0.5,
        stop=-math.inf,
        name="direct target loss",
        image_shape=None,
    ):
        super().__init__(comp, weight, stop, name, image_shape)
        self.pil_image = None
        self.has_latent = False
        w, h = image_shape
        self.direct_loss = MSELoss(
            TF.resize(comp.clone(), (h, w)), weight, stop, name, image_shape
        )

    @torch.no_grad()
    def set_comp(self, pil_image, device=DEVICE):
        self.pil_image = pil_image
        self.has_latent = False
        self.direct_loss.set_comp(
            pil_image.resize(self.image_shape, Image.LANCZOS)
        )  # to do: ResizeRight

    @classmethod
    @vram_usage_mode("Latent Image Loss")
    @torch.no_grad()
    def TargetImage(
        cls, prompt_string, image_shape, pil_image=None, is_path=False, device=DEVICE
    ):
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
            else torch.zeros(1, 1, 1, 1, device=device)
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
        if not self.has_latent:
            latent = img.make_latent(self.pil_image)
            with torch.no_grad():
                self.comp.set_(latent.clone())
            self.has_latent = True
        l1 = super().get_loss(img.get_latent_tensor(), img) / 2
        l2 = self.direct_loss.get_loss(input, img) / 10
        return l1 + l2
