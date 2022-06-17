from pytti.LossAug.MSELossClass import MSELoss
import gc, torch, os, math
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from PIL import Image
import copy, re
from pytti import DEVICE, fetch, parse, vram_usage_mode


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
        self.direct_loss.set_comp(pil_image.resize(self.image_shape, Image.LANCZOS))

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
