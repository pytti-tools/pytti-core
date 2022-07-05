import gc
import math

from adabins.infer import InferenceHelper
from loguru import logger
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from pytti import DEVICE, vram_usage_mode
from pytti.LossAug.MSELossClass import MSELoss


infer_helper = None


def init_AdaBins(device=None):
    global infer_helper
    if infer_helper is None:
        with vram_usage_mode("AdaBins"):
            logger.debug("Loading AdaBins...")
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            infer_helper = InferenceHelper(dataset="nyu", device=device)
            logger.debug("AdaBins loaded.")


class DepthLoss(MSELoss):
    @torch.no_grad()
    def set_comp(self, pil_image):
        self.comp.set_(DepthLoss.make_comp(pil_image))
        if self.use_mask and self.mask.shape[-2:] != self.comp.shape[-2:]:
            self.mask.set_(TF.resize(self.mask, self.comp.shape[-2:]))

    def get_loss(self, input, img):
        height, width = input.shape[-2:]
        max_depth_area = 500000
        image_area = width * height
        if image_area > max_depth_area:
            depth_scale_factor = math.sqrt(max_depth_area / image_area)
            height, width = int(height * depth_scale_factor), int(
                width * depth_scale_factor
            )
            depth_input = TF.resize(
                input, (height, width), interpolation=TF.InterpolationMode.BILINEAR
            )
        else:
            depth_input = input

        _, depth_map = infer_helper.model(depth_input)
        depth_map = F.interpolate(
            depth_map, self.comp.shape[-2:], mode="bilinear", align_corners=True
        )
        return super().get_loss(depth_map, img)

    @classmethod
    @vram_usage_mode("Depth Loss")
    def make_comp(cls, pil_image, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        depth, _ = DepthLoss.get_depth(pil_image, device=device)
        return torch.from_numpy(depth).to(device)

    @staticmethod
    def get_depth(pil_image, device=None):
        init_AdaBins(device=device)
        width, height = pil_image.size

        # if the area of an image is above this, the depth model fails
        max_depth_area = 500000
        image_area = width * height
        if image_area > max_depth_area:
            depth_scale_factor = math.sqrt(max_depth_area / image_area)
            depth_input = pil_image.resize(
                (int(width * depth_scale_factor), int(height * depth_scale_factor)),
                Image.LANCZOS,
            )
            depth_resized = True
        else:
            depth_input = pil_image
            depth_resized = False

        gc.collect()
        torch.cuda.empty_cache()
        _, depth_map = infer_helper.predict_pil(depth_input)
        gc.collect()
        torch.cuda.empty_cache()

        return depth_map, depth_resized
