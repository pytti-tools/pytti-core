import copy
from torch import nn
import numpy as np
from PIL import Image
from pytti.tensor_tools import named_rearrange

# for typing
import torch
from pytti.LossAug.BaseLossClass import Loss

SUPPORTED_MODES = ["L", "RGB", "I", "F"]


class DifferentiableImage(nn.Module):
    """
    Base class for defining differentiable images
    width:        (positive integer) image width in pixels
    height:       (positive integer) image height in pixels
    pixel_format: (string) PIL image mode. Either 'L','RGB','I', or 'F'
    """

    def __init__(self, width: int, height: int, pixel_format: str = "RGB"):
        super().__init__()
        if pixel_format not in SUPPORTED_MODES:
            raise ValueError(f"Pixel format {pixel_format} is not supported.")
        self.image_shape = (width, height)
        self.pixel_format = format
        self.output_axes = ("x", "y", "s")
        self.lr = 0.02
        self.latent_strength = 0

    def decode_training_tensor(self) -> torch.Tensor:
        """
        returns a decoded tensor of this image for training
        """
        return self.decode_tensor()

    def get_image_tensor(self) -> torch.Tensor:
        """
        optional method: returns an [n x w_i x h_i] tensor representing the local image data
        those data will be used for animation if afforded
        """
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError

    def get_latent_tensor(self, detach=False) -> torch.Tensor:
        if detach:
            return self.get_image_tensor().detach()
        else:
            return self.get_image_tensor()

    def set_image_tensor(self, tensor: torch.Tensor):
        """
        optional method: accepts an [n x w_i x h_i] tensor representing the local image data
        those data will be by the animation system
        """
        raise NotImplementedError

    def decode_tensor(self) -> torch.Tensor:
        """
        returns a decoded tensor of this image
        """
        raise NotImplementedError

    def encode_image(self, pil_image: Image):
        """
        overwrites this image with the input image
        pil_image: (Image) input image
        """
        raise NotImplementedError

    def encode_random(self):
        """
        overwrites this image with random noise
        """
        raise NotImplementedError

    def update(self):
        """
        callback hook called once per training step by the optimizer
        """
        pass

    def make_latent(self, pil_image: Image) -> torch.Tensor:
        try:
            dummy = self.clone()
        except NotImplementedError:
            dummy = copy.deepcopy(self)
        dummy.encode_image(pil_image)
        return dummy.get_latent_tensor(detach=True)

    @classmethod
    def get_preferred_loss(cls) -> Loss:
        from pytti.LossAug.HSVLossClass import HSVLoss

        return HSVLoss

    def image_loss(self):
        return []

    def decode_image(self) -> Image:
        """
        render a PIL Image version of this image
        """
        tensor = self.decode_tensor()
        tensor = named_rearrange(tensor, self.output_axes, ("y", "x", "s"))
        array = (
            tensor.mul(255)
            .clamp(0, 255)
            .cpu()
            .detach()
            .numpy()
            .astype(np.uint8)[:, :, :]
        )
        return Image.fromarray(array)

    def forward(self) -> torch.Tensor:
        """
        returns a decoded tensor of this image
        """
        if self.training:
            return self.decode_training_tensor()
        else:
            return self.decode_tensor()
