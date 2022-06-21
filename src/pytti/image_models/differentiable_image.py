import copy
from torch import nn
import numpy as np
from PIL import Image
from pytti.tensor_tools import named_rearrange

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
        self.pixel_format = pixel_format
        self.output_axes = ("x", "y", "s")
        self.lr = 0.02
        # TODO: 'latent_strength' shouldn't be a base class attribute.
        self.latent_strength = 0

    def decode_training_tensor(self):
        """
        returns a decoded tensor of this image for training
        """
        return self.decode_tensor()

    def clone(self):
        raise NotImplementedError

    def decode_tensor(self):
        """
        returns a decoded tensor of this image
        """
        raise NotImplementedError

    def encode_image(self, pil_image):
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

    @classmethod
    def get_preferred_loss(cls):
        from pytti.LossAug.HSVLossClass import HSVLoss

        return HSVLoss

    def get_image_tensor(self):
        tensor = self.decode_tensor()
        tensor = named_rearrange(tensor, self.output_axes, ("y", "x", "s"))
        return tensor.mul(255).clamp(0, 255)

    def decode_image(self):
        """
        render a PIL Image version of this image
        """
        array = self.get_image_tensor().detach().cpu().numpy().astype(np.uint8)[:, :, :]
        return Image.fromarray(array)

    def forward(self):
        """
        returns a decoded tensor of this image
        """
        if self.training:
            return self.decode_training_tensor()
        else:
            return self.decode_tensor()
