import copy

import torch
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

    def __init__(self, width: int, height: int, pixel_format: str = "RGB", device=None):
        super().__init__()
        if pixel_format not in SUPPORTED_MODES:
            raise ValueError(f"Pixel format {pixel_format} is not supported.")
        self.image_shape = (width, height)
        self.pixel_format = format
        self.output_axes = ("x", "y", "s")
        self.lr = 0.02
        self.latent_strength = 0
        self.image_representation_parameters = ImageRepresentationalParameters(
            width, height
        )
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.tensor = self.image_representation_parameters._new()

    def decode_training_tensor(self):
        """
        returns a decoded tensor of this image for training
        """
        return self.decode_tensor()

    def get_image_tensor(self):
        """
        optional method: returns an [n x w_i x h_i] tensor representing the local image data
        those data will be used for animation if afforded
        """
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError

    def get_latent_tensor(self, detach=False):
        if detach:
            return self.get_image_tensor().detach()
        else:
            return self.get_image_tensor()

    def set_image_tensor(self, tensor):
        """
        optional method: accepts an [n x w_i x h_i] tensor representing the local image data
        those data will be by the animation system
        """
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

    def make_latent(self, pil_image):
        try:
            dummy = self.clone()
        except NotImplementedError:
            dummy = copy.deepcopy(self)
        dummy.encode_image(pil_image)
        return dummy.get_latent_tensor(detach=True)

    @classmethod
    def get_preferred_loss(cls):
        from pytti.LossAug.HSVLossClass import HSVLoss

        return HSVLoss

    def image_loss(self):
        return []

    def decode_image(self):
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

    def forward(self):
        """
        returns a decoded tensor of this image
        """
        if self.training:
            return self.decode_training_tensor()
        else:
            return self.decode_tensor()

    @property
    def representation_parameters(self):
        return self.image_representation_parameters._container

        ## yeah I should really make this class an ABC
        # if not hasattr(self, "representation_parameters"):
        #    raise NotImplementedError
        # return self.tensor


class ImageRepresentationalParameters(nn.Module):
    """
    Base class for defining parameters of differentiable images
    width:        (positive integer) image width in pixels
    height:       (positive integer) image height in pixels
    pixel_format: (string) PIL image mode. Either 'L','RGB','I', or 'F'
    """

    def __init__(self, width: int, height: int, device=None):
        super().__init__()
        self.image_shape = (width, height)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._container = self._new()

    def _new(self):
        width, height = self.image_shape
        return nn.Parameter(
            torch.zeros(1, 3, height, width).to(
                device=self.device, memory_format=torch.channels_last
            )
        )
        # self.tensor = ImageRepresentationalParameters(width, height, pixel_format)

    ################
    def mul_():
        pass

    def add_():
        pass

    def copy():
        pass

    def div_():
        pass

    def set_():
        pass
