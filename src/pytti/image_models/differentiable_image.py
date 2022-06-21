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
        Decodes the backend image representation into a tensor of pixel values.
        E.g. if the image is represented implicitly or as a learned latent vector,
        this function returns a tensor of the explicit image representation.

        If the backend image has an ephemeral representation for training that differs
        from the "final" image, this function returns the training component. I.e. if
        the backend image representation uses EMA, this function decodes the 'instantaneous'
        tensor rather than the accumulated weighted average tensor.
        """
        return self.decode_tensor()

    def clone(self):
        raise NotImplementedError

    def decode_tensor(self):
        """
        Decodes the backend image representation into a tensor of pixel values in [0,1].
        E.g. if the image is represented implicitly or as a learned latent vector,
        this function returns a tensor of the explicit image representation.

        If the backend image has an ephemeral representation for training that differs
        from the "final" image, this function returns the "final" component. I.e. if
        the backend image representation uses EMA, this function decodes the accumulated
        weighted average tensor rather than the 'instantaneous' tensor.
        """
        raise NotImplementedError

    def encode_image(self, pil_image: Image):
        """
        Given an input PIL.Image, encodes the image to the backend
        representation and stores it on this DiffImg instance.

        Invert this process via decode_image()

        pil_image: (Image) input image
        """
        raise NotImplementedError

    def encode_random(self):
        """
        Overwrites the backend image representation with random noise
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
        """
        Returns a tensor of the explicit image representation in a format suitable for
        instantiating a PIL image.
        * dimensions are rearranged to [y x s] format
        * tensor values are projected from learning domain [0,1] to PIL RGB domain [0,255]
        """
        tensor = self.decode_tensor()
        tensor = named_rearrange(tensor, self.output_axes, ("y", "x", "s"))
        return tensor.mul(255).clamp(0, 255)

    def decode_image(self):
        """
        Render a PIL Image from the backend image representation.
        Basically a convenience wrapper for the processing step after calling get_image_tensor()
        """
        array = self.get_image_tensor().detach().cpu().numpy().astype(np.uint8)[:, :, :]
        return Image.fromarray(array)

    def forward(self):
        """
        Convenience wrapper for calling the appropriate decode_tensor method given training state.
        """
        if self.training:
            return self.decode_training_tensor()
        else:
            return self.decode_tensor()
