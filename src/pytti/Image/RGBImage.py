from pytti import DEVICE, clamp_with_grad
import torch
from torch import nn
from torchvision.transforms import functional as TF
from pytti.Image import DifferentiableImage
from PIL import Image
from torch.nn import functional as F


class RGBImage(DifferentiableImage):
    """
    Naive RGB image representation
    """

    def __init__(self, width, height, scale=1, device=DEVICE):
        super().__init__(width * scale, height * scale)
        self.tensor = nn.Parameter(
            torch.zeros(1, 3, height, width).to(
                device=device, memory_format=torch.channels_last
            )
        )
        self.output_axes = ("n", "s", "y", "x")
        self.scale = scale

    def decode_tensor(self):
        '''
        Given a tensor, resize it to the given image shape and clamp the values to be between 0 and 1
        :return: The decoded tensor.
        '''
        width, height = self.image_shape
        out = F.interpolate(self.tensor, (height, width), mode="nearest")
        return clamp_with_grad(out, 0, 1)

    def clone(self):
        '''
        It takes an image and returns a smaller version of it
        :return: A new RGBImage object with the same dimensions as the original image, but with the
        tensor values being a clone of the original image's tensor.
        '''
        width, height = self.image_shape
        dummy = RGBImage(width // self.scale, height // self.scale, self.scale)
        with torch.no_grad():
            dummy.tensor.set_(self.tensor.clone())
        return dummy

    def get_image_tensor(self):
        '''
        Gets the attached image tensor with batch dimension squeezed.
        '''
        return self.tensor.squeeze(0)

    @torch.no_grad()
    def set_image_tensor(self, tensor):
        self.tensor.set_(tensor.unsqueeze(0))

    @torch.no_grad()
    def encode_image(self, pil_image, device=DEVICE, **kwargs):
        '''
        1. Resize the image to the desired size.
        2. Convert the image to a tensor.
        3. Add a batch dimension to the tensor.
        4. Send the tensor to the device
        
        :param pil_image: The image to be encoded
        :param device: The device to use for the computation
        '''
        width, height = self.image_shape
        scale = self.scale
        pil_image = pil_image.resize((width // scale, height // scale), Image.LANCZOS)
        self.tensor.set_(
            TF.to_tensor(pil_image)
            .unsqueeze(0)
            .to(device, memory_format=torch.channels_last)
        )

    @torch.no_grad()
    def encode_random(self):
        '''
        Sets the attached tensor to uniform noise
        '''
        self.tensor.uniform_()
