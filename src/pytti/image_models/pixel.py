from pytti import DEVICE, named_rearrange, replace_grad, vram_usage_mode
from pytti.image_models.differentiable_image import DifferentiableImage
from pytti.LossAug.HSVLossClass import HSVLoss

import numpy as np
import torch, math
from torch import nn, optim
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from PIL import Image, ImageOps


def break_tensor(tensor):
    """
    Given a tensor, break it into a tuple of four tensors:
    the floor of the tensor, the ceiling of the tensor,
    the rounded tensor, and the fractional part of the tensor

    :param tensor: the tensor to be broken down
    :return: 4 tensors:
        - floors: tensor of integer values that are the largest integer less than or equal to the
    corresponding element in the input tensor
        - ceils: tensor of integer values that are the smallest integer greater than or equal to the
    corresponding element in the input tensor
        - rounds: tensor of integer values that are the nearest integer
    """
    floors = tensor.floor().long()
    ceils = tensor.ceil().long()
    rounds = tensor.round().long()
    fracs = tensor - floors
    return floors, ceils, rounds, fracs


class PalletLoss(nn.Module):
    """Palette normalization"""

    def __init__(self, n_pallets, weight=0.15, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.n_pallets = n_pallets
        self.register_buffer("weight", torch.as_tensor(weight).to(self.device))

    def forward(self, input: DifferentiableImage):
        """
        Given a pixel image, the function returns the mean of the loss of the softmax of the pixel image

        :param input: a PixelImage
        :return: The loss and the loss_raw.
        """
        if isinstance(input, PixelImage):
            tensor = (
                input.tensor.movedim(0, -1)
                .contiguous()
                .view(-1, self.n_pallets)
                .softmax(dim=-1)
            )
            N, n = tensor.shape
            mu = tensor.mean(dim=0, keepdim=True)
            sigma = tensor.std(dim=0, keepdim=True)
            tensor = tensor.sub(mu)
            # SVD
            S = (tensor.transpose(0, 1) @ tensor).div(sigma * sigma.transpose(0, 1) * N)
            # minimize correlation (anticorrelate palettes)
            S.sub_(torch.diag(S.diagonal()))
            loss_raw = S.mean()
            # maximze varience within each palette
            loss_raw.add_(sigma.mul(N).pow(-1).mean())
            return loss_raw * self.weight, loss_raw
        else:
            return 0, 0

    @torch.no_grad()
    def set_weight(self, weight, device=None):
        """
        Set the weight of the layer to the given value

        :param weight: The weight tensor
        :param device: The device to put the weights on
        """
        if device is None:
            device = self.device
        self.weight.set_(torch.as_tensor(weight, device=device))

    def __str__(self):
        return "Palette normalization"


class HdrLoss(nn.Module):
    def __init__(
        self,
        pallet_size: int,
        n_pallets: int,
        gamma: float = 2.5,
        weight: float = 0.15,
        device=None,
    ):
        """
        Create a tensor of size (pallet_size, n_pallets) and set the first row to be the pallet_size
        values raised to the power of gamma

        :param pallet_size: The number of colors in the pallet
        :param n_pallets: The number of pallets in the warehouse
        :param gamma: The gamma parameter for the power law
        :param weight: The weight of the loss
        :param device: The device to run the model on
        """
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.register_buffer(
            "comp",
            torch.linspace(0, 1, pallet_size)
            .pow(gamma)
            .view(pallet_size, 1)
            .repeat(1, n_pallets)
            .to(device),
        )
        self.register_buffer("weight", torch.as_tensor(weight).to(device))

    def forward(self, input: DifferentiableImage):
        """
        Given a Pixelimage and returns the loss.

        :param input: The input image
        :return: The loss and the loss itself.
        """
        if isinstance(input, PixelImage):
            pallet = input.sort_pallet()
            magic_color = pallet.new_tensor([[[0.299, 0.587, 0.114]]])
            color_norms = torch.linalg.vector_norm(
                pallet * (magic_color.sqrt()), dim=-1
            )
            loss_raw = F.mse_loss(color_norms, self.comp)
            return loss_raw * self.weight, loss_raw
        else:
            return 0, 0

    @torch.no_grad()
    def set_weight(self, weight, device=None):
        if device is None:
            device = self.device
        self.weight.set_(torch.as_tensor(weight, device=device))

    def __str__(self):
        return "HDR normalization"


def get_closest_color(a, b):
    """
    a: h1 x w1 x 3 pytorch tensor
    b: h2 x w2 x 3 pytorch tensor
    returns: h1 x w1 x 3 pytorch tensor containing the nearest color in b to the corresponding pixels in a"""
    a_flat = a.contiguous().view(1, -1, 3)
    b_flat = b.contiguous().view(-1, 1, 3)
    a_b = torch.norm(a_flat - b_flat, dim=2, keepdim=True)
    index = torch.argmin(a_b, dim=0)
    closest_color = b_flat[index]
    return closest_color.contiguous().view(a.shape)


class PixelImage(DifferentiableImage):
    """
    differentiable image format for pixel art images
    """

    @vram_usage_mode("Limited Palette Image")
    def __init__(
        self,
        width,
        height,
        scale,
        pallet_size,
        n_pallets,
        gamma=1,
        hdr_weight=0.5,
        norm_weight=0.1,
        device=None,
    ):
        super().__init__(width * scale, height * scale)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.pallet_inertia = 2
        pallet = (
            torch.linspace(0, self.pallet_inertia, pallet_size)
            .pow(gamma)
            .view(pallet_size, 1, 1)
            .repeat(1, n_pallets, 3)
        )
        self.pallet = nn.Parameter(pallet.to(self.device))

        self.pallet_size = pallet_size
        self.n_pallets = n_pallets
        self.value = nn.Parameter(torch.zeros(height, width).to(self.device))
        self.tensor = nn.Parameter(
            torch.zeros(n_pallets, height, width).to(self.device)
        )
        self.output_axes = ("n", "s", "y", "x")
        self.latent_strength = 0.1
        self.scale = scale
        self.hdr_loss = (
            HdrLoss(pallet_size, n_pallets, gamma, hdr_weight)
            if hdr_weight != 0
            else None
        )
        self.loss = PalletLoss(n_pallets, norm_weight)
        self.register_buffer("pallet_target", torch.empty_like(self.pallet))
        self.use_pallet_target = False

    def clone(self):
        """
        Returns a new PixelImage object with the same parameters as the original, and copies the
        tensor and pallet values from the original
        :return: A new PixelImage object with the same parameters as the
        original.
        """
        width, height = self.image_shape
        dummy = PixelImage(
            width // self.scale,
            height // self.scale,
            self.scale,
            self.pallet_size,
            self.n_pallets,
            hdr_weight=0 if self.hdr_loss is None else float(self.hdr_loss.weight),
            norm_weight=float(self.loss.weight),
        )
        with torch.no_grad():
            dummy.value.set_(self.value.clone())
            dummy.tensor.set_(self.tensor.clone())
            dummy.pallet.set_(self.pallet.clone())
            dummy.pallet_target.set_(self.pallet_target.clone())
            dummy.use_pallet_target = self.use_pallet_target
        return dummy

    def set_pallet_target(self, pil_image):
        """
        If the user provides a pallet image, encode it and set it as the pallet target

        :param pil_image: A PIL image (this might be wrong... maybe a DifferentiableImage?)
        :return: The return value is a tuple of the form (output, loss).
        """
        if pil_image is None:
            self.use_pallet_target = False
            return
        dummy = self.clone()
        dummy.use_pallet_target = False
        dummy.encode_image(pil_image)
        with torch.no_grad():
            self.pallet_target.set_(dummy.sort_pallet())
            self.pallet.set_(self.pallet_target.clone())
            self.use_pallet_target = True

    @torch.no_grad()
    def lock_pallet(self, lock=True):
        """
        If lock is True, set the pallet_target attribute to the value of the sort_pallet method

        :param lock: If True, the pallet_target is locked to the current pallet, defaults to True
        (optional)
        """
        if lock:
            self.pallet_target.set_(self.sort_pallet().clone())
        self.use_pallet_target = lock

    def image_loss(self):
        """
        If the loss is not None, return it
        :return: A list of losses
        """
        return [x for x in [self.hdr_loss, self.loss] if x is not None]

    def sort_pallet(self):
        """
        Given a pallet of colors, sort the pallet such that the colors are sorted by their brightness
        :return: The pallet is being returned.
        """
        if self.use_pallet_target:
            return self.pallet_target
        pallet = (self.pallet / self.pallet_inertia).clamp_(0, 1)
        # https://alienryderflex.com/hsp.html
        magic_color = pallet.new_tensor([[[0.299, 0.587, 0.114]]])
        color_norms = pallet.square().mul_(magic_color).sum(dim=-1)
        pallet_indices = color_norms.argsort(dim=0).T
        pallet = torch.stack(
            [pallet[i][:, j] for j, i in enumerate(pallet_indices)], dim=1
        )
        return pallet

    def get_image_tensor(self):
        return torch.cat([self.value.unsqueeze(0), self.tensor])

    @torch.no_grad()
    def set_image_tensor(self, tensor):
        """
        Set the image tensor to the given tensor

        :param tensor: the tensor to be set
        """
        self.value.set_(tensor[0])
        self.tensor.set_(tensor[1:])

    def decode_tensor(self):
        """
        Given a tensor of shape (batch_size, n_pallets, n_values),
        returns a tensor of shape (batch_size, height, width, 3)
        where each pixel is a color from the pallet
        :return: The image with the pallet applied.
        """
        width, height = self.image_shape
        pallet = self.sort_pallet()

        # brightnes values of pixels
        values = self.value.clamp(0, 1) * (self.pallet_size - 1)
        value_floors, value_ceils, value_rounds, value_fracs = break_tensor(values)
        value_fracs = value_fracs.unsqueeze(-1).unsqueeze(-1)

        pallet_weights = self.tensor.movedim(0, 2)
        pallets = F.one_hot(pallet_weights.argmax(dim=2), num_classes=self.n_pallets)

        pallet_weights = pallet_weights.softmax(dim=2).unsqueeze(-1)
        pallets = pallets.unsqueeze(-1)

        colors_disc = pallet[value_rounds]
        colors_disc = (colors_disc * pallets).sum(dim=2)
        colors_disc = F.interpolate(
            colors_disc.movedim(2, 0)
            .unsqueeze(0)
            .to(self.device, memory_format=torch.channels_last),
            (height, width),
            mode="nearest",
        )

        colors_cont = (
            pallet[value_floors] * (1 - value_fracs) + pallet[value_ceils] * value_fracs
        )
        colors_cont = (colors_cont * pallet_weights).sum(dim=2)
        colors_cont = F.interpolate(
            colors_cont.movedim(2, 0)
            .unsqueeze(0)
            .to(self.device, memory_format=torch.channels_last),
            (height, width),
            mode="nearest",
        )
        return replace_grad(colors_disc, colors_cont * 0.5 + colors_disc * 0.5)

    @torch.no_grad()
    def render_value_image(self):
        width, height = self.image_shape
        values = self.value.clamp(0, 1).unsqueeze(-1).repeat(1, 1, 3)
        array = np.array(
            values.mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8)
        )[:, :, :]
        return Image.fromarray(array).resize((width, height), Image.NEAREST)

    @torch.no_grad()
    def render_pallet(self):
        pallet = self.sort_pallet()
        width, height = self.n_pallets * 16, self.pallet_size * 32
        array = np.array(
            pallet.mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8)
        )[:, :, :]
        return Image.fromarray(array).resize((width, height), Image.NEAREST)

    @torch.no_grad()
    def render_channel(self, pallet_i):
        """
        Given a tensor of shape (batch_size, n_pallets, height, width),
        returns a tensor of shape (batch_size, height, width, n_pallets)

        :param pallet_i: The index of the channel to render
        :return: The image.
        """
        width, height = self.image_shape
        pallet = self.sort_pallet()
        pallet[:, :pallet_i, :] = 0.5
        pallet[:, pallet_i + 1 :, :] = 0.5

        values = self.value.clamp(0, 1) * (self.pallet_size - 1)
        value_floors, value_ceils, value_rounds, value_fracs = break_tensor(values)
        value_fracs = value_fracs.unsqueeze(-1).unsqueeze(-1)

        pallet_weights = self.tensor.movedim(0, 2)
        # pallets = F.one_hot(pallet_weights.argmax(dim=2), num_classes=self.n_pallets)
        pallet_weights = pallet_weights.softmax(dim=2).unsqueeze(-1)

        colors_cont = (
            pallet[value_floors] * (1 - value_fracs) + pallet[value_ceils] * value_fracs
        )
        colors_cont = (colors_cont * pallet_weights).sum(dim=2)
        colors_cont = F.interpolate(
            colors_cont.movedim(2, 0).unsqueeze(0), (height, width), mode="nearest"
        )

        tensor = named_rearrange(colors_cont, self.output_axes, ("y", "x", "s"))
        array = np.array(
            tensor.mul(255).clamp(0, 255).cpu().detach().numpy().astype(np.uint8)
        )[:, :, :]
        return Image.fromarray(array)

    @torch.no_grad()
    def update(self):
        """
        The pallet is clamped to the pallet inertia, the value is clamped to 1, and the tensor is
        clamped to infinity
        """
        self.pallet.clamp_(0, self.pallet_inertia)
        self.value.clamp_(0, 1)
        self.tensor.clamp_(0, float("inf"))
        # self.tensor.set_(self.tensor.softmax(dim = 0))

    def encode_image(self, pil_image, smart_encode=True, device=None):
        """
        Encodes the image into a tensor.

        :param pil_image: The image to encode
        :param smart_encode: If True, the pallet will be optimized to match the image, defaults to True
        (optional)
        :param device: The device to run the model on
        """
        width, height = self.image_shape
        if device is None:
            device = self.device

        scale = self.scale
        color_ref = pil_image.resize((width // scale, height // scale), Image.LANCZOS)
        color_ref = TF.to_tensor(color_ref).to(device)
        with torch.no_grad():
            # https://alienryderflex.com/hsp.html
            magic_color = self.pallet.new_tensor([[[0.299]], [[0.587]], [[0.114]]])
            value_ref = torch.linalg.vector_norm(
                color_ref * (magic_color.sqrt()), dim=0
            )
            self.value.set_(value_ref)

        # no embedder needed without any prompts
        if smart_encode:
            comp = HSVLoss.make_comp(pil_image)
            mse = HSVLoss(
                comp=comp,
                name="HSV loss",
                image_shape=self.image_shape,
                device=device,
            )

            if self.hdr_loss is not None:
                before_weight = self.hdr_loss.weight.detach()
                self.hdr_loss.set_weight(0.01)
            # no embedder, no prompts... we don't really need an "ImageGuide" class instance here, do we?
            # We could probably optimize a DifferentiableImage object directly here. I guess maybe
            # cutouts gets applied? Wait no, there's no embedder so there's no cutouts, right?
            from pytti.ImageGuide import DirectImageGuide

            guide = DirectImageGuide(
                self, None, optimizer=optim.Adam([self.pallet, self.tensor], lr=0.1)
            )
            # why is there a magic number here?
            guide.run_steps(201, [], [], [mse])
            if self.hdr_loss is not None:
                self.hdr_loss.set_weight(before_weight)

    @torch.no_grad()
    def encode_random(self, random_pallet=False):
        """
        Sets the value and pallet to random values (uniform noise).

        :param random_pallet: If True, the pallet is initialized to random values, defaults to False
        (optional)
        """
        self.value.uniform_()
        self.tensor.uniform_()
        if random_pallet:
            self.pallet.uniform_(to=self.pallet_inertia)
