from copy import deepcopy

from loguru import logger

from torch import optim

from pytti import clamp_with_grad
import torch
from torch import nn
from torchvision.transforms import functional as TF

# from pytti.image_models import DifferentiableImage
from pytti.image_models.ema import EMAImage, EMAParametersDict
from PIL import Image
from torch.nn import functional as F

from pytti.LossAug.MSELossClass import MSELoss

# scavenging code snippets from:
# - https://github.com/LAION-AI/notebooks/blob/main/DALLE2-Prior%2BDeep-Image-Prior.ipynb

import deep_image_prior
import deep_image_prior.models
from deep_image_prior.models import (
    get_hq_skip_net,
    get_non_offset_params,
    get_offset_params,
)

# from deep_image_prior import get_hq_skip_net, get_non_offset_params, get_offset_params

# foo = deep_image_prior.models


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


class DeepImagePrior(EMAImage):
    # class DeepImagePrior(DifferentiableImage):
    """
    https://github.com/nousr/deep-image-prior/
    """

    def __init__(
        self,
        width,
        height,
        scale=1,
        ###########
        input_depth=32,
        num_scales=7,
        offset_type="none",
        # offset_groups=1,
        disable_deformable_convolutions=False,
        lr=1e-3,
        offset_lr_fac=0.1,  # 1.0,
        ###########
        ema_val=0.99,
        ###########
        device="cuda",
        image_encode_steps=30,  # 500, # setting this low for prototyping.
        **kwargs,
    ):
        # super(super(EMAImage)).__init__()
        nn.Module.__init__(self)
        super().__init__(
            width=width * scale,
            height=height * scale,
            decay=ema_val,
            device=device,
        )
        net = load_dip(
            input_depth=input_depth,
            num_scales=num_scales,
            offset_type=offset_type,
            offset_groups=0 if disable_deformable_convolutions else 4,
            device=device,
        )
        # z = self.get_latent_tensor()
        # params = [
        #    {'params': get_non_offset_params(net), 'lr': lr},
        #    {'params': get_offset_params(net), 'lr': lr * offset_lr_fac}
        # ]
        # z = torch.cat(get_non_offset_params(net), get_offset_params(net))
        # logger.debug(z.shape)
        # super().__init__(width * scale, height * scale, z, ema_val)
        # self.net = net
        # self.tensor = self.net.params()
        self.output_axes = ("n", "s", "y", "x")
        self.scale = scale
        self.device = device
        self.image_encode_steps = image_encode_steps

        # self._net_input = torch.randn([1, input_depth, width, height], device=device)

        self.lr = lr
        self.offset_lr_fac = offset_lr_fac
        # self._params = [
        #    {'params': get_non_offset_params(net), 'lr': lr},
        #    {'params': get_offset_params(net), 'lr': lr * offset_lr_fac}
        # ]
        # z = {
        #    'non_offset':get_non_offset_params(net),
        #    'offset':get_offset_params(net),
        # }
        self.net = net
        self._net_input = torch.randn([1, input_depth, width, height], device=device)

        # I think this is the attribute I want to use for "comp" in the latent loss
        self.image_representation_parameters = EMAParametersDict(
            z=self.net, decay=ema_val, device=device
        )

        # super().__init__(
        #    width = width * scale,
        #    height = height * scale,
        #    tensor = z,
        #    decay = ema_val,
        #    device=device,
        # )

    # def get_image_tensor(self):
    def decode_tensor(self):
        with torch.cuda.amp.autocast():
            # out = net(net_input_noised * input_scale).float()
            # logger.debug(self.net)
            # logger.debug(self._net_input.shape)
            out = self.net(self._net_input).float()
            # logger.debug(out.shape)
        width, height = self.image_shape
        out = F.interpolate(out, (height, width), mode="nearest")
        return clamp_with_grad(out, 0, 1)
        # return out

    def get_latent_tensor(self, detach=False):
        # this will get used as the "comp" downstream
        # pass
        net = self.net
        lr = self.lr
        offset_lr_fac = self.offset_lr_fac
        # params = self.image_representation_parameters._container
        # params = [
        #    {"params": get_non_offset_params(net), "lr": lr},
        #    {"params": get_offset_params(net), "lr": lr * offset_lr_fac},
        # ]
        # params = torch.cat(
        #    get_non_offset_params(net),
        #    get_offset_params(net)
        # )
        # return params
        # return self.net.params()
        # return self.net.parameters()
        # return self.image_representation_parameters # throws error from LatentLossClass.get_loss() --> self.comp.set_(latent.clone())
        return self.representation_parameters

    def clone(self) -> "DeepImagePrior":
        # dummy = VQGANImage(*self.image_shape)
        # with torch.no_grad():
        #     dummy.representation_parameters.set_(self.representation_parameters.clone())
        #     dummy.accum.set_(self.accum.clone())
        #     dummy.biased.set_(self.biased.clone())
        #     dummy.average.set_(self.average.clone())
        #     dummy.decay = self.decay
        # return dummy
        dummy = DeepImagePrior(*self.image_shape)
        with torch.no_grad():
            # dummy.representation_parameters.set_(self.representation_parameters.clone())
            dummy.image_representation_parameters.set_(
                self.image_representation_parameters.clone()
            )
        return dummy  # output of this function is expected to have an encode_image() method
        # return dummy.image_representation_parameters

    # def clone(self):
    #     # dummy = super().__init__(*self.image_shape)
    #     # with torch.no_grad():
    #     #    #dummy.tensor.set_(self.tensor.clone())
    #     #    dummy.net.copy_(self.net)
    #     #    dummy.accum.set_(self.accum.clone())
    #     #    dummy.biased.set_(self.biased.clone())
    #     #    dummy.average.set_(self.average.clone())
    #     #    dummy.decay = self.decay
    #     dummy = deepcopy(self)
    #     return dummy

    def encode_random(self):
        pass

    @classmethod
    def get_preferred_loss(cls):
        from pytti.LossAug.LatentLossClass import LatentLoss

        return LatentLoss

    def make_latent(self, pil_image):
        """
        Takes a PIL image as input,
        encodes it appropriately to the image representation (via .encode_image(pil_image)),
        and returns the output of .get_latent_tensor(detach=True).

        NB: default behavior of .get_latent_tensor() is to just return the output of .get_image_tensor()
        """
        try:
            dummy = self.clone()
        except NotImplementedError:
            dummy = copy.deepcopy(self)
        dummy.encode_image(pil_image)
        # return dummy.get_latent_tensor(detach=True)
        return dummy.image_representation_parameters

    @classmethod
    def default_comp(*args, **kargs):
        device = kargs.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        net = load_dip(
            input_depth=32,
            num_scales=7,
            offset_type="none",
            offset_groups=4,
            device=device,
        )
        return EMAParametersDict(z=net, decay=0.99, device=device)

    def encode_image(self, pil_image, device="cuda"):
        """
        Encodes the image into a tensor.

        :param pil_image: The image to encode
        :param smart_encode: If True, the pallet will be optimized to match the image, defaults to True
        (optional)
        :param device: The device to run the model on
        """
        width, height = self.image_shape
        scale = self.scale

        mse = MSELoss.TargetImage("MSE loss", self.image_shape, pil_image)

        from pytti.ImageGuide import DirectImageGuide

        params = [
            {"params": get_non_offset_params(self.net), "lr": self.lr},
            {"params": get_offset_params(self.net), "lr": self.lr * self.offset_lr_fac},
        ]

        guide = DirectImageGuide(
            self,
            None,
            optimizer=optim.Adam(
                # self.get_latent_tensor()
                params
            ),
        )
        # why is there a magic number here?
        guide.run_steps(self.image_encode_steps, [], [], [mse])