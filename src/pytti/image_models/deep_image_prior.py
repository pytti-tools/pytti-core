from copy import deepcopy

from loguru import logger

from torch import optim

from pytti import clamp_with_grad
import torch
from torch import nn
from torchvision.transforms import functional as TF
from pytti.image_models import DifferentiableImage, EMAImage
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

from ema_pytorch import EMA

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


# class DeepImagePrior(EMAImage):
class DeepImagePrior(DifferentiableImage):
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
        **kwargs,
    ):
        super().__init__(width * scale, height * scale)
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
        self.net = net
        # self.tensor = self.net.params()
        self.ema = EMA(
            self.net,
            # update_every=1,
            update_every=1,
            beta=0.99,
            update_after_step=1,
            param_or_buffer_names_no_ema=[name for name, _ in self.net.named_buffers()],
        )
        self.output_axes = ("n", "s", "y", "x")
        self.scale = scale
        self.device = device

        self._net_input = torch.randn([1, input_depth, width, height], device=device)

        self.lr = lr
        self.offset_lr_fac = offset_lr_fac
        # self._params = [
        #    {'params': get_non_offset_params(net), 'lr': lr},
        #    {'params': get_offset_params(net), 'lr': lr * offset_lr_fac}
        # ]

    def update(self):
        self.ema.update()
        logger.debug(self.ema.get_current_decay())
        # pass

    def _decode_with(self, net):
        with torch.cuda.amp.autocast():
            # out = net(net_input_noised * input_scale).float()
            # logger.debug(self.net)
            # logger.debug(self._net_input.shape)
            out = net(self._net_input).float()
            # logger.debug(out.shape)
        width, height = self.image_shape
        out = F.interpolate(out, (height, width), mode="nearest")
        return clamp_with_grad(out, 0, 1)

    # def get_image_tensor(self):
    def decode_tensor(self):
        return self._decode_with(self.ema)
        # return self._decode_with(self.net)

    def decode_training_tensor(self):
        return self._decode_with(self.net)
        # return self._decode_with(self.ema)

    def get_latent_tensor(self, detach=False):
        # pass
        net = self.net
        lr = self.lr
        offset_lr_fac = self.offset_lr_fac
        params = [
            {"params": get_non_offset_params(net), "lr": lr},
            {"params": get_offset_params(net), "lr": lr * offset_lr_fac},
        ]
        # params = torch.cat(
        #    get_non_offset_params(net),
        #    get_offset_params(net)
        # )
        return params

    def clone(self):
        # dummy = super().__init__(*self.image_shape)
        # with torch.no_grad():
        #    #dummy.tensor.set_(self.tensor.clone())
        #    dummy.net.copy_(self.net)
        #    dummy.accum.set_(self.accum.clone())
        #    dummy.biased.set_(self.biased.clone())
        #    dummy.average.set_(self.average.clone())
        #    dummy.decay = self.decay
        dummy = deepcopy(self)  # maybe this is the issue? unlikely.
        return dummy

    def encode_random(self):
        pass

    @classmethod
    def get_preferred_loss(cls):
        # from pytti.LossAug.LatentLossClass import LatentLoss
        from pytti.LossAug.HSVLossClass import HSVLoss

        return HSVLoss
        # return LatentLoss # I think all that's special about this is regularizing the magnitude of the latent?
        # return DipLoss

    def encode_image(self, pil_image, device="cuda"):
        """
        Encodes the image into a tensor.

        :param pil_image: The image to encode
        :param smart_encode: If True, the pallet will be optimized to match the image, defaults to True
        (optional)
        :param device: The device to run the model on
        """
        from pytti.LossAug.HSVLossClass import HSVLoss

        width, height = self.image_shape
        scale = self.scale

        # mse = MSELoss.TargetImage("HSV loss", self.image_shape, pil_image)
        mse = HSVLoss.TargetImage("HSV loss", self.image_shape, pil_image)

        from pytti.ImageGuide import DirectImageGuide

        guide = DirectImageGuide(
            self, None, optimizer=optim.Adam(self.get_latent_tensor())
        )
        # why is there a magic number here?
        # guide.run_steps(201, [], [], [mse])
        # guide.run_steps(501, [], [], [mse])
        for _ in range(501):
            guide.run_steps(1, [], [], [mse])
            self.update()

        # here's a weird idea... reset EMA here?


from pytti.LossAug.LatentLossClass import LatentLoss
import copy

# class DipLoss(MSELoss):
class DipLoss(LatentLoss):
    def get_loss(self, input, img):
        if not self.has_latent:
            latent = img.make_latent(self.pil_image)
            latent = torch.cat(*[d["params"] for d in latent])
            with torch.no_grad():
                self.comp.set_(latent.clone().flatten())
            self.has_latent = True
        l1 = super().get_loss(img.get_latent_tensor(), img) / 2
        l2 = self.direct_loss.get_loss(input, img) / 10
        return l1 + l2
