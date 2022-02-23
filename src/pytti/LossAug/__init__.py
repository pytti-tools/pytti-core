from PIL import Image
import torch
from torch import nn
from torchvision.transforms import functional as TF

from pytti import DEVICE, replace_grad, parametric_eval


class Loss(nn.Module):
    def __init__(self, weight, stop, name):
        super().__init__()
        # self.register_buffer('weight', torch.as_tensor(weight))
        # self.register_buffer('stop', torch.as_tensor(stop))
        self.weight = weight
        self.stop = stop
        self.input_axes = ("n", "s", "y", "x")
        self.name = name
        self.enabled = True

    def get_loss(self, input, img):
        raise NotImplementedError

    def set_enabled(self, enabled):
        self.enabled = enabled

    def set_weight(weight):
        self.weight = weight

    def set_stop(stop):
        self.stop = stop

    def __str__(self):
        return self.name

    def forward(self, input, img, device=DEVICE):
        if not self.enabled or self.weight in [0, "0"]:
            return 0, 0

        weight = torch.as_tensor(parametric_eval(self.weight), device=device)
        stop = torch.as_tensor(parametric_eval(self.stop), device=device)
        loss_raw = self.get_loss(input, img)
        loss = loss_raw * weight.sign()
        return weight.abs() * replace_grad(loss, torch.maximum(loss, stop)), loss_raw


# This all goes down here because it'll want to import Loss from LossAug...
# ugh, these circular imports.
from pytti.LossAug.TVLoss import TVLoss
from pytti.LossAug.MSELoss import MSELoss
from pytti.LossAug.OpticalFlowLoss import OpticalFlowLoss, TargetFlowLoss
from pytti.LossAug.DepthLoss import DepthLoss
from pytti.LossAug.EdgeLoss import EdgeLoss
from pytti.LossAug.LatentLoss import LatentLoss
from pytti.LossAug.HSVLoss import HSVLoss

# yeesh the ordering fragility in here...
LOSS_DICT = {"edge": EdgeLoss, "depth": DepthLoss}


# uh.... should the places this is beind used maybe just use Loss.__init__?
def build_loss(weight_name, weight, name, img, pil_target) -> Loss:
    """
    Given a weight name, weight, name, image, and target image, returns a loss object

    :param weight_name: The name of the loss function
    :param weight: The weight of the loss
    :param name: The name of the loss function
    :param img: The image to be optimized
    :param pil_target: The target image
    :return: The loss function.
    """

    weight_name, suffix = weight_name.split("_", 1)
    if weight_name == "direct":
        Loss = type(img).get_preferred_loss()
    else:
        Loss = LOSS_DICT[weight_name]
    out = Loss.TargetImage(
        f"{weight_name} {name}:{weight}", img.image_shape, pil_target
    )
    out.set_enabled(pil_target is not None)
    return out
