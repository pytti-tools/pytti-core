# This all goes down here because it'll want to import Loss from LossAug...
# ugh, these circular imports.
from pytti.LossAug.BaseLossClass import Loss
from pytti.LossAug.TVLossClass import TVLoss
from pytti.LossAug.MSELossClass import MSELoss
from pytti.LossAug.OpticalFlowLossClass import OpticalFlowLoss, TargetFlowLoss
from pytti.LossAug.DepthLossClass import DepthLoss
from pytti.LossAug.EdgeLossClass import EdgeLoss
from pytti.LossAug.LatentLossClass import LatentLoss
from pytti.LossAug.HSVLossClass import HSVLoss

# from loguru import logger
# logger.debug(dir(HSVLoss))
# logger.debug(help(HSVLoss.TargetImage))

# yeesh the ordering fragility in here...
# TO DO: let's make this a class attribute on something
LOSS_DICT = {"edge": EdgeLoss, "depth": DepthLoss}


# uh.... should the places this is beind used maybe just use Loss.__init__?
# TO DO: let's make this a class attribute on something
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
