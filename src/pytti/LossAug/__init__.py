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
