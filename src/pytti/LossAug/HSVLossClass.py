from pytti.LossAug.MSELossClass import MSELoss
import torch
from kornia.color import rgb_to_hsv
from loguru import logger


class HSVLoss(MSELoss):
    @classmethod
    def convert_input(cls, input, img):
        logger.debug(input.requires_grad)
        out = rgb_to_hsv(input)
        logger.debug(out.requires_grad)
        out = torch.cat([input, out[:, 1:, ...]], dim=1)
        logger.debug(out.requires_grad)
        return out
