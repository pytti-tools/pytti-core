import torch
from torch import nn
from pytti.image_models.differentiable_image import DifferentiableImage


class EMAImage(DifferentiableImage):
    """
    Base class for differentiable images with Exponential Moving Average filtering
    Based on code by Katherine Crowson
    """

    def __init__(self, width, height, tensor, decay):
        super().__init__(width, height)
        # self.representation_parameters = nn.Parameter(tensor)
        self.register_buffer("biased", torch.zeros_like(tensor))
        self.register_buffer("average", torch.zeros_like(tensor))
        self.decay = decay
        self.register_buffer("accum", torch.tensor(1.0))
        self.update()

    @torch.no_grad()
    def update(self):
        if not self.training:
            raise RuntimeError("update() should only be called during training")
        self.accum.mul_(self.decay)
        self.biased.mul_(self.decay)
        self.biased.add_((1 - self.decay) * self.representation_parameters)
        self.average.copy_(self.biased)
        self.average.div_(1 - self.accum)

    @torch.no_grad()
    def reset(self):
        if not self.training:
            raise RuntimeError("reset() should only be called during training")
        self.biased.set_(torch.zeros_like(self.biased))
        self.average.set_(torch.zeros_like(self.average))
        self.accum.set_(torch.ones_like(self.accum))
        self.update()

    def decode_training_tensor(self):
        return self.decode(self.representation_parameters)

    def decode_tensor(self):
        return self.decode(self.average)

    def decode(self, tensor):
        raise NotImplementedError
