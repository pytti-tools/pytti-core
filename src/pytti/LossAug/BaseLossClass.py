import torch
from torch import nn
from pytti import replace_grad, parametric_eval


class Loss(nn.Module):
    def __init__(self, weight, stop, name, device=None):
        super().__init__()
        # self.register_buffer('weight', torch.as_tensor(weight))
        # self.register_buffer('stop', torch.as_tensor(stop))
        self.weight = weight
        self.stop = stop
        self.input_axes = ("n", "s", "y", "x")
        self.name = name
        self.enabled = True
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def get_loss(self, input, img):
        raise NotImplementedError

    def set_enabled(self, enabled):
        self.enabled = enabled

    def __str__(self):
        return self.name

    def forward(self, input, img, device=None):
        if not self.enabled or self.weight in [0, "0"]:
            return 0, 0
        if device is None:
            device = self.device
        weight = torch.as_tensor(parametric_eval(self.weight), device=device)
        stop = torch.as_tensor(parametric_eval(self.stop), device=device)
        loss_raw = self.get_loss(input, img)
        loss = loss_raw * weight.sign()
        return weight.abs() * replace_grad(loss, torch.maximum(loss, stop)), loss_raw
