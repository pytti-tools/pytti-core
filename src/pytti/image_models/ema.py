import torch
from torch import nn
from pytti.image_models.differentiable_image import (
    DifferentiableImage,
    ImageRepresentationalParameters,
)


class EMATensor(nn.Module):
    """implmeneted by Katherine Crowson"""

    def __init__(self, tensor, decay):
        super().__init__()
        self.tensor = nn.Parameter(tensor)
        self.register_buffer("biased", torch.zeros_like(tensor))
        self.register_buffer("average", torch.zeros_like(tensor))
        self.decay = decay
        self.register_buffer("accum", torch.tensor(1.0))
        self.update()

    @torch.no_grad()
    def update(self):
        if not self.training:
            raise RuntimeError("update() should only be called during training")

        self.accum *= self.decay
        self.biased.mul_(self.decay)
        self.biased.add_((1 - self.decay) * self.tensor)
        self.average.copy_(self.biased)
        self.average.div_(1 - self.accum)

    def forward(self):
        if self.training:
            return self.tensor
        return self.average

    def clone(self):
        new = EMATensor(self.tensor.clone(), self.decay)
        new.accum.copy_(self.accum)
        new.biased.copy_(self.biased)
        new.average.copy_(self.average)
        return new

    def set_(self, other):
        self.tensor.set_(other.tensor)
        self.accum.set_(other.accum)
        self.biased.set_(other.biased)
        self.average.set_(other.average)
        # self.update()


class EMAParametersDict(ImageRepresentationalParameters):
    """
    LatentTensor with a singleton dimension for the EMAParameters
    """

    def __init__(self, z=None, decay=0.99, device=None):
        super(ImageRepresentationalParameters).__init__()
        self.decay = decay
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._container = self._new(z)

    def _new(self, z=None):
        if z is None:
            # I think this can all go in the constructor and doesn't need to call .to()
            z = torch.zeros(1, 3, self.height, self.width).to(
                device=self.device, memory_format=torch.channels_last
            )
        # d_ = z
        d_ = {}
        if isinstance(z, EMAParametersDict):
            for k, v in z.items():
                d_[k] = EMATensor(v.tensor, self.decay)
        elif isinstance(z, EMATensor):
            d_["z"] = z.clone()
        elif hasattr(z, "named_parameters"):
            d_ = {
                name: EMATensor(param, decay=self.decay)
                for name, param in z.named_parameters()
            }
        elif isinstance(z, dict):
            print(type(z))
            for k, v in z.items():
                # print(k)
                # print(type(v))
                # d_[k] = EMATensor(v, self.decay)
                # d_[k] = EMATensor(v.clone(), self.decay)
                if isinstance(v, EMATensor):
                    d_[k] = v.clone()
        elif isinstance(z, torch.Tensor):
            d_["z"] = EMATensor(z, decay=self.decay)
        else:
            raise ValueError(
                "z must be a dict, torch.nn.Module, EMATensor, EMAParametersDict, or torch.Tensor"
            )

        # if not isinstance(z, dict):
        ##if hasattr(z, "named_parameters"):
        ##    d_ = {name: EMATensor(param, decay=self.decay) for name, param in z.named_parameters()}
        # else:
        #    d_['z'] = EMATensor(z, decay=self.decay)
        # else:
        #    #d_ = {name: EMATensor(param, self.decay) for name, param in z.items()}
        #    try:
        #        d_ = {name: EMATensor(param.data, self.decay) for name, param in z.items()}
        #    except AttributeError:
        #        d_ = {name: EMATensor(param, self.decay) for name, param in z.items()}
        return d_

    def clone(self):
        # d_ = {k: v.clone() for k, v in self._container.items()}
        d_ = {k: v.clone() for k, v in self._container.items()}
        return EMAParametersDict(z=d_, decay=self.decay, device=self.device)

    def update(self):
        for param in self._container.values():
            param.update()

    @property
    def average(self):
        return {k: v.average for k, v in self._container.items()}

    def set_(self, d):
        d_ = d
        if isinstance(d, EMAParametersDict):
            d_ = d._container
        for k, v in d_.items():
            self._container[k].set_(v)
            # self._container[k].tensor.set_(v)
            # self._container[k].tensor.set_(v.tensor)
            # self._container[k].tensor.set_(v.tensor)


class EMAImage(DifferentiableImage):
    def __init__(self, width, height, tensor, decay, device=None):
        super().__init__(width=width, height=height, device=device)
        self.image_representation_parameters = EMAParametersDict(
            z=tensor, decay=decay, device=device
        )


class LatentTensor(EMAImage):
    pass


class EMAImage_old(DifferentiableImage):
    """
    Base class for differentiable images with Exponential Moving Average filtering
    Based on code by Katherine Crowson
    """

    def __init__(self, width, height, tensor, decay):
        # super().__init__(width, height)
        super().__init__(width=width, height=height, z=tensor)
        # self.representation_parameters = nn.Parameter(tensor)
        # self.image_representation_parameters._container = nn.Parameter(tensor)
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
