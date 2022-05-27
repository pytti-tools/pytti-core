import kornia.augmentation as K
import torch
from torch import nn
from torchvision import transforms as T


def pytti_classic():
    return nn.Sequential(
        K.RandomHorizontalFlip(p=0.3),
        K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode="border"),
        K.RandomPerspective(
            0.2,
            p=0.4,
        ),
        K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
        K.RandomErasing(
            scale=(0.1, 0.4), ratio=(0.3, 1 / 0.3), same_on_batch=False, p=0.7
        ),
        nn.Identity(),
    )


def dango():
    return T.Compose(
        [
            # T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                # scale=(0.9,0.95),
                fill=-1,
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            # T.RandomPerspective(p=1, interpolation = T.InterpolationMode.BILINEAR, fill=-1,distortion_scale=0.2),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.1),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
        ]
    )
