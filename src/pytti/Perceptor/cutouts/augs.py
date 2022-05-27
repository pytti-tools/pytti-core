import kornia.augmentation as K
from torch import nn


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
