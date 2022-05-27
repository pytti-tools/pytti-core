import pytest
from loguru import logger
import torch

import pytti.image_models
from pytti.image_models.differentiable_image import DifferentiableImage
from pytti.image_models.ema import EMAImage
from pytti.image_models.pixel import PixelImage
from pytti.image_models.rgb_image import RGBImage
from pytti.image_models.vqgan import VQGANImage


@pytest.mark.parametrize(
    "ImageModel",
    [DifferentiableImage, RGBImage],
)
def test_simple_image_models(ImageModel):
    """
    Test that the image models can be instantiated
    """
    image = ImageModel(
        width=10,
        height=10,
    )
    assert image


def test_ema_image():
    """
    Test that the EMAImage can be instantiated
    """
    image = EMAImage(
        width=10,
        height=10,
        tensor=torch.zeros(10, 10),
        decay=0.5,
    )
    assert image


def test_pixel_image():
    """
    Test that the PixelImage can be instantiated
    """
    image = PixelImage(
        width=10,
        height=10,
        scale=1,
        pallet_size=1,
        n_pallets=1,
    )
    assert image


# def test_vqgan_image_valid():
#     """
#     Test that the VQGANImage can be instantiated
#     """
#     image = VQGANImage(
#         width=10,
#         height=10,
#         model=SOME_VQGAN_MODEL,
#     )
#     assert image


def test_vqgan_image_invalid_string():
    """
    Test that the VQGANImage can be instantiated
    """
    with pytest.raises(AttributeError):
        image = VQGANImage(
            width=10,
            height=10,
            model="this isn't actually a valid value for this field",
        )
