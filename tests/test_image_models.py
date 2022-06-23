import pytest
from loguru import logger
import torch

import pytti.image_models
from pytti.image_models.differentiable_image import DifferentiableImage
from pytti.image_models.ema import EMAImage
from pytti.image_models.pixel import PixelImage
from pytti.image_models.rgb_image import RGBImage
from pytti.image_models.vqgan import VQGANImage


## simple models ##


def test_differentiabble_image_model():
    """
    Test that the DifferentiableImage can be instantiated
    """
    image = DifferentiableImage(
        width=10,
        height=10,
    )
    logger.debug(image.output_axes)  # x y s
    assert image


def test_rgb_image_model():
    """
    Test that the RGBImage can be instantiated
    """
    image = RGBImage(
        width=10,
        height=10,
    )
    logger.debug(image.output_axes)  # n x y s ... when does n != 1?
    assert image


## more complex models ##


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
    logger.debug(image.output_axes)  # x y s
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
    logger.debug(image.output_axes)  # n s y x ... uh ok, sure.
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
#     logger.debug(image.output_axes)
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
        logger.debug(image.output_axes)
