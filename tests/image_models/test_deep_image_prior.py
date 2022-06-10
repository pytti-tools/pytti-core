from pytti.image_models.differentiable_image import DifferentiableImage
from pytti.image_models.ema import EMAImage
from pytti.image_models.pixel import PixelImage
from pytti.image_models.rgb_image import RGBImage
from pytti.image_models.vqgan import VQGANImage
from pytti.image_models.deep_image_prior import DeepImagePrior

### DIP ###


def test_dip_init():
    obj = DeepImagePrior(512, 512)
    assert obj


def test_dip_update():
    obj = DeepImagePrior(512, 512)
    obj.update()


def test_dip_forward():
    obj = DeepImagePrior(512, 512)
    obj.forward()


def test_dip_decode_training_tensor():
    obj = DeepImagePrior(512, 512)
    obj.decode_training_tensor()


def test_dip_decode_tensor():
    obj = DeepImagePrior(512, 512)
    obj.decode_tensor()


def test_dip_clone():
    obj = DeepImagePrior(512, 512)
    obj.clone()


def test_dip_get_latent_tensor():
    obj = DeepImagePrior(10, 10)
    obj.get_latent_tensor()
