from pytti.image_models.differentiable_image import DifferentiableImage
from pytti.image_models.ema import EMAImage
from pytti.image_models.vqgan import VQGANImage

from pathlib import Path


def test_init_model():
    models_parent_dir = "~/.cache/vqgan/"
    vqgan_model = "coco"
    model_artifacts_path = Path(models_parent_dir) / "vqgan"
    VQGANImage.init_vqgan(vqgan_model, model_artifacts_path)
    img = VQGANImage(512, 512, 1)
    # img.encode_random()


def test_init():
    obj = VQGANImage(512, 512)
    assert obj


def test_update():
    obj = VQGANImage(512, 512)
    obj.update()


def test_forward():
    obj = VQGANImage(512, 512)
    obj.forward()


def test_decode_training_tensor():
    obj = VQGANImage(512, 512)
    obj.decode_training_tensor()


def test_decode_tensor():
    obj = VQGANImage(512, 512)
    obj.decode_tensor()


def test_clone():
    obj = VQGANImage(512, 512)
    obj.clone()


def test_get_latent_tensor():
    obj = VQGANImage(512, 512)
    obj.get_latent_tensor()
