from pathlib import Path
import pytest
from loguru import logger

from pytti.Image.VQGANImage import (
    VQGANImage,
    VQGAN_MODEL_NAMES,
    VQGAN_CONFIG_URLS,
    VQGAN_CHECKPOINT_URLS,
    load_vqgan_model,
)

# vqgan

"""
@pytest.mark.parametrize(
    "model_name",
    VQGAN_MODEL_NAMES,
)
def test_load_vqgan_no_download(model_name):
    VQGAN_MODEL, VQGAN_IS_GUMBEL = None, None
    #vqgan_config, vqgan_checkpoint = VQGAN_CONFIG_URLS[model_name], VQGAN_CHECKPOINT_URLS[model_name]
    vqgan_config = model_artifacts_path / f"{model_name}.yaml"
    vqgan_checkpoint = model_artifacts_path / f"{model_name}.ckpt"
    VQGAN_MODEL, VQGAN_IS_GUMBEL = load_vqgan_model(vqgan_config, vqgan_checkpoint)
    assert VQGAN_MODEL is not None
    assert VQGAN_IS_GUMBEL is not None
"""

# cannibalized from e2e tests
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
import pytest

CONFIG_BASE_PATH = "config"
CONFIG_DEFAULTS = "default.yaml"


class Test_VQGANImage:
    def load_params(self):
        with initialize(config_path=CONFIG_BASE_PATH):
            cfg = compose(
                config_name=CONFIG_DEFAULTS,
                overrides=[f"conf=_test_vqgan.yaml"],
            )
            return cfg

    @pytest.mark.parametrize(
        "model_name",
        VQGAN_MODEL_NAMES,
    )
    def test_init_vqgan(self, model_name):
        # cannibalized from workhorse phase 3
        params = self.load_params()
        model_artifacts_path = Path(params.models_parent_dir) / "vqgan"
        logger.debug(model_artifacts_path)
        # VQGANImage.init_vqgan(params.vqgan_model, model_artifacts_path)
        VQGANImage.init_vqgan(model_name, model_artifacts_path)
        img = VQGANImage(params.width, params.height, params.pixel_size)
        img.encode_random()
        assert True


########################################

# clip

from pytti.Notebook import load_clip, _sanitize_for_config


def test_load_clip_w_at_symbol_in_identifier():
    openai_model_name = "ViT-L/14@336px"
    config_model_name = "ViTL14_336px"
    assert _sanitize_for_config(openai_model_name) == config_model_name
    load_clip({config_model_name: True})


# def test_load_clip_with_download():
#    pass


# def test_load_clip_no_download():
#    pass


# adabins


# def test_load_adabins_with_download():
#    pass


# def test_load_adabins_no_download():
#    pass
