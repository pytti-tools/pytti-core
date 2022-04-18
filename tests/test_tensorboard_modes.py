import pytest
from hydra import initialize, compose
from loguru import logger
from pytti.workhorse import _main as render_frames


def test_tensorboard_off():

    CONFIG_BASE_PATH = "config"
    CONFIG_DEFAULTS = "default.yaml"

    settings = {
        "scenes": '" "',
        #################################################
        "steps_per_scene": "15",
        "steps_per_frame": "15",
        "save_every": "5",
        "display_every": "5",
        "use_tensorboard": False,
        #################################################
        "conf": "empty",
    }

    with initialize(config_path=CONFIG_BASE_PATH):
        cfg = compose(
            config_name=CONFIG_DEFAULTS,
            overrides=[f"{k}={v}" for k, v in settings.items()],
        )
        render_frames(cfg)


def test_tensorboard_on():

    CONFIG_BASE_PATH = "config"
    CONFIG_DEFAULTS = "default.yaml"

    settings = {
        "scenes": '" "',
        #################################################
        "steps_per_scene": "15",
        "steps_per_frame": "15",
        "save_every": "5",
        "display_every": "5",
        "use_tensorboard": True,
        #################################################
        "conf": "empty",
    }

    with initialize(config_path=CONFIG_BASE_PATH):
        cfg = compose(
            config_name=CONFIG_DEFAULTS,
            overrides=[f"{k}={v}" for k, v in settings.items()],
        )
        render_frames(cfg)
