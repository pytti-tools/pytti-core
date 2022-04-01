import pytest
import os
from hydra import initialize, compose
from loguru import logger
from pytti.workhorse import _main as render_frames


def test_the_thing():
    with pytest.raises(AssertionError):
        reproduce_error()


def reproduce_error():
    # url_seattle1 = "https://image.cnbcfm.com/api/v1/image/104540684-GettyImages-530874379.jpg"
    # url_seattle2 = "https://media.cntraveler.com/photos/60480c67ff9cba52f2a91899/16:9/w_2560%2Cc_limit/01-velo-header-seattle-needle.jpg"
    # url_original = "https://i0.wp.com/digital-photography-school.com/wp-content/uploads/2018/02/DSC00500-Edit.jpg"
    # url_tiltshift = "https://i0.wp.com/digital-photography-school.com/wp-content/uploads/2018/02/DSC00500-Edit-Edit.jpg"
    # tilt_shift_str = f"'[{url_tiltshift}]:1 | [{url_original}]:-1'"

    CONFIG_BASE_PATH = "config"
    CONFIG_DEFAULTS = "default.yaml"

    settings = {
        #'init_image': #url_seattle2,
        #'init_image': f"'{url_seattle2}'", # I thought this supported URLs? Is this *another* hydra thing???
        #'init_image': f"'[{url_seattle2}]'",
        #'init_image': f'"[{url_seattle2}]"', # gdamn... let's see if it works after downloading the image...
        "init_image": "/home/dmarx/Downloads/01-velo-header-seattle-needle.jpg",  # TO DO: download this locally
        #'direct_init_weight':10,
        #'scenes':tilt_shift_str,
        "animation_mode": "Off",
        "image_model": "Unrestricted Palette",  # this should probably be default.
        "scenes": '" "',
        "learning_rate": 0.001,
        #################################################
        "steps_per_scene": "500",
        "steps_per_frame": "500",
        "save_every": "20",
        "display_every": "20",
        "file_namespace": "tiltshift_1",
        "seed": "123",
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
