from hydra import initialize, compose
from omegaconf import OmegaConf
import pytest
from pytti.config import structured_config


def test_initialization_of_default_structured_config():
    with initialize(config_path="config"):
        cfg = compose(
            config_name="_structured_config",
        )
    OmegaConf.to_object(cfg)


def test_acceptance_of_overwrite_with_valid_config():
    with initialize(config_path="config"):
        cfg = compose(
            config_name="_structured_config",
            overrides=[
                "+conf=_test_structured_config/_valid_animation"
            ],  # animation_mode = 2D
        )
    OmegaConf.to_object(cfg)


def test_rejection_of_overwrite_with_invalid_config():
    with initialize(config_path="config"):
        cfg = compose(
            config_name="_structured_config",
            overrides=[
                "+conf=_test_structured_config/_invalid_animation"
            ],  # animation_mode = 1D
        )
    with pytest.raises(ValueError):
        OmegaConf.to_object(cfg)
