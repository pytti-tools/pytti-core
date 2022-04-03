import clip
from loguru import logger

# from pytti import Perceptor
import torch

from pytti import DEVICE, vram_usage_mode

# from clip import clip


def _sanitize_for_config(in_str):
    for char in ("/", "-"):
        in_str = in_str.replace(char, "")
    return in_str


SUPPORTED_CLIP_MODELS = {
    _sanitize_for_config(model_name): model_name
    for model_name in clip.available_models()
}

logger.debug(SUPPORTED_CLIP_MODELS)


CLIP_MODEL_NAMES = None

# this is the only function that makes it out of here (into workhorse)
def load_clip(params):
    logger.debug("loading clip model(s)")
    # refactor to specify this stuff in a config file
    global CLIP_MODEL_NAMES
    global CLIP_PERCEPTORS  # FML...
    if CLIP_MODEL_NAMES is not None:
        last_names = CLIP_MODEL_NAMES
    else:
        last_names = []
    CLIP_MODEL_NAMES = []
    # this "last_names" thing is way over complicated,
    # and also a notebook-specific... pattern. deprecate this later as part of
    # cleaning up globals.

    for config_name, clip_name in SUPPORTED_CLIP_MODELS.items():
        if params.get(config_name):
            CLIP_MODEL_NAMES.append(clip_name)

    if (last_names != CLIP_MODEL_NAMES) or (
        # Perceptor.CLIP_PERCEPTORS is None
        CLIP_PERCEPTORS
        is None
    ):
        if CLIP_MODEL_NAMES == []:
            # Perceptor.free_clip()
            free_clip()
            raise RuntimeError("Please select at least one CLIP model")
        # Perceptor.free_clip()
        free_clip()
        logger.debug("Loading CLIP...")
        # Perceptor.init_clip(CLIP_MODEL_NAMES)
        init_clip(CLIP_MODEL_NAMES)
        logger.debug("CLIP loaded.")


####################################

# this global makes it out too for some reason.
# pytti.Perceptor.Embedder
# pytti.Perceptor.Prompt
#
# ...this is the container that actually holdes the loaded models
CLIP_PERCEPTORS = None

# this should probably be a method on the multiperceptor guide
@vram_usage_mode("CLIP")
def init_clip(clip_models):
    global CLIP_PERCEPTORS
    if CLIP_PERCEPTORS is None:
        CLIP_PERCEPTORS = [
            # clip.load(model, jit=False)[0]
            clip.clip.load(model, jit=False)[0]
            .eval()
            .requires_grad_(False)
            .to(DEVICE, memory_format=torch.channels_last)
            for model in clip_models
        ]


def free_clip():
    global CLIP_PERCEPTORS
    CLIP_PERCEPTORS = None
