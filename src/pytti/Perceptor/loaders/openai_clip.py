import clip
from loguru import logger
from pytti import Perceptor


def _sanitize_for_config(in_str):
    for char in ("/", "-"):
        in_str = in_str.replace(char, "")
    return in_str


SUPPORTED_CLIP_MODELS = {
    _sanitize_for_config(model_name): model_name
    for model_name in clip.available_models()
}

logger.debug(SUPPORTED_CLIP_MODELS)

# this doesn't belong here
CLIP_MODEL_NAMES = None


def load_clip(params):

    # refactor to specify this stuff in a config file
    global CLIP_MODEL_NAMES
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

    if last_names != CLIP_MODEL_NAMES or Perceptor.CLIP_PERCEPTORS is None:
        if CLIP_MODEL_NAMES == []:
            Perceptor.free_clip()
            raise RuntimeError("Please select at least one CLIP model")
        Perceptor.free_clip()
        logger.debug("Loading CLIP...")
        Perceptor.init_clip(CLIP_MODEL_NAMES)
        logger.debug("CLIP loaded.")
