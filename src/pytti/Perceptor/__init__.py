import torch

# from clip import clip
import mmc
from pytti import DEVICE, vram_usage_mode
from loguru import logger
from collections import namedtuple

CLIP_PERCEPTORS = None

# this should probably be a method on the multiperceptor guide
@vram_usage_mode("CLIP")
def init_clip(clip_models):
    global CLIP_PERCEPTORS
    if CLIP_PERCEPTORS is None:
        # CLIP_PERCEPTORS = [
        #    clip.load(model, jit=False)[0]
        #    .eval()
        #    .requires_grad_(False)
        #    .to(DEVICE, memory_format=torch.channels_last)
        #    for model in clip_models
        # ]
        CLIP_PERCEPTORS = []
        for model_name in clip_models:
            # doing it this way ensures we only load a single model for a given ID
            # just being careful since MLF publishes OAI models as well as their own
            for publisher in ["openai", "mlfoundations"]:
                hits = mmc.REGISTRY.find(
                    architecture="clip", publisher=publisher, id=model_name
                )
                if hits:
                    break
            logger.debug(hits)
            ldr = hits[0]
            # logger.debug(type(ldr))
            model = ldr.load(device=DEVICE)
            # logger.debug(model)
            model = model._model
            # logger.debug(model)
            # if not hasattr(model, 'visual'):
            #    model.visual = object()
            #    model.visual.input_resolution = (224,224)
            CLIP_PERCEPTORS.append(model)


def free_clip():
    global CLIP_PERCEPTORS
    CLIP_PERCEPTORS = None
