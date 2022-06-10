import torch
from clip import clip
from pytti import vram_usage_mode

CLIP_PERCEPTORS = None

# this should probably be a method on the multiperceptor guide
@vram_usage_mode("CLIP")
def init_clip(clip_models, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global CLIP_PERCEPTORS
    if CLIP_PERCEPTORS is None:
        CLIP_PERCEPTORS = [
            clip.load(model, jit=False)[0]
            .eval()
            .requires_grad_(False)
            .to(device, memory_format=torch.channels_last)
            for model in clip_models
        ]


def free_clip():
    global CLIP_PERCEPTORS
    CLIP_PERCEPTORS = None
