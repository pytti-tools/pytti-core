import torch, math, gc, re
from torchvision import transforms
from torch.nn import functional as F
import requests, io
from collections import defaultdict
import pandas as pd
import numpy as np
from PIL import Image as PIL_Image
from loguru import logger

from pytti.vram_tools import (
    vram_usage_mode, 
    print_vram_usage,
    reset_vram_usage,
    freeze_vram_usage,
    vram_profiling
)

from pytti.tensor_tools import (
    named_rearrange,
    format_input,
    pad_tensor,
    cat_with_pad,
    format_module,
    to_pil,
    replace_grad,
    clamp_with_grad,
    clamp_grad,
    normalize, 
)

from pytti.eval_tools import (
    fetch,
    parametric_eval,
    parse,
    set_t
)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


__all__  = ['DEVICE', 
            'named_rearrange', 'format_input', 'pad_tensor', 'cat_with_pad', 'format_module', 'to_pil',
            'replace_grad', 'clamp_with_grad', 'clamp_grad', 'normalize', 
            'fetch', 'parse', 'parametric_eval', 'set_t', 'vram_usage_mode', 
            'print_vram_usage', 'reset_vram_usage', 'freeze_vram_usage', 'vram_profiling']
