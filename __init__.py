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

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



math_env = None
global_t = 0
eval_memo = {}
def parametric_eval(string, **vals):
  global math_env
  if string in eval_memo:
    return eval_memo[string]
  if isinstance(string, str):
    if math_env is None:
      math_env = {'abs':abs, 'max':max, 'min':min, 'pow':pow, 'round':round, '__builtins__': None}
      math_env.update({key: getattr(math, key) for key in dir(math) if '_' not in key})
    math_env.update(vals)
    math_env['t'] = global_t
    try:
      output = eval(string, math_env)
    except SyntaxError as e:
      raise RuntimeError('Error in parametric value ' + string)
    eval_memo[string] = output
    return output
  else:
    return string

def set_t(t):
  global global_t, eval_memo
  global_t = t
  eval_memo = {}

def fetch(url_or_path):
  if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
    r = requests.get(url_or_path)
    r.raise_for_status()
    fd = io.BytesIO()
    fd.write(r.content)
    fd.seek(0)
    return fd
  return open(url_or_path, 'rb')


def parse(string, split, defaults):
  tokens = re.split(split, string, len(defaults)-1)
  tokens = tokens+defaults[len(tokens):]
  return tokens

def to_pil(tensor, image_shape = None):
  h, w = tensor.shape[-2:]
  if tensor.dim() == 2:
    tensor = tensor.unsqueeze(0).unsqueeze(0).expand(1,3,h,w)
  elif tensor.dim() == 3:
    tensor = tensor.unsqueeze(0).expand(1,3,h,w)
  pil_image = PIL_Image.fromarray(tensor.squeeze(0).movedim(0,-1).mul(255).clamp(0,255).detach().cpu().numpy().astype(np.uint8))
  if image_shape is not None:
    pil_image = pil_image.resize(image_shape, PIL_Image.LANCZOS)
  return pil_image

__all__  = ['DEVICE', 
            'named_rearrange', 'format_input', 'pad_tensor', 'cat_with_pad', 'format_module', 'to_pil',
            'replace_grad', 'clamp_with_grad', 'clamp_grad', 'normalize', 
            'fetch', 'parse', 'parametric_eval', 'set_t', 'vram_usage_mode', 
            'print_vram_usage', 'reset_vram_usage', 'freeze_vram_usage', 'vram_profiling']
