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

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def named_rearrange(tensor, axes, new_positions):
  """
  Permute and unsqueeze tensor to match target dimensional arrangement
  tensor:        (Tensor) input
  axes:          (string tuple) names of dimensions in tensor
  new_positions: (string tuple) names of dimensions in result
                 optionally including new names which will be unsqueezed into singleton dimensions
  """
  #this probably makes it slower honestly
  if axes == new_positions:
    return tensor
  #list to dictionary pseudoinverse
  axes = {k:v for v,k in enumerate(axes)}
  #squeeze axes that need to be gone
  missing_axes = [d for d in axes if d not in new_positions]
  for d in missing_axes:
    dim = axes[d]
    if tensor.shape[dim] != 1:
      raise ValueError(f"Can't convert tensor of shape {tensor.shape} due to non-singelton axis {d} (dim {dim})")
    tensor = tensor.squeeze(axes[d])
    del axes[d]
    axes.update({k:v-1 for k,v in axes.items() if v > dim})
  #add singleton dimensions for missing axes
  extra_axes = [d for d in new_positions if d not in axes]
  for d in extra_axes:
    tensor = tensor.unsqueeze(-1)
    axes[d] = tensor.dim()-1
  #permute to match output
  permutation = [axes[d] for d in new_positions]
  return tensor.permute(*permutation)

def format_input(tensor, source, dest):
  return named_rearrange(tensor, source.output_axes, dest.input_axes)

def pad_tensor(tensor, target_len):
  l = tensor.shape[-1]
  if l >= target_len:
    return tensor
  return F.pad(tensor, (0,target_len-l))

def cat_with_pad(tensors):
  max_size = max(t.shape[-1] for t in tensors)
  return torch.cat([pad_tensor(t, max_size) for t in tensors])

def format_module(module, dest, *args, **kwargs):
  output = module(*args, **kwargs)
  if isinstance(output, tuple):
    output = output[0]
  return format_input(output, module, dest)

class ReplaceGrad(torch.autograd.Function):
  """
  returns x_forward during forward pass, but evaluates derivates as though
  x_backward was retruned instead.
  """
  @staticmethod
  def forward(ctx, x_forward, x_backward):
    ctx.shape = x_backward.shape
    return x_forward
  @staticmethod
  def backward(ctx, grad_in):
    return None, grad_in.sum_to_size(ctx.shape)
replace_grad = ReplaceGrad.apply

class ClampWithGrad(torch.autograd.Function):
  """
  clamp function
  """
  @staticmethod
  def forward(ctx, input, min, max):
    ctx.min = min
    ctx.max = max
    ctx.save_for_backward(input)
    return input.clamp(min, max)
  @staticmethod
  def backward(ctx, grad_in):
    input, = ctx.saved_tensors
    return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None
clamp_with_grad = ClampWithGrad.apply

def clamp_grad(input, min, max):
  return replace_grad(input.clamp(min,max), input)

normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

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
