from collections import defaultdict
import gc
import torch
from loguru import logger

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

track_vram = False
usage_mode = 'Unknown'
prev_usage = torch.cuda.memory_allocated(device = DEVICE)
usage_dict = defaultdict(lambda:0)
usage_frozen = defaultdict(lambda:False)

def vram_profiling(enabled):
  global track_vram
  track_vram = enabled

def reset_vram_usage():
  global prev_usage, usage_dict, usage_mode, usage_frozen
  if not track_vram:
    return
  if usage_dict:
    logger.warning('WARNING: VRAM tracking does not work more than once per session. Select `Runtime > Restart runtime` for accurate VRAM usage.')
  usage_mode = 'Unknown'
  prev_usage = torch.cuda.memory_allocated(device = DEVICE)
  usage_dict = defaultdict(lambda:0)
  usage_frozen = defaultdict(lambda:False)

def set_usage_mode(new_mode, force_update = False):
  global usage_mode, prev_usage, usage_dict
  if not track_vram:
    return
  if (usage_mode != new_mode or force_update):
    if not usage_frozen[usage_mode]:
      gc.collect()
      torch.cuda.empty_cache()
      gc.collect()
      torch.cuda.empty_cache()
      current_usage = torch.cuda.memory_allocated(device = DEVICE)
      delta = current_usage - prev_usage
      if delta < 0 and usage_mode != 'Unknown':
        logger.warning('WARNING:',usage_mode, 'has negavive delta of',delta)

      usage_dict[usage_mode] += delta
      prev_usage = current_usage
    usage_mode = new_mode

def freeze_vram_usage(mode = None):
  if not track_vram:
    return
  global usage_mode, usage_frozen
  mode = usage_mode if mode is None else mode
  if not usage_frozen[mode]:
    set_usage_mode(usage_mode, force_update = True)
    usage_frozen[mode] = True

class vram_usage_mode:
  def __init__(self, mode):
    self.mode = mode
  def __call__(self, func):
    global usage_mode
    def wrapper(*args, **kwargs):
      cached_mode = usage_mode
      set_usage_mode(self.mode)
      output = func(*args, **kwargs)
      set_usage_mode(cached_mode)
      return output
    return wrapper
  def __enter__(self):
    global usage_mode
    self.cached_mode = usage_mode
    set_usage_mode(self.mode)
  def __exit__(self, type, value, traceback):
    set_usage_mode(self.cached_mode)

def print_vram_usage():
  global usage_mode
  if not track_vram:
    return
  set_usage_mode(usage_mode, force_update = True)
  total = sum(usage_dict.values()) - usage_dict['Unknown']
  usage_dict['Unknown'] = torch.cuda.memory_allocated(device = DEVICE) - total
  for k,v in usage_dict.items():
    if v < 1000:
      logger.info(f'{k}:',f'{v}B')
    elif v < 1000000:
      logger.info(f'{k}:',f'{v/1000:.2f}kB')
    elif v < 1000000000:
      logger.info(f'{k}:',f'{v/1000000:.2f}MB')
    else:
      logger.info(f'{k}:',f'{v/1000000000:.2f}GB')
  
  logger.info('Total:',f'{total/1000000000:.2f}GB')
  if total != 0:
    overhead = (torch.cuda.max_memory_allocated(device = DEVICE) - total)/total
    logger.info(f'Overhead: {overhead*100:.2f}%')
