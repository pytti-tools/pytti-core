# would be better if Task.init() was called inside the hydra app
USE_CLEARML = True
try:
    from clearml import Task
    task = Task.init(
        project_name="PYTTI",
        task_name="cli test",
        reuse_last_task_id=False
    )
    #task.execute_remotely(queue_name="art")
except ImportError:
    USE_CLEARML = False
    
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

import os, sys

# I really hate this...
# make this a commandline arg or maybe even an environment variable
TASK_IS_LOCAL=True
if USE_CLEARML:
    if not task.running_locally():
        os.chdir('/opt/colab')
        TASK_IS_LOCAL=False

# fix path for our shitty imports.
if TASK_IS_LOCAL:
    cwd = os.getcwd()
    sys.path.append(f'{cwd}/GMA/core')
    sys.path.append(f'{cwd}/pytti')
    

print(os.getcwd())
        
#auto_connect_arg_parser=True,

# TO DO: 
# https://github.com/allegroai/clearml/blob/master/examples/frameworks/hydra/hydra_example.py

def outdir_from_clearml_task(task):
    ts = task.data.created.strftime("%Y-%m-%d_%H-%M-%S")
    path = f"{task.project}/{ts}_{task.id}"
    return path

OUTPREFIX = 'foobar'
if USE_CLEARML:
    OUTPREFIX = outdir_from_clearml_task(task)
#OUTPATH = f"images_out/{OUTPREFIX}"
#OUTTERPATH = f"/opt/projdata/{OUTPREFIX}"

OUTPATH = f"{os.getcwd()}/images_out/{OUTPREFIX}"
if not TASK_IS_LOCAL:
    OUTPATH = f"/opt/colab/images_out/{OUTPREFIX}"
OUTTERPATH = OUTPATH

#import json
from bunch import Bunch
#with open('default_params.json','r') as f:
#    #default_params = Bunch(json.load(f))
#    default_params = json.load(f)

#####################

from pathlib import Path

import torch

from os.path import exists as path_exists
#if path_exists('/content/drive/MyDrive/pytti_test'):
#  %cd /content/drive/MyDrive/pytti_test
#  drive_mounted = True
#else:
#  drive_mounted = False
drive_mounted = False
try:
  from pytti.Notebook import *
except ModuleNotFoundError:
  if drive_mounted:
    #THIS IS NOT AN ERROR. This is the code that would
    #make an error if something were wrong.
    raise RuntimeError('ERROR: please run setup (step 1.3).')
  else:
    #THIS IS NOT AN ERROR. This is the code that would
    #make an error if something were wrong.
    raise RuntimeError('WARNING: drive is not mounted.\nERROR: please run setup (step 1.3).')
change_tqdm_color()
import sys
sys.path.append('./AdaBins')
print(sys.path)

try:
  from pytti import Perceptor
except ModuleNotFoundError:
  if drive_mounted:
    #THIS IS NOT AN ERROR. This is the code that would
    #make an error if something were wrong.
    raise RuntimeError('ERROR: please run setup (step 1.3).')
  else:
    #THIS IS NOT AN ERROR. This is the code that would
    #make an error if something were wrong.
    raise RuntimeError('WARNING: drive is not mounted.\nERROR: please run setup (step 1.3).')
print("Loading pytti...")
from pytti.Image import PixelImage, RGBImage, VQGANImage
from pytti.ImageGuide import DirectImageGuide
from pytti.Perceptor.Embedder import HDMultiClipEmbedder
from pytti.Perceptor.Prompt import parse_prompt
from pytti.LossAug import TVLoss, HSVLoss, OpticalFlowLoss, TargetFlowLoss
from pytti.Transforms import zoom_2d, zoom_3d, apply_flow
from pytti import *
from pytti.LossAug.DepthLoss import init_AdaBins
print("pytti loaded.")

import torch, gc, glob, subprocess, warnings, re, math, json
import numpy as np
from IPython import display
from PIL import Image, ImageEnhance

from torchvision.transforms import functional as TF

#display settings, because usability counts
#warnings.filterwarnings("error", category=UserWarning)
#%matplotlib inline 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
plt.style.use('bmh')
pd.options.display.max_columns = None
pd.options.display.width = 175

#####################
    
import hydra
from omegaconf import OmegaConf, DictConfig

#conf = OmegaConf.create(default_params)
#OmegaConf.save(conf, f="default_params.yaml")

from loguru import logger
    
@hydra.main(config_path="config", config_name="default")
def _main(cfg: DictConfig):
    default_params = OmegaConf.to_container(cfg, resolve=True)
    logger.debug(default_params)
    latest = -1
    #@markdown check `batch_mode` to run batch settings
    batch_mode = False #@param{type:"boolean"}
    if batch_mode:
      try:
        batch_list
      except NameError:
        raise RuntimeError("ERROR: no batch settings. Please run 'batch settings' cell at the bottom of the page to use batch mode.")
    else:
      try:
        params = default_params
        # https://clear.ml/docs/latest/docs/guides/reporting/hyper_parameters/
        if USE_CLEARML:
            params = task.connect(params) # 
        #writer.add_hparams(hparam_dict=params, metric_dict={})
        # this tensorboard call does nothing. 
        # better approach: pass parameters to script via OmegaConf/Hydra
        # https://github.com/allegroai/clearml/blob/master/examples/frameworks/hydra/hydra_example.py
        # uh...
        params = Bunch(params) # fuck it... # probably easier to use an argparse namesapce here
      except NameError:
        raise RuntimeError("ERROR: no parameters. Please run parameters (step 2.1).")
    #@markdown check `restore` to restore from a previous run
    restore = False#@param{type:"boolean"}
    #@markdown check `reencode` if you are restoring with a modified image or modified image settings
    reencode = False#@param{type:"boolean"}
    #@markdown which run to restore
    restore_run = latest #@param{type:"raw"}
    if restore and restore_run == latest:
      _, restore_run = get_last_file(f'backup/{params.file_namespace}', 
                               f'^(?P<pre>{re.escape(params.file_namespace)}\\(?)(?P<index>\\d*)(?P<post>\\)?_\\d+\\.bak)$')

    def do_run():
      clear_rotoscopers()#what a silly name
      vram_profiling(params.approximate_vram_usage)
      reset_vram_usage()
      global CLIP_MODEL_NAMES
      #@markdown which frame to restore from
      restore_frame =  latest#@param{type:"raw"}

      #set up seed for deterministic RNG
      if params.seed is not None:
        torch.manual_seed(params.seed)

      #load CLIP
      load_clip(params)
      embedder = HDMultiClipEmbedder(cutn=params.cutouts, 
                                     cut_pow = params.cut_pow, 
                                     padding = params.cutout_border,
                                     border_mode = params.border_mode)

      #load scenes
      with vram_usage_mode('Text Prompts'):
        print('Loading prompts...')
        prompts = [[parse_prompt(embedder, p.strip()) 
                  for p in (params.scene_prefix + stage + params.scene_suffix).strip().split('|') if p.strip()]
                  for stage in params.scenes.split('||') if stage]
        print('Prompts loaded.')

      #load init image
      if params.init_image != '':
        init_image_pil = Image.open(fetch(params.init_image)).convert('RGB')
        init_size = init_image_pil.size
        #automatic aspect ratio matching
        if params.width == -1:
          params.width = int(params.height*init_size[0]/init_size[1])
        if params.height == -1:
          params.height = int(params.width*init_size[1]/init_size[0])
      else:
        init_image_pil = None

      #video source
      if params.animation_mode == "Video Source":
        print(f'loading {params.video_path}...')
        video_frames = get_frames(params.video_path)
        params.pre_animation_steps = max(params.steps_per_frame, params.pre_animation_steps)
        if init_image_pil is None:
          init_image_pil = Image.fromarray(video_frames.get_data(0)).convert('RGB')
          #enhancer = ImageEnhance.Contrast(init_image_pil)
          #init_image_pil = enhancer.enhance(2)
          init_size = init_image_pil.size
          if params.width == -1:
            params.width = int(params.height*init_size[0]/init_size[1])
          if params.height == -1:
            params.height = int(params.width*init_size[1]/init_size[0])

      #set up image
      if params.image_model == "Limited Palette":
        img = PixelImage(*format_params(params,
                         'width', 'height', 'pixel_size', 
                         'palette_size', 'palettes', 'gamma', 
                         'hdr_weight', 'palette_normalization_weight'))
        img.encode_random(random_pallet = params.random_initial_palette)
        if params.target_palette.strip() != '':
          img.set_pallet_target(Image.open(fetch(params.target_palette)).convert('RGB'))
        else:
          img.lock_pallet(params.lock_palette)
      elif params.image_model == "Unlimited Palette":
        img = RGBImage(params.width, params.height, params.pixel_size)
        img.encode_random()
      elif params.image_model == "VQGAN":
        VQGANImage.init_vqgan(params.vqgan_model)
        img = VQGANImage(params.width, params.height, params.pixel_size)
        img.encode_random()

      loss_augs = []

      if init_image_pil is not None:
        if not restore:
          print("Encoding image...")
          img.encode_image(init_image_pil)
          print("Encoded Image:")
          display.display(img.decode_image())
        #set up init image prompt
        init_augs = ['direct_init_weight']
        init_augs = [build_loss(x,params[x],f'init image ({params.init_image})', img, init_image_pil) 
                      for x in init_augs if params[x] not in ['','0']]
        loss_augs.extend(init_augs)
        if params.semantic_init_weight not in ['','0']:
          semantic_init_prompt = parse_prompt(embedder, 
                                        f"init image [{params.init_image}]:{params.semantic_init_weight}", 
                                        init_image_pil)
          prompts[0].append(semantic_init_prompt)
        else:
          semantic_init_prompt = None
      else:
        init_augs, semantic_init_prompt = [], None

      #other image prompts

      loss_augs.extend(type(img).get_preferred_loss().TargetImage(p.strip(), img.image_shape, is_path = True) 
                       for p in params.direct_image_prompts.split('|') if p.strip())

      #stabilization

      stabilization_augs = ['direct_stabilization_weight',
                            'depth_stabilization_weight',
                            'edge_stabilization_weight']
      stabilization_augs = [build_loss(x,params[x],'stabilization',
                                       img, init_image_pil) 
                            for x in stabilization_augs if params[x] not in ['','0']]
      loss_augs.extend(stabilization_augs)

      if params.semantic_stabilization_weight not in ['0','']:
        last_frame_semantic = parse_prompt(embedder, 
                                           f"stabilization:{params.semantic_stabilization_weight}", 
                                           init_image_pil if init_image_pil else img.decode_image())
        last_frame_semantic.set_enabled(init_image_pil is not None)
        for scene in prompts:
          scene.append(last_frame_semantic)
      else:
        last_frame_semantic = None

      #optical flow
      if params.animation_mode == 'Video Source':
        if params.flow_stabilization_weight == '':
          params.flow_stabilization_weight = '0'
        optical_flows = [OpticalFlowLoss.TargetImage(f"optical flow stabilization (frame {-2**i}):{params.flow_stabilization_weight}", 
                                                     img.image_shape) 
                         for i in range(params.flow_long_term_samples + 1)]
        for optical_flow in optical_flows:
          optical_flow.set_enabled(False)
        loss_augs.extend(optical_flows)
      elif params.animation_mode == '3D' and params.flow_stabilization_weight not in ['0','']:
        optical_flows = [TargetFlowLoss.TargetImage(f"optical flow stabilization:{params.flow_stabilization_weight}", 
                                                    img.image_shape)]
        for optical_flow in optical_flows:
          optical_flow.set_enabled(False)
        loss_augs.extend(optical_flows)
      else:
        optical_flows = []
      #other loss augs
      if params.smoothing_weight != 0:
        loss_augs.append(TVLoss(weight = params.smoothing_weight))

      #set up filespace
      Path(f'{OUTPATH}/{params.file_namespace}').mkdir(parents=True, exist_ok=True)
      Path(f'backup/{params.file_namespace}').mkdir(parents=True, exist_ok=True)
      if restore:
        base_name = params.file_namespace if restore_run == 0 else f'{params.file_namespace}({restore_run})'
      elif not params.allow_overwrite:
        #finds the next available base_name to save files with. Why did I do this with regex? 
        _, i = get_next_file(f'{OUTPATH}/{params.file_namespace}', 
                             f'^(?P<pre>{re.escape(params.file_namespace)}\\(?)(?P<index>\\d*)(?P<post>\\)?_1\\.png)$',
                             [f"{params.file_namespace}_1.png",f"{params.file_namespace}(1)_1.png"])
        base_name = params.file_namespace if i == 0 else f'{params.file_namespace}({i})'
      else:
        base_name = params.file_namespace

      #restore
      if restore:
        if not reencode:
          if restore_frame == latest:
            filename, restore_frame = get_last_file(f'backup/{params.file_namespace}', 
                                                    f'^(?P<pre>{re.escape(base_name)}_)(?P<index>\\d*)(?P<post>\\.bak)$')
          else: 
            filename = f'{base_name}_{restore_frame}.bak'
          print("restoring from", filename)
          img.load_state_dict(torch.load(f'backup/{params.file_namespace}/{filename}'))
        else:#reencode
          if restore_frame == latest:
            filename, restore_frame = get_last_file(f'{OUTPATH}/{params.file_namespace}', 
                                                    f'^(?P<pre>{re.escape(base_name)}_)(?P<index>\\d*)(?P<post>\\.png)$')
          else: 
            filename = f'{base_name}_{restore_frame}.png'
          print("restoring from", filename)
          img.encode_image(Image.open(f'{OUTPATH}/{params.file_namespace}/{filename}').convert('RGB'))
        i = restore_frame*params.save_every
      else:
        i = 0

      #graphs
      if params.show_graphs:
        fig, axs = plt.subplots(4, 1, figsize=(21,13))
        axs  = np.asarray(axs).flatten()
        #fig.facecolor = (0,0,0)
      else:
        fig, axs = None, None

      #make the main model object
      model = DirectImageGuide(img, embedder, lr = params.learning_rate)

      #Update is called each step.
      def update(i, stage_i):
        #display
        if params.clear_every > 0 and i > 0 and i % params.clear_every == 0:
          display.clear_output()
        if params.display_every > 0 and i % params.display_every == 0:
          print(f"Step {i} losses:")
          if model.dataframe:
            rec = model.dataframe[0].iloc[-1]
            print(rec)
            for k,v in rec.iteritems():
                #task.get_logger().report_scalar(
                #    "losses",
                #    f"{k}",
                #    value=v,
                #    iteration=i
                #    )
                writer.add_scalar(
                    tag=f"losses/{k}",
                    scalar_value=v,
                    global_step=i
                    )

          if params.approximate_vram_usage:
            print("VRAM Usage:")
            print_vram_usage()
          display_width = int(img.image_shape[0]*params.display_scale)
          display_height = int(img.image_shape[1]*params.display_scale)
          if stage_i > 0 and params.show_graphs:
            model.plot_losses(axs)
            im = img.decode_image()
            sidebyside = make_hbox(im.resize((display_width, display_height), Image.LANCZOS), fig)
            display.display(sidebyside)
          else:
            im = img.decode_image()
            display.display(im.resize((display_width, display_height), Image.LANCZOS))
          if params.show_palette and isinstance(img, PixelImage):
            print('Palette:')
            display.display(img.render_pallet())
        #save
        if i > 0 and params.save_every > 0 and i % params.save_every == 0:
          try:
            im
          except NameError:
            im = img.decode_image()
          n = i//params.save_every
          filename = f"{OUTPATH}/{params.file_namespace}/{base_name}_{n}.png"
          im.save(filename)
          ####################
          ## DMARX
          #task.upload_artifact(name=filename, artifact_object=filename)
          #task.upload_artifact(name=filename, artifact_object=OUTTERPATH) # fuck that....
        
          #logger = task.get_logger()
          #logger.report_media(
          #  'images', 'pytti output', iteration=i,
          #   local_path=filename)
          im_np = np.array(im)
          writer.add_image(
              tag='pytti output',
              #img_tensor=filename, # thought this would work?
              img_tensor=im_np,
              global_step=i,
              dataformats="HWC" # this was the key
          )
        
        
          #reuse_last_task_id=False
          #auto_connect_arg_parser=True,
          ####################
          if params.backups > 0:
            filename = f"backup/{params.file_namespace}/{base_name}_{n}.bak"
            torch.save(img.state_dict(), filename)
            if n > params.backups:
              subprocess.run(['rm', f"backup/{params.file_namespace}/{base_name}_{n-params.backups}.bak"])
        #animate
        t = (i - params.pre_animation_steps)/(params.steps_per_frame*params.frames_per_second)
        set_t(t)
        if i >= params.pre_animation_steps:
          if (i - params.pre_animation_steps) % params.steps_per_frame == 0:
            print(f"Time: {t:.4f} seconds")
            update_rotoscopers(((i - params.pre_animation_steps)//params.steps_per_frame+1)*params.frame_stride)
            if params.reset_lr_each_frame:
              model.set_optim(None)
            if params.animation_mode == "2D":
              tx, ty = parametric_eval(params.translate_x), parametric_eval(params.translate_y)
              theta = parametric_eval(params.rotate_2d)
              zx, zy = parametric_eval(params.zoom_x_2d), parametric_eval(params.zoom_y_2d)
              next_step_pil = zoom_2d(img, 
                                      (tx,ty), (zx,zy), theta, 
                                      border_mode = params.infill_mode, sampling_mode = params.sampling_mode)
              ################ DMARX
              for k,v in {'tx':tx,
                          'ty':ty,
                          'theta':theta,
                          'zx':zx,
                          'zy':zy, 
                          't':t}.items():
                task.get_logger().report_scalar(
                    "translation_2d",
                    f"{k}",
                    value=v,
                    iteration=i
                    )
              ###########################
            elif params.animation_mode == "3D":
              try:
                im
              except NameError:
                im = img.decode_image()
              with vram_usage_mode('Optical Flow Loss'):
                flow, next_step_pil = zoom_3d(img, 
                                            (params.translate_x,params.translate_y,params.translate_z_3d), params.rotate_3d, 
                                            params.field_of_view, params.near_plane, params.far_plane,
                                            border_mode = params.infill_mode, sampling_mode = params.sampling_mode,
                                            stabilize = params.lock_camera)
                freeze_vram_usage()

              for optical_flow in optical_flows:
                optical_flow.set_last_step(im)
                optical_flow.set_target_flow(flow)
                optical_flow.set_enabled(True)
            elif params.animation_mode == "Video Source":
              frame_n = min((i - params.pre_animation_steps)*params.frame_stride//params.steps_per_frame, len(video_frames) - 1)
              next_frame_n = min(frame_n + params.frame_stride, len(video_frames) - 1)
              next_step_pil = Image.fromarray(video_frames.get_data(next_frame_n)).convert('RGB').resize(img.image_shape, Image.LANCZOS)
              for j, optical_flow in enumerate(optical_flows):
                old_frame_n = frame_n - (2**j - 1)*params.frame_stride
                save_n = i//params.save_every - (2**j - 1)
                if old_frame_n < 0 or save_n < 1:
                  break
                current_step_pil = Image.fromarray(video_frames.get_data(old_frame_n)).convert('RGB').resize(img.image_shape, Image.LANCZOS)
                filename = f"backup/{params.file_namespace}/{base_name}_{save_n}.bak"
                filename = None if j == 0 else filename
                flow_im, mask_tensor = optical_flow.set_flow(current_step_pil, next_step_pil, 
                                                            img, filename, 
                                                            params.infill_mode, params.sampling_mode)
                optical_flow.set_enabled(True)
                #first flow is previous frame
                if j == 0:
                  mask_accum = mask_tensor.detach()
                  valid = mask_tensor.mean()
                  print("valid pixels:", valid)
                  if params.reencode_each_frame or valid < .03:
                    if isinstance(img, PixelImage) and valid >= .03:
                      img.lock_pallet()
                      img.encode_image(next_step_pil, smart_encode = False)
                      img.lock_pallet(params.lock_palette)
                    else:
                      img.encode_image(next_step_pil)
                    reencoded = True
                  else:
                    reencoded = False
                else:
                  with torch.no_grad():
                    optical_flow.set_mask((mask_tensor - mask_accum).clamp(0,1))
                    mask_accum.add_(mask_tensor)
            if params.animation_mode != 'off':
              for aug in stabilization_augs:
                aug.set_comp(next_step_pil)
                aug.set_enabled(True)
              if last_frame_semantic is not None:
                last_frame_semantic.set_image(embedder, next_step_pil)
                last_frame_semantic.set_enabled(True)
              for aug in init_augs:
                aug.set_enabled(False)
              if semantic_init_prompt is not None:
                semantic_init_prompt.set_enabled(False)


      model.update = update

      print(f"Settings saved to {OUTPATH}/{params.file_namespace}/{base_name}_settings.txt")
      save_settings(params, f"{OUTPATH}/{params.file_namespace}/{base_name}_settings.txt")

      skip_prompts = i // params.steps_per_scene
      skip_steps   = i %  params.steps_per_scene
      last_scene = prompts[0] if skip_prompts == 0 else prompts[skip_prompts - 1]
      for scene in prompts[skip_prompts:]:
        print("Running prompt:", ' | '.join(map(str,scene)))
        i += model.run_steps(params.steps_per_scene-skip_steps, 
                             scene, last_scene, loss_augs, 
                             interp_steps = params.interpolation_steps,
                             i_offset = i, skipped_steps = skip_steps)
        skip_steps = 0
        model.clear_dataframe()
        last_scene = scene
      if fig:
        del fig, axs
      ############################# DMARX
      writer.close()
      #############################
        
    #if __name__ == '__main__':
    try:
      gc.collect()
      torch.cuda.empty_cache()
      if batch_mode:
        if restore:
          settings_list = batch_list[restore_run:]
        else:
          settings_list = batch_list
          namespace = batch_list[0]['file_namespace']
          subprocess.run(['mkdir','-p',f'{OUTPATH}/{namespace}'])
          save_batch(batch_list, f'{OUTPATH}/{namespace}/{namespace}_batch settings.txt')
          print(f"Batch settings saved to {OUTPATH}/{namespace}/{namespace}_batch settings.txt")
        for settings in settings_list:
          setting_string = json.dumps(settings)
          print("SETTINGS:")
          print(setting_string)
          params = load_settings(setting_string)
          if params.animation_mode == '3D':
            init_AdaBins()
          params.allow_overwrite = False
          do_run()
          restore = False
          reencode = False
          gc.collect()
          torch.cuda.empty_cache()
      else:
        if params.animation_mode == '3D':
          pass
          #init_AdaBins()
        do_run()
        print("Complete.")
        gc.collect()
        torch.cuda.empty_cache()
    except KeyboardInterrupt:
      pass
    except RuntimeError:
      print_vram_usage()
      raise

if __name__ == '__main__':
    _main()