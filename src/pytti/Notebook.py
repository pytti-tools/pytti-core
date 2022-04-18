# this library was originally designed for use with google colab runtimes.
# This file defined utility functions for use with notebooks.
# Many of the functions previously defined here have been moved into the following:
#  * src/pytti/eval_tools.py
#  * src/pytti/tensor_tools.py
#  * src/pytti/vram_tools.py
#
# It seems like most of the functionality that remains defined here is actually
# connected with animation/video/rotoscoping logic, i.e. should probably be moved
# elsewhere (TBD)

from loguru import logger
from omegaconf import OmegaConf, DictConfig
import json, random
import os, re
from PIL import Image
import clip

from pytti import Perceptor


# https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        elif shell == "Shell":
            return True  # Google Colab
        else:
            logger.debug("DEGBUG: unknown shell type:", shell)
            return False
    except NameError:
        return False  # Probably standard Python interpreter


# what is this doing in here? This should be in the notebook...
if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def change_tqdm_color():
    if not is_notebook():
        return

    from IPython import display
    from IPython.display import HTML

    def set_css_in_cell_output():
        display.display(
            HTML(
                """
      <style>
          .jupyter-widgets {color: #d5d5d5 !important;}
          .widget-label {color: #d5d5d5 !important;}
      </style>
    """
            )
        )

    get_ipython().events.register("pre_run_cell", set_css_in_cell_output)


# this doesn't belong in here
def get_last_file(directory, pattern):
    def key(f):
        index = re.match(pattern, f).group("index")
        return 0 if index == "" else int(index)

    files = [f for f in os.listdir(directory) if re.match(pattern, f)]
    if len(files) == 0:
        return None, None
    files.sort(key=key)
    index = key(files[-1])
    return files[-1], index


# this doesn't belong in here
def get_next_file(directory, pattern, templates):
    """
    Given a directory, a file pattern, and a list of templates,
    return the next file name and index that matches the pattern.

    If no files match the pattern, return the first template and 0.

    If multiple files match the pattern, sort the files by index and return the next index.

    If the index is the last index in the list of templates, return the first template and 0

    :param directory: The directory where the files are located
    :param pattern: The pattern to match files against
    :param templates: A list of file names that are used to create the new file
    :return: The next file name and the next index.
    """

    files = [f for f in os.listdir(directory) if re.match(pattern, f)]
    if len(files) == 0:
        return templates[0], 0

    def key(f):
        index = re.match(pattern, f).group("index")
        return 0 if index == "" else int(index)

    files.sort(key=key)
    n = len(templates) - 1
    for i, f in enumerate(files):
        index = key(f)
        if i != index:
            return (
                (templates[0], 0)
                if i == 0
                else (
                    re.sub(
                        pattern,
                        lambda m: f"{m.group('pre')}{i}{m.group('post')}",
                        templates[min(i, n)],
                    ),
                    i,
                )
            )
    return (
        re.sub(
            pattern,
            lambda m: f"{m.group('pre')}{i+1}{m.group('post')}",
            templates[min(i, n)],
        ),
        i + 1,
    )


# deprecate this (tensorboard)
def make_hbox(im, fig):
    # https://stackoverflow.com/questions/51315566/how-to-display-the-images-side-by-side-in-jupyter-notebook/51316108
    import io
    import ipywidgets as widgets
    from ipywidgets import Layout

    with io.BytesIO() as buf:
        im.save(buf, format="png")
        buf.seek(0)
        wi1 = widgets.Image(
            value=buf.read(),
            format="png",
            layout=Layout(border="0", margin="0", padding="0"),
        )
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        wi2 = widgets.Image(
            value=buf.read(),
            format="png",
            layout=Layout(border="0", margin="0", padding="0"),
        )
    return widgets.HBox(
        [wi1, wi2],
        layout=Layout(border="0", margin="0", padding="0", align_items="flex-start"),
    )


# deprecate this (hydra)
def load_settings(settings_string, random_seed=True):
    params = OmegaConf.create(json.loads(settings_string))
    if random_seed or params.seed is None:
        params.seed = random.randint(-0x8000_0000_0000_0000, 0xFFFF_FFFF_FFFF_FFFF)
        logger.debug("using seed:", params.seed)
    return params


# deprecate this (hydra)
def write_settings(settings_dict, f):
    json.dump(settings_dict, f)
    f.write("\n\n")
    # params = settings_dict
    params = OmegaConf.create(settings_dict)
    # why are scenes being parsed and iterated over here?
    # Is this maybe part of the resume behavior?
    # ... whatever it is... parsing logic needs to live in the code.
    # omegaconf/hydra will be responsible for loading and persisting params.
    scenes = [
        (params.scene_prefix + scene + params.scene_suffix).strip()
        for scene in params.scenes.split("||")
        if scene
    ]
    for i, scene in enumerate(scenes):
        frame = i * params.steps_per_scene / params.save_every
        f.write(str(f"{frame:.2f}: {scene}".encode("utf-8", "ignore")))
        f.write("\n")


# deprecate this (hydra)
def save_settings(settings_dict, path):
    if isinstance(settings_dict, DictConfig):
        settings_dict = OmegaConf.to_container(settings_dict, resolve=True)

    with open(path, "w+") as f:
        write_settings(settings_dict, f)
        # OmegaConf.save(config=settings_dict, f=f)


# deprecate this (hydra)
def save_batch(settings_list, path):

    with open(path, "w+") as f:
        for batch_index, settings_dict in enumerate(settings_list):
            f.write(f"batch_index: {batch_index}")
            f.write("\n")
            write_settings(dict(settings_dict), f)
            f.write("\n\n")


def _sanitize_for_config(in_str):
    for char in ("/", "-"):
        in_str = in_str.replace(char, "")
    return in_str


SUPPORTED_CLIP_MODELS = {
    _sanitize_for_config(model_name): model_name
    for model_name in clip.available_models()
}
SUPPORTED_CLIP_MODELS["RN50__yfcc15m"] = "RN50--yfcc15m"

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
