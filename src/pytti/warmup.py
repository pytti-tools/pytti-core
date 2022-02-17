"""
Startup helpers.
"""
# import os
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf
from pytti import __path__


__path__ = __path__[0]
logger.debug(__path__)

# This should match the path used by PyttiLocalConfigSearchPathPlugin
# ...which means I should probably initialize it in a way that ensures
# the path here is kept in synch with the path there.
# ::sigh:: add it to the pile.
local_path = Path.cwd() / "config"
full_local = local_path / "conf"
default_fname = "default.yaml"
demo_fname = "demo.yaml"

dest_fpath_default = Path(local_path) / default_fname
dest_fpath_demo = Path(full_local) / demo_fname

logger.debug(__path__)
install_dir = Path(__path__)  # uh... I hope this is correct?
shipped_fpath = install_dir / "assets"

src_fpath_default = Path(shipped_fpath) / default_fname
src_fpath_demo = Path(shipped_fpath) / demo_fname


def ensure_configs_exist():
    """
    If the config directory doesn't exist, create it with the default and demo configs
    """
    # if not default_fpath.exists(): # too aggressive
    if (
        local_path.exists()
    ):  # slightly less stable, but less likely overwrite user content
        logger.debug("Local config directory detected.")
    else:
        logger.info("Local config directory not detected.")
        logger.info("Creating local config directory with default and demo configs")
        # make the ./config/conf
        full_local.mkdir(parents=True, exist_ok=True)

        # might be better to use shutils.copy()?
        read_fpath_default = str(src_fpath_default.resolve())
        read_fpath_demo = str(src_fpath_demo.resolve())
        with open(read_fpath_default, "r") as f:
            default_yaml = f.read()
        with open(read_fpath_demo, "r") as f:
            demo_yaml = f.read()

        write_fpath_default = str(dest_fpath_default.resolve())
        write_fpath_demo = str(dest_fpath_demo.resolve())
        with open(write_fpath_default, "w") as f:
            f.write(default_yaml)
        with open(write_fpath_demo, "w") as f:
            f.write(demo_yaml)


# Path.home() == os.path.expanduser('~')
# user_cache = Path.home() / '.cache'
# logger.debug(f'user_cache: {user_cache}')
OmegaConf.register_new_resolver("user_cache", lambda: Path.home() / ".cache")

OmegaConf.register_new_resolver("path_join", lambda a, b: Path(a) / Path(b))
