import os

from loguru import logger

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


# https://hydra.cc/docs/advanced/search_path/#
# https://github.com/facebookresearch/hydra/issues/763

class PyttiLocalConfigSearchPathPlugin(SearchPathPlugin):

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:

        local_path = f"{os.getcwd()}/config/"
        # If path doesn't exist, we should create it and copy configs to it
        logger.debug(local_path)

        search_path.append(
            provider="pytti_hydra_pathplugin", path=f"file://{local_path}/"
        )