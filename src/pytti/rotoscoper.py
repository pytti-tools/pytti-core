import imageio, subprocess
from os.path import exists as path_exists

from loguru import logger
from PIL import Image


class RotoscopingOrchestrator:
    def __init__(self):
        self.rotoscopers = []

    def add(self, other):
        self.rotoscopers.append(other)

    def clear_rotoscopers(self):
        self.rotoscopers = []

    def update_rotoscopers(self, frame_n: int):
        for r in self.rotoscopers:
            r.update(frame_n)


ROTOSCOPERS = RotoscopingOrchestrator()  # fml...
rotoscopers = ROTOSCOPERS.rotoscopers
update_rotoscopers = ROTOSCOPERS.update_rotoscopers
clear_rotoscopers = ROTOSCOPERS.clear_rotoscopers

# surprised we're not using opencv here.
# let's call this another unnecessary subprocess call to deprecate.
def get_frames(path):
    """reads the frames of the mp4 file `path` and returns them as a list of PIL images"""

    if not path_exists(path + "_converted.mp4"):
        logger.debug(f"Converting {path}...")
        subprocess.run(["ffmpeg", "-i", path, path + "_converted.mp4"])
        logger.debug(f"Converted {path} to {path}_converted.mp4.")

        # yeah I don't think this is actually true, but it probably should be.
        logger.warning(
            f"WARNING: future runs will automatically use {path}_converted.mp4, unless you delete it."
        )

    vid = imageio.get_reader(path + "_converted.mp4", "ffmpeg")
    n_frames = vid._meta["nframes"]
    logger.info(f"loaded {n_frames} frames. for {path}")
    return vid


class Rotoscoper:
    def __init__(self, video_path, target=None, thresh=None):
        global ROTOSCOPERS  # redundant, but leaving it here to document the globals
        if video_path[0] == "-":
            video_path = video_path[1:]
            inverted = True
        else:
            inverted = False

        self.frames = get_frames(video_path)
        self.target = target
        self.inverted = inverted
        ROTOSCOPERS.add(self)  # uh.... why. why does it work this way. weird af.

    def update(self, frame_n):
        """
        Updates the mask of the attached target.

        :param frame_n: The frame number to update the mask for
        :return: Nothing.
        """
        if self.target is None:
            return
        mask_pil = Image.fromarray(self.frames.get_data(frame_n)).convert("L")
        self.target.set_mask(mask_pil, self.inverted)
