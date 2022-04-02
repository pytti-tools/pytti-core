# from pytti.Notebook import get_frames
from loguru import logger


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

# rotoscopers = []


# def clear_rotoscopers():
#     global rotoscopers
#     rotoscopers = []


# this is... weird. also why the globals?
# def update_rotoscopers(frame_n: int):
#     """
#     For each rotoscope in the global list of rotoscopes, call the update function

#     :param frame_n: The current frame number
#     """
#     global rotoscopers
#     for r in rotoscopers:
#         r.update(frame_n)


# surprised we're not using opencv here.
# let's call this another unnecessary subprocess call to deprecate.
def get_frames(path):
    """reads the frames of the mp4 file `path` and returns them as a list of PIL images"""
    import imageio, subprocess
    from PIL import Image
    from os.path import exists as path_exists

    if not path_exists(path + "_converted.mp4"):
        logger.debug(f"Converting {path}...")
        subprocess.run(["ffmpeg", "-i", path, path + "_converted.mp4"])
        logger.debug(f"Converted {path} to {path}_converted.mp4.")
        logger.warning(
            f"WARNING: future runs will automatically use {path}_converted.mp4, unless you delete it."
        )
    vid = imageio.get_reader(path + "_converted.mp4", "ffmpeg")
    n_frames = vid._meta["nframes"]
    logger.info(f"loaded {n_frames} frames. for {path}")
    return vid


class Rotoscoper:
    def __init__(self, video_path, target=None, thresh=None):
        # global rotoscopers
        global ROTOSCOPERS  # redundant, but leaving it here to document the globals
        if video_path[0] == "-":
            video_path = video_path[1:]
            inverted = True
        else:
            inverted = False

        self.frames = get_frames(video_path)
        self.target = target
        self.inverted = inverted
        # rotoscopers.append(self) # uh.... why. why does it work this way. weird af.
        ROTOSCOPERS.add(self)

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
