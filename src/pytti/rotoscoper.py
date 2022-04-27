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
def get_frames(path, params=None):
    """reads the frames of the mp4 file `path` and returns them as a list of PIL images"""

    in_fname = path
    out_fname = f"{path}_converted.mp4"
    if not path_exists(path + "_converted.mp4"):
        logger.debug(f"Converting {path}...")
        cmd = ["ffmpeg", "-i", in_fname]
        # if params is None:
        # subprocess.run(["ffmpeg", "-i", in_fname, out_fname])
        if params is not None:
            # https://trac.ffmpeg.org/wiki/ChangingFrameRate
            cmd += ["-filter:v", f"fps={params.frames_per_second}"]

        # https://trac.ffmpeg.org/wiki/Encode/H.264
        cmd += [
            "-c:v",
            "libx264",
            "-crf",
            "17",  # = effectively lossless
            "-preset",
            "veryslow",  # = effectively lossless
            "-tune",
            "fastdecode",  # not sure this is what I want, zerolatency and stillimage might make sense? can experiment I guess?
            "-pix_fmt",
            "yuv420p",  # may be necessary for "dumb players"
            "-acodec",
            "copy",  # copy audio codec cause why not
            out_fname,
        ]
        logger.debug(cmd)

        subprocess.run(cmd)

        logger.debug(f"Converted {in_fname} to {out_fname}.")

        # yeah I don't think this is actually true, but it probably should be.
        logger.warning(
            f"WARNING: future runs will automatically use {out_fname}, unless you delete it."
        )

    vid = imageio.get_reader(out_fname, "ffmpeg")
    n_frames = vid._meta["nframes"]
    logger.info(f"loaded {n_frames} frames from {out_fname}")
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
