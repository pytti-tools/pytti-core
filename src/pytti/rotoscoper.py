from pytti.Notebook import get_frames

rotoscopers = []


def clear_rotoscopers():
    global rotoscopers
    rotoscopers = []


# this is... weird. also why the globals?
def update_rotoscopers(frame_n: int):
    """
    For each rotoscope in the global list of rotoscopes, call the update function

    :param frame_n: The current frame number
    """
    global rotoscopers
    for r in rotoscopers:
        r.update(frame_n)


class Rotoscoper:
    def __init__(self, video_path, target=None, thresh=None):
        global rotoscopers
        if video_path[0] == "-":
            video_path = video_path[1:]
            inverted = True
        else:
            inverted = False

        self.frames = get_frames(video_path)
        self.target = target
        self.inverted = inverted
        rotoscopers.append(self)

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
