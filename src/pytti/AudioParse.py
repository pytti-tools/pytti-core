import numpy as np
import audiofile
import typing

class SpectralAudioParser:
    """
    Audio Parser reads a given input file, scans along it and parses its spectrum using FFT.
    The FFT output is split into three bands (low,mid,high), the (average) amplitude of which is then returned for use in animation functions.
    """
    def __init__(
        self,
        params=None
        ):
        if params.input_audio:
            self.audio_samples, self.sample_rate = audiofile.read(params.input_audio, offset=params.input_audio_offset, always_2d=True)

    def get_params(self, t) -> typing.Tuple[float, float, float]:
        """
        Return the amplitude parameters at the given point in time t within the audio track, or 0 if the track has ended.
        """
        # Get the point in time (sample-offset) in the track in seconds based on sample-rate
        sample_offset = int(t * self.sample_rate)
        if sample_offset < self.audio_samples.shape[0]:
            # TODO: read back up on numpy array slicing, read [sample_offset, sample_offset+window_size] here
            #       read back up on fft window size parameters etc.
            #       read up on whether to use fft2 here or to sum the audio file on initialization into a mono signal first maybe
            np.fft.fft2(self.audio_samples[:sample_offset])
        else:
            return (0, 0, 0)