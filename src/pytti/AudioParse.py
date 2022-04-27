import numpy as np
import typing
import subprocess
from loguru import logger
from scipy.signal import butter, sosfilt, sosfreqz

SAMPLERATE = 44100


class SpectralAudioParser:
    """
    reads a given input file, scans along it and parses the amplitude in selected bands using butterworth bandpass filters.
    the amplitude is normalized into the 0..1 range for easier use in transformation functions.
    """

    def __init__(
            self,
            input_audio,
            offset,
            frames_per_second,
            filters
    ):
        if len(filters) < 1:
            raise RuntimeError("When using input_audio, at least 1 filter must be specified")

        pipe = subprocess.Popen(['ffmpeg', '-i', input_audio,
                                 '-f', 's16le',
                                 '-acodec', 'pcm_s16le',
                                 '-ar', str(SAMPLERATE),
                                 '-ac', '1',
                                 '-'], stdout=subprocess.PIPE, bufsize=10 ** 8)

        self.audio_samples = np.array([], dtype=np.int16)

        # read the audio file from the pipe in 0.5s blocks (2 bytes per sample)
        while True:
            buf = pipe.stdout.read(SAMPLERATE)
            self.audio_samples = np.append(self.audio_samples, np.frombuffer(buf, dtype=np.int16))
            if len(buf) < SAMPLERATE:
                break
        if len(self.audio_samples) < 0:
            raise RuntimeError("Audio samples are empty, assuming load failed")
        self.duration = len(self.audio_samples) / SAMPLERATE
        logger.debug(
            f"initialized audio file {input_audio}, samples read: {len(self.audio_samples)}, total duration: {self.duration}s")
        self.offset = offset
        if offset > self.duration:
            raise RuntimeError(f"Audio offset set at {offset}s but input audio is only {duration}s long")
        # analyze all samples for the current frame
        self.window_size = int(1 / frames_per_second * SAMPLERATE)
        self.filters = filters

        # parse band maxima first for normalizing the filtered signal to 0..1 at arbitrary points in the file later
        # this initialization is a bit compute intensive, especially for higher fps numbers, but i couldn't find a cleaner way
        # (band-passing the entire track instead of windows creates maxima that are way off, some filtering anomaly i don't understand...)
        steps = int((self.duration - self.offset) * frames_per_second)
        interval = 1 / frames_per_second
        maxima = {}
        time_steps = np.linspace(0, steps, num=steps) * interval
        for t in time_steps:
            sample_offset = int(t * SAMPLERATE)
            cur_maxima = bp_filtered(self.audio_samples[sample_offset:sample_offset + self.window_size], filters)
            for key in cur_maxima:
                if key in maxima:
                    maxima[key] = max(maxima[key], cur_maxima[key])
                else:
                    maxima[key] = cur_maxima[key]
        self.band_maxima = maxima
        logger.debug(f"initialized band maxima for {len(filters)} filters: {self.band_maxima}")

    def get_params(self, t) -> typing.Dict[str, float]:
        """
        Return the amplitude parameters at the given point in time t within the audio track, or 0 if the track has ended.
        Amplitude/energy parameters are normalized into the [0,1] range.
        """
        # Get the point in time (sample-offset) in the track in seconds based on sample-rate
        sample_offset = int(t * SAMPLERATE + self.offset * SAMPLERATE)
        logger.debug(f"Analyzing audio at {self.offset + t}s")
        if sample_offset < len(self.audio_samples):
            window_samples = self.audio_samples[sample_offset:sample_offset + self.window_size]
            if len(window_samples) < self.window_size:
                # audio input file has likely ended
                logger.debug(
                    f"Warning: sample offset is out of range at time offset {t + self.offset}s. Returning null result")
                return {}
            return bp_filtered_norm(window_samples, self.filters, self.band_maxima)
        else:
            logger.debug(f"Warning: Audio input has ended. Returning null result")
            return {}

    def get_duration(self):
        return self.duration


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='bandpass', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def bp_filtered(window_samples, filters) -> typing.Dict[str, float]:
    results = {}
    for filter in filters:
        offset = filter.f_width / 2
        lower = filter.f_center - offset
        upper = filter.f_center + offset
        filtered = butter_bandpass_filter(window_samples, lower, upper, SAMPLERATE, order=filter.order)
        results[filter.variable_name] = np.max(np.abs(filtered))
    return results


def bp_filtered_norm(window_samples, filters, norm_factors) -> typing.Dict[str, float]:
    results = bp_filtered(window_samples, filters)
    for key in results:
        # normalize
        results[key] = results[key] / norm_factors[key]
    return results
