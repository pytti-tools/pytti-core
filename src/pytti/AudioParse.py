import numpy as np
import typing
import subprocess
from loguru import logger
from scipy.signal import butter, sosfilt, sosfreqz

SAMPLERATE=44100

class SpectralAudioParser:
    """
    Audio Parser reads a given input file, scans along it and parses its spectrum using FFT.
    The FFT output is split into three bands (low,mid,high), the (average) amplitude of which is then returned for use in animation functions.
    """
    def __init__(
        self,
        input_audio,
        offset,
        window_size,
        filters
        ):
        pipe = subprocess.Popen(['ffmpeg', '-i', input_audio,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', str(SAMPLERATE),
            '-ac', '1',
            '-'], stdout=subprocess.PIPE, bufsize=10**8)

        self.audio_samples = np.array([], dtype=np.int16)
        
        # read the audio file from the pipe in 0.5s blocks (2 bytes per sample)
        while True:
            buf = pipe.stdout.read(SAMPLERATE)
            self.audio_samples = np.append(self.audio_samples, np.frombuffer(buf, dtype=np.int16))
            if len(buf) < SAMPLERATE:
                break
        if len(self.audio_samples) < 0:
            raise RuntimeError("Audio samples are empty, assuming load failed")
        logger.debug(f"initialized audio file {input_audio}, samples read: {len(self.audio_samples)}")
        self.offset = offset
        self.window_size = window_size
        self.filters = filters
        # pink noise normalization blatantly stolen from https://github.com/aiXander/Realtime_PyAudio_FFT/blob/275c8b1fc268ac946470b0d7a80de56eb2212b58/src/stream_analyzer.py#L107
        self.fftx = np.arange(int(self.window_size/2), dtype=float) * SAMPLERATE / self.window_size
        self.power_normalization_coefficients = np.logspace(np.log2(1), np.log2(np.log2(SAMPLERATE/2)), len(self.fftx), endpoint=True, base=2, dtype=None)



    def get_params(self, t) -> typing.Tuple[float, float, float]:
        """
        Return the amplitude parameters at the given point in time t within the audio track, or 0 if the track has ended.
        Amplitude/energy parameters are normalized into the [0,1] range.
        """
        # Get the point in time (sample-offset) in the track in seconds based on sample-rate
        sample_offset = int(t * SAMPLERATE + self.offset * SAMPLERATE)
        logger.debug(f"Analyzing audio at {self.offset+t}s")
        if sample_offset < len(self.audio_samples):
            window_samples = self.audio_samples[sample_offset:sample_offset+self.window_size]
            if len(window_samples) < self.window_size:
                # audio input file has likely ended
                # TODO could round down to the next lower pow2 then do it anyway. not a critical case though IMO.
                logger.debug(f"Warning: sample offset is out of range at time offset {t+self.offset}s. Returning 0 vector")
                return (0, 0, 0)
                    
            # fade-in / fade-out window to taper off the signal
            window_samples = window_samples * np.hamming(len(window_samples))
            return bp_tuple(t, window_samples, self.filters)
            #return fft_tuple(t)
        else:
            logger.debug(f"Warning: Audio input has ended. Returning 0 vector")
            return (0, 0, 0)

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y

def bp_tuple(t, window_samples, filters) -> typing.Dict[str, float]:
    for filter in filters:
        offset = filter.f_width/2
        lower = filter.f_center - offset
        upper = filter.f_center + offset
        filtered = butter_bandpass_filter(window_samples, lower, upper, SAMPLERATE, order=filter.order)
        # Normalize from signed 16-bit max value to 0..1 range
        val = np.max(np.abs(filtered)) / 32768
        return (val, 0, 0)

def fft_tuple(t, window_samples) -> typing.Tuple[float, float, float]:
    fft = np.fft.fft(window_samples)
    # summing together the real and imaginary components, i think(??)
    left, right = np.split(np.abs(fft), 2)
    fft = np.add(left, right[::-1])

    # pink noise adjust
    fft = fft * self.power_normalization_coefficients

    freq_buckets = np.fft.fftfreq(self.window_size, 1 / SAMPLERATE)
    # collect energy for each frequency band
    # TODO: this could probably be done in a much nicer way with bandpass filters somehow... not sure on the correct arithmetic though
    low_bucket = 0
    low_count = 0
    mid_bucket = 0
    mid_count = 0
    high_bucket = 0
    high_count = 0
    for i in range(len(fft)):
        freq = self.fftx[i]
        if freq < self.low_cutoff:
            low_bucket += fft[i]
            low_count += 1
        elif freq < self.mid_cutoff:
            mid_bucket += fft[i]
            mid_count += 1
        else:
            high_bucket += fft[i]
            high_count += 1
    # mean energy per bucket
    if low_count > 0 and mid_count > 0 and high_count > 0:    
        low_bucket = low_bucket / low_count
        mid_bucket = mid_bucket / mid_count
        high_bucket = high_bucket / high_count
    else:
        logger.debug(f"Warning: There were empty buckets in the audio frequency analysis. Returning 0 vector")
        return (0,0,0)
    # normalize to [0,1] range
    max_val = np.max(fft)
    if max_val > 0:
        low_bucket = low_bucket / max_val
        mid_bucket = mid_bucket / max_val
        high_bucket = high_bucket / max_val
    else:
        logger.debug(f"Warning: Max val was 0 in the audio frequency analysis. Returning 0 vector")
        return (0,0,0)
    return (float(low_bucket), float(mid_bucket), float(high_bucket))