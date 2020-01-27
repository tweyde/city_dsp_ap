import numpy as np
from scipy import fft
from .signals import TimeSignal, get_samples_and_rate
from .util import adsr_window_log

def circ_convolution(signal1, signal2, num_points=None):
    # return signal.convolve(signal1, np.concatenate([signal2, signal2]), mode='same')
    if num_points == None:
        num_points = len(signal1)
    return fft.irfft(fft.rfft(signal1, num_points) * fft.rfft(signal2, num_points))


def quantise(input, depth, max_amp=1):
    """signal_quant = quantise(input, depth)

    quantises the input signal uniformly to "depth" bits
    input is assumed to fall within the range [-max_amp, max_amp]
    """
    if isinstance(input, TimeSignal):
        samples = input.samples
    elif np.ndim(input) > 0:
        samples = np.asarray(input)
    else:
        raise TypeError('Only AudioSignals, Numpy arrays or other iterables are supported as input, not {}'.format(type(input)))

    # clip out of range values
    samples_quant = np.clip(samples, -max_amp, max_amp)

    # scale data into positive range
    samples_quant = (samples_quant + max_amp) / (2 * max_amp)

    # expand data into number of integer values
    samples_quant = samples_quant * ((2**depth) - 1)

    # quantise by rounding
    samples_quant = np.around(samples_quant) / ((2**depth)-1)

    # rescale data
    samples_quant = samples_quant * 2 - max_amp

    if isinstance(input, TimeSignal):
        return type(input)(samples_quant, input.samplerate)
    else:
        return samples_quant


def apply_adsr(input, max_amp, sustain_amp, attack, decay, release, samplerate=None):
    samples, samplerate = get_samples_and_rate(input, samplerate)
    attack_size = round(attack * samplerate)
    decay_size = round(decay * samplerate)
    release_size = round(release * samplerate)
    sustain_size = len(samples) - attack_size - decay_size - release_size
    adsr_window = adsr_window_log(max_amp, sustain_amp, attack_size, decay_size, sustain_size, release_size)
    samples_adsr = samples * adsr_window

    if isinstance(input, TimeSignal):
        return type(input)(samples_adsr, input.samplerate)
    else:
        return samples_adsr
