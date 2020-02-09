import numpy as np
from scipy import fft
from .signals import TimeSignal, get_samples, get_samples_and_rate, get_both_samples, same_type_as
from .util import adsr_window_log


def circ_convolve(signal1, signal2, num_points=None):
    samples1, samples2 = get_both_samples(signal1, signal2)
    if num_points == None:
        num_points = len(signal1)
    conv_samples = fft.irfft(fft.rfft(samples1, num_points) * fft.rfft(samples2, num_points))
    # conv_samples = signal.convolve(signal1, np.concatenate([signal2, signal2]), mode='same')
    return same_type_as(conv_samples, signal1)


def quantise(input_signal, depth, max_amp=1):
    """signal_quant = quantise(input, depth)

    quantises the input signal uniformly to "depth" bits
    input is assumed to fall within the range [-max_amp, max_amp]
    """
    samples = get_samples(input_signal)

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

    return same_type_as(samples_quant, input_signal)


def apply_adsr(input_signal, max_amp, sustain_amp, attack, decay, release, samplerate=None):
    samples, samplerate = get_samples_and_rate(input_signal, samplerate)

    attack_size = round(attack * samplerate)
    decay_size = round(decay * samplerate)
    release_size = round(release * samplerate)
    sustain_size = len(samples) - attack_size - decay_size - release_size
    adsr_window = adsr_window_log(max_amp, sustain_amp, attack_size, decay_size, sustain_size, release_size)
    samples_adsr = samples * adsr_window

    return same_type_as(samples_adsr, input_signal)
