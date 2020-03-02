import IPython.display
import numpy as np
from math import log10


class Audio(IPython.display.Audio):

    def __init__(self, data=None, filename=None, url=None, embed=None, rate=None, autoplay=False, normalize=False):
        if not normalize and data is not None:
            data_array = np.asarray(data)
            # convert non-floating point data to floating point in interval [-1, 1]
            if np.issubdtype(data_array.dtype, np.signedinteger):
                data = 1 / 2**(8*data_array.dtype.itemsize-1) * data_array
            elif np.issubdtype(data_array.dtype, np.unsignedinteger):
                data = 1 / 2**(8*data_array.dtype.itemsize-1) * data_array - 1
        try:
            super().__init__(data=data, filename=filename, url=url, embed=embed, rate=rate, autoplay=autoplay, normalize=normalize)
        except TypeError:
            if not normalize and data is not None:
                s = list(data.shape)
                s[-1] = 1
                data = np.append(data, np.ones(s), axis=-1)
            super().__init__(data=data, filename=filename, url=url, embed=embed, rate=rate, autoplay=autoplay)        


def adsr_window_log(max_amp, sustain_amp, attack_size, decay_size, sustain_size, release_size):
    # adsr_window_log(max_amp, sustain_amp, attack_size, decay_size, sustain_size, release_size)
    # 
    # generates ADSR window with
    # component lengths:         attack_size, decay_size, sustain_size, release_size in samples
    # component amplitudes:      max_amp, sustain_amp
    # 
    #  
    # Example adsr = adsr_window_log(1, 0.3, 128, 128, 512, 64)

    silence = 0.001

    # generate attack
    a = np.logspace(log10(silence), log10(max_amp), attack_size)

    # generate decay
    d = np.logspace(log10(max_amp), log10(sustain_amp), decay_size)

    # generate sustain
    s = np.full(sustain_size, sustain_amp)

    # generate release
    r = np.logspace(log10(sustain_amp), log10(silence), release_size)

    # concatenate
    return np.concatenate((a, d, s, r))
