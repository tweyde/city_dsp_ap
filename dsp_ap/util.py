import IPython.display
import numpy as np

class Audio(IPython.display.Audio):

    def __init__(self, data=None, filename=None, url=None, embed=None, rate=None, autoplay=False, normalize=False):
        try:
            super().__init__(data=data, filename=filename, url=url, embed=embed, rate=rate, autoplay=autoplay, normalize=normalize)
        except TypeError:
            if not normalize and data is not None:
                data = np.asarray(data)
                s = list(data.shape)
                s[-1] = 1
                data = np.append(data, np.ones(s), axis=-1)
            super().__init__(data=data, filename=filename, url=url, embed=embed, rate=rate, autoplay=autoplay)