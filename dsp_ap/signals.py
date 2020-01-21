from .util import Audio
from abc import ABC, abstractmethod
import numpy as np
from scipy import fft
from IPython.display import display
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
output_notebook()



class Signal(ABC):

    @abstractmethod
    def plot(self):
        pass

    def _repr_html_(self):
        return show(self.plot())

class TimeSignal(Signal):

    def __init__(self, samples, samplerate):
        self.samples = samples
        self.samplerate = samplerate
        self.timepoints = np.arange(len(samples)) / samplerate

    def plot(self):
        fig = figure(width=800, height=400, x_axis_label='time [s]', y_axis_label='amplitude',
             tools='pan,wheel_zoom,zoom_in,zoom_out,reset', active_drag='pan')
        fig.line(self.timepoints, self.samples, line_width=2)
        return fig


class AudioSignal(TimeSignal):

    def __init__(self, samples, samplerate):
        super().__init__(samples, samplerate)

    def play(self):
        return display(Audio(self.samples, rate=self.samplerate, normalize=False))

    def plot(self):
        fig = figure(width=800, height=400, x_axis_label='time [s]', y_axis_label='amplitude', y_range=(-1, 1), 
             tools='xpan,xwheel_zoom,xzoom_in,xzoom_out,reset', active_drag='xpan')
        fig.line(self.timepoints, self.samples, line_width=2)
        return fig


class Spectrum(Signal):

    def __init__(self, input, samplerate=None, num_bins=None, power=1, decibels=True):
        if isinstance(input, AudioSignal):
            if samplerate is not None:
                print('Explicitly defined samplerate gets ignored when input is an AudioSignal', samplerate)
            samples = input.samples
            samplerate = input.samplerate
        elif np.ndim(input) > 0:
            if samplerate is None:
                raise ValueError('The samplerate needs to be defined explicitly when input is an array or other iterable')
            samples = np.asarray(input)
        else:
            raise TypeError('Only AudioSignals, Numpy arrays or other iterables are supported as input', type(input))

        if num_bins is None:
            num_bins = max(min(len(samples), 1024), 2**14)

        self.spectrum = np.abs(fft.rfft(samples, num_bins))
        self.frequencies = np.arange(len(self.spectrum)) * samplerate / num_bins

        if decibels:
            self.spectrum = power * 10 * np.log10(self.spectrum)
        else:
            self.spectrum **= power


    def plot(self):
        fig = figure(width=800, height=400, x_axis_label='frequency [Hz]', y_axis_label='power [dB]',
             tools='pan,wheel_zoom,zoom_in,zoom_out,reset', active_drag='pan')
        fig.line(self.frequencies, self.spectrum, line_width=2)
        return fig


class PowerSpectrum(Spectrum):
    def __init__(self, input, samplerate=None, num_bins=None, decibels=True):
        super().__init__(input, samplerate=samplerate, num_bins=num_bins, power=2, decibels=decibels)


class Spectrogram(Signal):

    def __init__(self, input, frame_duration, step_duration, samplerate=None, num_bins=None, power=1, decibels=True):
        self.array = None
    
