from .util import Audio
from abc import ABC, abstractmethod
import numpy as np
from scipy import fft, signal
from IPython.display import display
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
output_notebook()


def get_samples_and_rate(input, samplerate):
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
        raise TypeError('Only AudioSignals, Numpy arrays or other iterables are supported as input, not {}'.format(type(input)))
    return samples, samplerate


class Signal(ABC):

    @abstractmethod
    def plot(self, **fig_args):
        pass

    def _repr_html_(self):
        return show(self.plot())

    def display(self, **fig_args):
        show(self.plot(**fig_args))

class TimeSignal(Signal):

    def __init__(self, samples, samplerate):
        self.samples = samples
        self.samplerate = samplerate
        self.timepoints = np.arange(len(samples)) / samplerate

    def plot(self, **fig_args):
        fig = figure(width=800, height=400, x_axis_label='time [s]', y_axis_label='amplitude',
             tools='pan,wheel_zoom,box_zoom,zoom_in,zoom_out,save,reset', active_drag='pan')
        fig.line(self.timepoints, self.samples, line_width=2)
        return fig


class AudioSignal(TimeSignal):

    def __init__(self, samples, samplerate):
        super().__init__(samples, samplerate)

    def play(self):
        return display(Audio(self.samples, rate=self.samplerate, normalize=False))

    def plot(self, **fig_args):
        default_args = {
            'width': 900, 'height': 300, 
            'x_axis_label': 'time [s]', 'y_axis_label': 'amplitude', 
            'y_range': (-1, 1), 
            'tools': 'xpan,xwheel_zoom,box_zoom,xzoom_in,xzoom_out,save,reset', 
            'active_drag': 'xpan',
            'toolbar_location': 'above',
        }
        fig = figure(**{**default_args, **fig_args})
        fig.line(self.timepoints, self.samples, line_width=2)
        return fig


class Spectrum(Signal):

    def __init__(self, input, samplerate=None, num_bins=None, power=1, decibels=True):
        samples, samplerate = get_samples_and_rate(input, samplerate)

        if num_bins is None:
            num_bins = np.clip(len(samples), 2**10, 2**14)

        self.power = power
        self.decibels = decibels

        self.spectrum = np.abs(fft.rfft(samples, num_bins))
        self.frequencies = np.arange(len(self.spectrum)) * samplerate / num_bins

        if decibels:
            self.spectrum = power * 10 * np.log10(self.spectrum)
        else:
            self.spectrum **= power


    def plot(self, **fig_args):
        default_args = {
            'width': 900, 'height': 300, 
            'x_axis_label': 'frequency [Hz]', 'y_axis_label': 'amplitude',
            'tools': 'pan,wheel_zoom,box_zoom,zoom_in,zoom_out,save,reset', 
            'active_drag': 'pan',
            'toolbar_location': 'above',
        }
        if self.power == 2:
            default_args['y_axis_label'] = 'power'
        if self.decibels:
            default_args['y_axis_label'] += ' [dB]'
        fig = figure(**{**default_args, **fig_args})
        fig.line(self.frequencies, self.spectrum, line_width=2)
        return fig


class PowerSpectrum(Spectrum):
    def __init__(self, input, samplerate=None, num_bins=None, decibels=True):
        super().__init__(input, samplerate=samplerate, num_bins=num_bins, power=2, decibels=decibels)


class Spectrogram(Signal):

    def __init__(self, input, frame_duration, step_duration, samplerate=None, num_bins=None, power=1, decibels=True):
        self.array = None
    
