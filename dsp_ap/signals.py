from .util import Audio
from abc import ABC, abstractmethod
import numpy as np
from scipy import fft, signal
from IPython.display import display
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.models.mappers import LinearColorMapper
from bokeh.models.ranges import DataRange1d
from bokeh.models.tools import HoverTool
from bokeh.palettes import Viridis256
from bokeh.io import output_notebook
output_notebook()


def get_samples_and_rate(input_signal, samplerate):
    if isinstance(input_signal, TimeSignal):
        if samplerate is not None:
            print('Explicitly defined samplerate gets ignored when input is a TimeSignal', samplerate)
        samples = input_signal.samples
        samplerate = input_signal.samplerate
    elif np.ndim(input_signal) > 0:
        if samplerate is None:
            raise ValueError('The samplerate needs to be defined explicitly when input is an array or other iterable')
        samples = np.asarray(input_signal)
    else:
        raise TypeError('Only TimeSignals, Numpy arrays or other iterables are supported as input, not {}'.format(type(input_signal)))
    return samples, samplerate


def get_samples(input_signal):
    if isinstance(input_signal, TimeSignal):
        return input_signal.samples
    elif np.ndim(input_signal) > 0:
        return np.asarray(input_signal)
    else:
        raise TypeError('Only TimeSignals, Numpy arrays or other iterables are supported as input, not {}'.format(type(input_signal)))


def get_both_samples_and_rate(input_signal1, input_signal2, samplerate=None):
    samples1, samplerate1 = get_samples_and_rate(input_signal1, samplerate)
    samples2, samplerate2 = get_samples_and_rate(input_signal2, samplerate)
    if samplerate1 != samplerate2:
        raise ValueError('Both signals need to have the same samplerate')
    return samples1, samples2, samplerate1


def get_both_samples(input_signal1, input_signal2):
    samples1 = get_samples(input_signal1)
    samples2 = get_samples(input_signal2)
    if isinstance(input_signal1, TimeSignal) and isinstance(input_signal2, TimeSignal) and input_signal1.samplerate != input_signal2.samplerate:
        raise ValueError('Both signals need to have the same samplerate')
    return samples1, samples2


def same_type_as(output_samples, input_signal):
    if isinstance(input_signal, TimeSignal):
        return type(input_signal)(output_samples, input_signal.samplerate)
    else:
        return output_samples


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

    def play(self, normalize=False):
        return display(Audio(self.samples, rate=self.samplerate, normalize=normalize))

    def plot(self, **fig_args):
        default_args = {
            'width': 900, 'height': 300, 
            'x_axis_label': 'time [s]', 'y_axis_label': 'amplitude', 
            'y_range': (-1, 1), 
            'tools': 'xpan,xwheel_zoom,box_zoom,xzoom_in,xzoom_out,save,reset', 
            'active_drag': 'xpan',
            'active_inspect': None,
            'active_scroll': None,
            'toolbar_location': 'above',
        }
        hover_tool = HoverTool(
            tooltips=[('time [s]', '$x{0.000}'), ('amplitude', '$y{0.000}')],
            mode='vline',
        )
        fig = figure(**{**default_args, **fig_args})
        fig.line(self.timepoints, self.samples, line_width=2)
        fig.add_tools(hover_tool)
        return fig


class Spectrum(Signal):

    def __init__(self, input, samplerate=None, num_bins=None, power=1, decibels=True):
        samples, samplerate = get_samples_and_rate(input, samplerate)

        if num_bins is None:
            num_bins = len(samples)

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
            'active_inspect': None,
            'active_scroll': None,
            'toolbar_location': 'above',
        }
        hover_tool = HoverTool(
            tooltips=[('frequency [Hz]', '$x{0.}'), ['amplitude', '$y{0.000}']],
            mode='vline',
        )
        if self.power == 2:
            default_args['y_axis_label'] = 'power'
            hover_tool.tooltips[1][0] = 'power'
        if self.decibels:
            default_args['y_axis_label'] += ' [dB]'
            hover_tool.tooltips[1][0] += ' [dB]'
        fig = figure(**{**default_args, **fig_args})
        fig.line(self.frequencies, self.spectrum, line_width=2)
        fig.add_tools(hover_tool)
        return fig


class PowerSpectrum(Spectrum):
    def __init__(self, input, samplerate=None, num_bins=None, decibels=True):
        super().__init__(input, samplerate=samplerate, num_bins=num_bins, power=2, decibels=decibels)


class Spectrogram(Signal):

    def __init__(self, input_signal, frame_duration, step_duration, samplerate=None, num_bins=None, window='hann', power=1, decibels=True):
        samples, samplerate = get_samples_and_rate(input_signal, samplerate)

        self.power = power
        self.decibels = decibels

        frame_size = round(frame_duration * samplerate)
        overlap_size = round((frame_duration-step_duration) * samplerate)

        self.frequencies, self.times, self.array = signal.stft(samples, fs=samplerate, window=window, nperseg=frame_size, noverlap=overlap_size)

        if decibels:
            self.array = power * 10 * np.log10(self.array)
        else:
            self.array **= power

    def plot(self, lowest_value=None, highest_value=None, palette=None, **fig_args):
        if not palette:
            palette = list(reversed(Viridis256))
        if not lowest_value:
            lowest_value = np.min(np.abs(self.array))
        if not highest_value:
            highest_value = np.max(np.abs(self.array))
        
        default_args = {
            'width': 900, 'height': 400, 
            'x_axis_label': 'time [s]', 'y_axis_label': 'frequency [Hz]',
            'tools': 'hover,pan,wheel_zoom,box_zoom,zoom_in,zoom_out,save,reset',
            'active_drag': 'pan',
            'active_inspect': None,
            'active_scroll': None,
            'toolbar_location': 'above',
            'tooltips': [('time [s]', '$x{0.000}'), ('frequency [Hz]', '$y{0.}'), ['amplitude', '@image']],
        }

        if self.power == 2:
            default_args['tooltips'][2][0] = 'power'
        if self.decibels:
            default_args['tooltips'][2][0] += ' [dB]'

        fig = figure(**{**default_args, **fig_args})
        if isinstance(fig.x_range, DataRange1d):
            fig.x_range.range_padding = 0
        if isinstance(fig.y_range, DataRange1d):
            fig.y_range.range_padding = 0
        mapper = LinearColorMapper(palette=palette, low=lowest_value, high=highest_value)
        fig.image([np.abs(self.array)], x=self.times[0], y=self.frequencies[0], dw=self.times[-1], dh=self.frequencies[-1], color_mapper=mapper)
        return fig
