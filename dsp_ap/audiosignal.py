from IPython.display import Audio, display


class AudioSignal:

    def __init__(self, samples, samplerate):
        self.samples = samples
        self.samplerate = samplerate

    def play(self):
        return display(Audio(self.samples, rate=self.samplerate))

