from scipy import fft

def circ_convolution(signal1, signal2, num_points=None):
    # return signal.convolve(signal1, np.concatenate([signal2, signal2]), mode='same')
    if num_points == None:
        num_points = len(signal1)
    return fft.irfft(fft.rfft(signal1, num_points) * fft.rfft(signal2, num_points))