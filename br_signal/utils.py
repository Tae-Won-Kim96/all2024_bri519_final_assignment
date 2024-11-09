from scipy.signal import chirp
import numpy as np

# function for creating chirp signal using default parameters
def create_chirp(fs=1024, duration=2, start_freq=100, end_freq=200):
    # Generate chirp signal
    time = np.linspace(0,duration,fs*duration) #1-2
    chirp_signal = chirp(time, start_freq, duration, end_freq, method='quadratic') #1-3

    freq2 = 50
    # freq2 = 50 # for combining sin wave
    signal2 = np.sin(2 * np.pi * freq2 * time)
    chirp_signal += signal2

    n = len(time) 
    noise = np.random.randn(n) 
    noise = 0.1 * (noise - np.min(noise)) / (np.max(noise) - np.min(noise))  

    chirp_signal += noise
    
    return time, chirp_signal


def calc_freq(fs, chirp_signal, fft_result):
    nyquist = fs / 2
    fBase = np.linspace(0, nyquist, int(np.floor(len(chirp_signal) / 2) + 1))

    halfTheSignal = fft_result[:len(fBase)] 
    complexConjugate = np.conj(halfTheSignal)
    powe = halfTheSignal*complexConjugate

    return powe, fBase

# Forward and backward filtering
def mfilter(b, a, x):

    y_forward = np.zeros_like(x)
    y_backward = np.zeros_like(x)
    
    # Forward filtering manually
    for n in range(len(x)):
        y_forward[n] = b[0] * x[n]
        for i in range(1, len(b)):
            if n - i >= 0:
                y_forward[n] += b[i] * x[n - i] - a[i] * y_forward[n - i]

    # Reverse the forward filtered signal
    y_reversed = y_forward[::-1]

    # Backward filtering manually
    for n in range(len(y_reversed)):
        y_backward[n] = b[0] * y_reversed[n]
        for i in range(1, len(b)):
            if n - i >= 0:
                y_backward[n] += b[i] * y_reversed[n - i] - a[i] * y_backward[n - i]

    # Reverse again to get the final zero-phase filtered output
    y = y_backward[::-1]

    return y