import numpy as np

def pad_to_power_of_two(signal):
    n = len(signal)
    next_power_of_two = 2**np.ceil(np.log2(n)).astype(int)
    padded_signal = np.pad(signal, (0, next_power_of_two - n), mode='constant')
    return padded_signal

    
#2,3,4,5 FFT Function
def fft(a):
    n = len(a)
    if n == 1: #5
        return a
    else:
        #2 Divide into even and odd samples and recursively apply the DFT
        even = fft(a[0::2]) #3
        odd = fft(a[1::2]) #3
        
        #4 Compute the twiddle factor
        t = [np.exp(-2j * np.pi * k / n) * odd[k] for k in range(n // 2)]
        
        # Combine even and odd with twiddle factors
        return [even[k] + t[k] for k in range(n // 2)] + [even[k] - t[k] for k in range(n // 2)]
