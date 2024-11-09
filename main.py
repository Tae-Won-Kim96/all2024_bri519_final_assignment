import numpy as np
from scipy.signal import chirp, butter, cheby1, cheby2, spectrogram
from scipy.signal.windows import chebwin
import matplotlib.pyplot as plt
from br_signal import utils, FFT
from scipy.io import loadmat
from tqdm import tqdm
import os


def main():
    if os.path.isdir('figure'):
        pass
    else:
        os.mkdir("figure")

    # Setting parameters
    fs = 1024
    duration = 2
    start_freq = 100
    end_freq = 200
    nyquist = fs / 2
    # Generate a chirp signal
    time, chirp_signal = utils.create_chirp(fs=fs, duration=duration, start_freq=start_freq, end_freq=end_freq)

    plt.figure()
    plt.plot(time, chirp_signal, c='k', lw=0.5)
    plt.title('Chirp signal')
    plt.savefig('figure/App_original_chirp_signal.png')

    # Performing an FFT
    fft_result = np.array(FFT.fft(chirp_signal))
    powe, fBase = utils.calc_freq(fs, chirp_signal, fft_result / len(chirp_signal))

    # Power spectrum of the original signal
    plt.figure()
    plt.plot(fBase / nyquist, powe, c='k', lw=1)
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Power')
    plt.xlim([0, 1])
    plt.title('Power of Chirp Signal')
    plt.savefig('figure/App_original_power_signal.png')

    # Filter settings (digital filters)
    normalized_cutoff = 0.1/nyquist
    B_low, A_low = butter(5, normalized_cutoff, btype='low', analog=False)
    B_high, A_high = butter(5, normalized_cutoff, btype='high', analog=False)

    # Apply filtering
    chirp_signal_low = utils.mfilter(B_low, A_low, chirp_signal)
    chirp_signal_high = utils.mfilter(B_high, A_high, chirp_signal)

    # Performing an FFT and calculating power after filtering
    fft_original = FFT.fft(chirp_signal)
    fft_low = FFT.fft(chirp_signal_low)
    fft_high = FFT.fft(chirp_signal_high)

    # Frequency and power calculations (using normalised frequency bands)
    pow_ori, o_fBase = utils.calc_freq(fs, chirp_signal, fft_original)
    powe_low, l_fBase = utils.calc_freq(fs, chirp_signal_low, fft_low)
    powe_high, h_fBase = utils.calc_freq(fs, chirp_signal_high, fft_high)

    o_fBase_normalized = o_fBase / nyquist
    l_fBase_normalized = l_fBase / nyquist
    h_fBase_normalized = h_fBase / nyquist

    # Power spectrum plot of the original signal
    plt.figure()
    plt.plot(o_fBase / nyquist, pow_ori, c='k', lw=1)
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Power')
    plt.xlim([0, 1])
    plt.title('Original Signal Power (Normalized Frequency)')
    plt.savefig('figure/(1_4_1)_original_power_signal.png')

    # Power spectrum plot of low-pass filtered signal
    plt.figure()
    plt.plot(l_fBase / nyquist, powe_low, c='k', lw=1)
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Power')
    plt.xlim([0, 1])
    plt.title('Low-pass Filtered Signal Power (Normalized Frequency)')
    plt.savefig('figure/(1_4_2)_lowpass_fft_signal.png')

    # Power spectrum plot of high-pass filtered signal
    plt.figure()
    plt.plot(h_fBase / nyquist, powe_high, c='k', lw=1)
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Power')
    plt.xlim([0, 1])
    plt.title('High-pass Filtered Signal Power (Normalized Frequency)')
    plt.savefig('figure/(1_4_3)_highpass_fft_signal.png')


    windLength = 256
    overl = windLength-1
    freqBins = 250


    wind=np.kaiser(windLength,0)
    frequencies, times, Sxx = spectrogram(chirp_signal,fs,wind,len(wind),overl)
    wind=np.hanning(windLength)
    frequenciesh, timesh, Sxxh = spectrogram(chirp_signal,fs,wind,len(wind),overl)
    wind= chebwin(windLength, at=100)
    frequenciesc, timesc, Sxxc = spectrogram(chirp_signal,fs,wind,len(wind))


    # Spectrogram visualisation
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, frequencies, Sxx)
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram of Chirp Signal with kaiser window')
    plt.savefig('figure/(1_5_1)_Kaiser_window.png')

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(timesh, frequenciesh, Sxxh)
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram of Chirp Signal with hanning window')
    plt.savefig('figure/(1_5_2)_Hanning_window.png')

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(timesc, frequenciesc, Sxxc)
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram of Chirp Signal with cheb window')
    plt.savefig('figure/(1_5_3)_Chev_window.png')

    print('Part 1 ends')



    #########################################
    # MouseLFP analysis                     #
    #########################################
    # 1
    # Initialization and Define constants
    fs = 10000
    cutoff = 1000
    num_trials = 200
    num_sessions = 4
    wind_length = 256
    overlap = 255

    #2
    # Load Dataset 
    # C:\Users\Auros\Documents\code\bri519_fall2024\assignment\midterm
    data_file = 'data/mouseLFP.mat'
    data = loadmat(data_file)
    DATA = data['DATA']
    data_samples = DATA[0, 0].shape[1]

    ############################################################
    # Filtering with various filters(butter, cheb1, cheb2)     #
    # Saved as 2_Plot_low_high..                               #
    ############################################################

    # Low-pass Butterworth Filter 
    order = 5  # Filter order
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    b_butter, a_butter = butter(order, normal_cutoff, btype='low', analog=False)
    low_pass_data_butter = np.zeros((num_sessions, num_trials, data_samples))

    for session in range(num_sessions):
        for trial in tqdm(range(num_trials)):
            low_pass_data_butter[session, trial, :] = utils.mfilter(b_butter, a_butter, DATA[session, 0][trial, :])
        print(f'Butterworth Session {session+1} Ends')


    # Chebyshev Type I Filter
    ripple = 0.5

    b_cheby1, a_cheby1 = cheby1(order, ripple, normal_cutoff, btype='low', analog=False)
    filtered_data_cheby1 = np.zeros((num_sessions, num_trials, data_samples))

    for session in range(num_sessions):
        for trial in tqdm(range(num_trials)):
            filtered_data_cheby1[session, trial, :] = utils.mfilter(b_cheby1, a_cheby1, DATA[session, 0][trial, :])
        print(f'Chev type1 Session {session+1} Ends')

    # Chebyshev Type II Filter
    ripple_stopband = 0.5

    b_cheby2, a_cheby2 = cheby2(order, ripple_stopband, normal_cutoff, btype='low', analog=False)
    filtered_data_cheby2 = np.zeros((num_sessions, num_trials, data_samples))

    for session in range(num_sessions):
        for trial in tqdm(range(num_trials)):
            filtered_data_cheby2[session, trial, :] = utils.mfilter(b_cheby2, a_cheby2, DATA[session, 0][trial, :])
        print(f'Chev type2 Session {session+1} Ends')


    # Plotting the filtered results for visual comparison
    plt.figure(figsize=(15, 15))
    for session in range(num_sessions):
        # Butterworth Filter
        plt.subplot(num_sessions, 3, session * 3 + 1)
        plt.plot(low_pass_data_butter[session, 0, :], label='Trial 1', alpha=0.5)
        plt.plot(low_pass_data_butter[session, 1, :], label='Trial 2', alpha=0.5)
        plt.title(f'Butterworth Filtered Data - Session {session + 1}')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()

        # Chebyshev Type I Filter
        plt.subplot(num_sessions, 3, session * 3 + 2)
        plt.plot(filtered_data_cheby1[session, 0, :], label='Trial 1', alpha=0.5)
        plt.plot(filtered_data_cheby1[session, 1, :], label='Trial 2', alpha=0.5)
        plt.title(f'Chebyshev Type I Filtered Data - Session {session + 1}')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()

        # Chebyshev Type II Filter
        plt.subplot(num_sessions, 3, session * 3 + 3)
        plt.plot(filtered_data_cheby2[session, 0, :], label='Trial 1', alpha=0.5)
        plt.plot(filtered_data_cheby2[session, 1, :], label='Trial 2', alpha=0.5)
        plt.title(f'Chebyshev Type II Filtered Data - Session {session + 1}')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()

        print(f'Session {session+1} complete')

    plt.savefig(f'figure/(2_filtering)_Comparing_filters_{session+1}.png')


    ##############################
    # session low-high tone plot #
    # Saved as 2_Plot_low_high.. #
    ##############################

    filtered_data = low_pass_data_butter
    tone_info = np.zeros((num_sessions, num_trials))

    for session in range(num_sessions):
        tone_info[session] = DATA[session, 4].flatten()  # The fifth column stores the tone information
        for trial in range(num_trials):
            filtered_data[session, trial, :] = utils.mfilter(b_butter, a_butter, DATA[session, 0][trial, :])

    # Defining ERP and spectrogram generation functions
    def plot_erp_and_spectrogram(session_data, tone_type, fs):
        mean_signal = np.mean(session_data, axis=0)
        sem_signal = np.std(session_data, axis=0) / np.sqrt(session_data.shape[0])
        
        # ERP plots
        plt.subplot(2, 2, 1 if tone_type == 'Low' else 2)
        plt.plot(mean_signal, label=f'{tone_type} Tone ERP')
        plt.fill_between(range(len(mean_signal)), mean_signal - sem_signal, mean_signal + sem_signal, color='gray', alpha=0.3)
        plt.xlabel('Time [ms]')
        plt.ylabel('Voltage [V]')
        plt.title(f'{tone_type} Tone ERP')
        plt.legend()

        # Spectrogram Plot
        plt.subplot(2, 2, 3 if tone_type == 'Low' else 4)
        f, t, Sxx = spectrogram(mean_signal, fs=fs, nperseg=200, noverlap=150, nfft=400, scaling='spectrum')
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.colorbar(label='Power/Frequency (dB/Hz)')
        plt.ylim(0, 200)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.title(f'{tone_type} Tone Spectrogram')


    window_sizes = [1, wind_length // 2, wind_length]  # Window sizes for spectrograms
    mean_low_tones = np.zeros((num_sessions, filtered_data.shape[2]))  # (number of sessions, number of samples)
    mean_high_tones = np.zeros((num_sessions, filtered_data.shape[2]))

    cnt=1
    for session in range(num_sessions):
        plt.figure(figsize=(12, 8))
        plt.suptitle(f'Session {session + 1} ERP and Spectrogram')

        # Extracting unique tone values per session
        unique_tones = np.unique(tone_info[session])
        low_tone_value, high_tone_value = min(unique_tones), max(unique_tones)
        
        # Distinguishing low and high tones
        low_tone_indices = np.where(tone_info[session] == low_tone_value)[0]
        high_tone_indices = np.where(tone_info[session] == high_tone_value)[0]
        
        low_tone_data = filtered_data[session, low_tone_indices, :]
        high_tone_data = filtered_data[session, high_tone_indices, :]

        # Generate ERPs and spectrograms for bass and treble tones
        if len(low_tone_data) > 0:
            mean_low_tones[session, :] = np.mean(low_tone_data, axis=0)
            plot_erp_and_spectrogram(low_tone_data, 'Low', fs)
        else:
            print(f"Session {session + 1}: No low-tone trials found.")

        if len(high_tone_data) > 0:
            mean_high_tones[session, :] = np.mean(high_tone_data, axis=0)
            plot_erp_and_spectrogram(high_tone_data, 'High', fs)
        else:
            print(f"Session {session + 1}: No high-tone trials found.")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'figure/(2_low_high)_Plot_low_high_tones_{cnt}.png')
        cnt+=1

    ##############################
    # Mean LFPs and Spectrograms #
    # Saved as 2_MeanLFP_...     #
    ##############################


    cnt=1
    for session in range(num_sessions):
        plt.figure(figsize=(15, 10))

        # Left Column - Mean LFPs
        plt.subplot(2, 2, 1)  # Low-tone Mean LFP
        plt.plot(mean_low_tones[session, :], color='k')
        plt.title(f'Mean Low-tone Response - Session {session + 1}')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid()

        plt.subplot(2, 2, 2)  # High-tone Mean LFP
        plt.plot(mean_high_tones[session, :], color='k')
        plt.title(f'Mean High-tone Response - Session {session + 1}')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid()

        # Right Column - Spectrograms
        for idx, wind in enumerate(window_sizes):
            plt.subplot(2, 3, idx + 4)  # Spectrograms for Low tone
            f, t, Sxx = spectrogram(mean_low_tones[session, :], fs, nperseg=wind, noverlap=wind // 2, window='hann')
            plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='jet', vmax=0, vmin=-80)
            plt.title(f'Spectrogram - Session {session + 1} - Low Tone (Window Size: {wind})')
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (s)')
            plt.colorbar(label='Power (dB)')
            plt.ylim([0, 200])

        for idx, wind in enumerate(window_sizes):
            plt.subplot(2, 3, idx + 4)  # Spectrograms for High tone
            f, t, Sxx = spectrogram(mean_high_tones[session, :], fs, nperseg=wind, noverlap=wind // 2, window='hann')
            plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='jet', vmax=0, vmin=-80)
            plt.title(f'Spectrogram - Session {session + 1} - High Tone (Window Size: {wind})')
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (s)')
            plt.colorbar(label='Power (dB)')
            plt.ylim([0, 200])

        plt.tight_layout()
        plt.savefig(f'figure/(2_MeanLFP_Spectrogram)_Plot_session_{session + 1}.png')

if __name__ == "__main__":
    main()