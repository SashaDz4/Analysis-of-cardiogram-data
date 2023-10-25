import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from t1 import all_channels


def fourier_transform(data):
    # Perform the Fourier transform using the formulas from the lecture.
    # Calculate the amplitude spectrum and the phase spectrum.

    fourier_data = fft(data)
    amplitude_spectrum = np.abs(fourier_data)
    phase_spectrum = np.angle(fourier_data)

    # Calculate the frequency of the first sinusoid (or frequency step).
    frequency = fftfreq(len(data), 1 / 1000)[1]
    print(f"Frequency: {frequency}")

    # draw a graph of the amplitude spectrum
    plt.figure(figsize=(25, 5))
    plt.plot(amplitude_spectrum[:1500], color='red')
    plt.show()


def inverse_fourier_transform(data):
    # Perform the inverse Fourier transform using the formulas from the lecture.
    # Compare the results with the original data.

    fourier_data = fft(data)
    inverse_fourier_data = np.fft.ifft(fourier_data)

    # draw a graph of the amplitude spectrum
    plt.figure(figsize=(25, 5))
    plt.plot(inverse_fourier_data[:1500], color='red')
    plt.show()


if __name__ == "__main__":
    for i in range(12):
        print(f"Channel {i + 1}")
        fourier_transform(all_channels[:, i])
        inverse_fourier_transform(all_channels[:, i])
        if i == 1:
            break
