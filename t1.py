import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro

# Load the dataset
all_channels = np.loadtxt('A25.txt', delimiter=',')


def basic_statistical_parameters(data):
    # calculate mean
    mean = np.mean(data)
    print(f"Mean: {mean}")

    # calculate harmonic mean
    harmonic_mean = np.mean(1 / data)
    print(f"Harmonic mean: {harmonic_mean}")

    # calculate geometric mean
    geometric_mean = np.exp(np.mean(np.log(data)))
    if geometric_mean != geometric_mean or geometric_mean:
        print(f"Geometric mean: {geometric_mean}. Geometric mean is negative")

    # calculate dispersion
    dispersion = np.std(data)
    print(f"Dispersion: {dispersion}")

    # calculate the average Gini difference
    gini = np.mean(np.abs(np.subtract.outer(data, data)))
    print(f"Gini: {gini}")

    # calculate median
    median = np.median(data)
    print(f"Median: {median}")

    # calculate skewness coefficient
    skewness = np.mean((data - mean) ** 3) / (dispersion ** 3)
    print(f"Skewness: {skewness}")

    # calculate kurtosis coefficient
    kurtosis = np.mean((data - mean) ** 4) / (dispersion ** 4)
    print(f"Kurtosis: {kurtosis}")

    # construct a histogram
    plt.hist(data, bins=100)
    plt.show()

    # test the hypothesis of the normal distribution law using the Shapiro-Wilk test
    stat, p = shapiro(data)
    print('Statistics: %.3f, p: %.3f' % (stat, p))


def plot_cardiogram(data):
    plt.figure(figsize=(25, 5))
    plt.plot(range(0, len(data) * 2, 2), data, color='red')
    plt.show()


if __name__ == "__main__":
    file_name = 'A23.txt'

    for i in range(all_channels.shape[1]):
        print(f"Chanel {i + 1}")
        channel = all_channels[:, i]

        # print basic statistical parameters
        basic_statistical_parameters(channel)

        # plot data
        plot_cardiogram(channel)

        # Normalize data
        channel = (channel - np.mean(channel)) / np.std(channel)

        # plot data
        plot_cardiogram(channel)
        print()
