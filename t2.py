from scipy.stats import f_oneway
from t1 import all_channels


def check_one_factor_anova(data, items=12):
    # One-factor analysis. The null hypothesis is that the means of the samples are equal.
    return f_oneway(*[data[:, i] for i in range(items)])


def check_two_factor(data):
    print("\nTwo-factor analysis of variance")
    data = data.reshape(5, 12, 1000)
    # check two-factor analysis of variance for each channel with each channel
    # calculate the average value for each channel
    print("\nFactor A: channels")
    for i in range(12):
        print(f"Channel {i + 1}")
        stat, p = check_one_factor_anova(data[:, i, :], items=5)
        print('Statistics: %.3f, p: %.3f' % (stat, p))

    print("\nFactor B: channels")
    for i in range(5):
        print(f"Channel {i + 1}")
        stat, p = check_one_factor_anova(data[i, :, :], items=12)
        print('Statistics: %.3f, p: %.3f' % (stat, p))


if __name__ == "__main__":
    print("One-factor analysis of variance")
    stat, p = check_one_factor_anova(all_channels)
    print('Statistics: %.3f, p: %.3f' % (stat, p))

    # check two-factor analysis of variance
    check_two_factor(all_channels)
