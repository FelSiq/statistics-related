"""Compute confidence intervals for normal data.

Using Null Hypothesis Significance Testing (NHST) framework as
context:
A (1 - alpha) confidence interval is an interval statistic
centered at some test statistic which will contain the null
hypothesis value 100*(1 - alpha)% of the time.
"""
import scipy.stats
import numpy as np


def z_ci_for_mean(samples, true_var: float, confidence: float = 0.95):
    """Confidence interval for the sample mean.

    Assumptions:
        i.i.d. x_1, ..., x_n ~ N(mu, sigma^2)
    Where:
        mu is unknown;
        sigma^2 is known and provided by the user (`true_var`).

    Confidence = 1 - (Type I Error).
    """
    assert 0 <= confidence <= 1.0

    n = len(samples)
    sample_mean = np.mean(samples)

    crit_point = scipy.stats.norm.isf(0.5 * (1.0 - confidence))

    dist = crit_point * np.sqrt(true_var / n)

    lower_bound = sample_mean - dist
    upper_bound = sample_mean + dist

    return (lower_bound, upper_bound)


def t_ci_for_mean(samples, confidence: float = 0.95):
    """Confidence interval for the sample mean.

    Assumptions:
        i.i.d. x_1, ..., x_n ~ N(mu, sigma^2)
    Where:
        mu and sigma^2 is unknown.

    Confidence = 1 - (Type I Error).
    """
    assert 0 <= confidence <= 1.0

    n = len(samples)
    sample_mean = np.mean(samples)
    sample_var = np.var(samples, ddof=1)

    crit_point = scipy.stats.t(n - 1).isf(0.5 * (1.0 - confidence))

    dist = crit_point * np.sqrt(sample_var / n)

    lower_bound = sample_mean - dist
    upper_bound = sample_mean + dist

    return (lower_bound, upper_bound)


def chi_square_ci_for_variance(samples, confidence: float = 0.95):
    """Confidence interval for the sample variance.

    Assumptions:
        i.i.d. x_1, ..., x_n ~ N(mu, sigma^2)
    Where:
        mu and sigma^2 is unknown.

    Confidence = 1 - (Type I Error).
    """
    assert 0 <= confidence <= 1.0

    n = len(samples)
    sample_var = np.var(samples, ddof=n - 1)
    half_err_type_1_rate = 0.5 * (1.0 - confidence)

    dist = scipy.stats.chi2(n - 1)

    crit_point_low = dist.isf(half_err_type_1_rate)
    crit_point_upper = dist.ppf(half_err_type_1_rate)

    lower_bound = sample_var / crit_point_low
    upper_bound = sample_var / crit_point_upper

    return (lower_bound, upper_bound)


def _test():
    true_var = 64
    n_samples = 95
    true_mean = 3
    samples = true_mean + np.sqrt(true_var) * np.random.randn(n_samples)
    print(z_ci_for_mean(samples, true_var))
    print(t_ci_for_mean(samples))
    print(chi_square_ci_for_variance(samples))


if __name__ == "__main__":
    _test()
