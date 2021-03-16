"""Wilson score interval.

Gives a confidence interval for the mean of data sampled from i.i.d
Bernoulli distributions B(p).

In practice, this is extremely useful since it can be used to estimate
confidence intervals for machine learning models accuracy (and other
evaluation metrics as well).

source:
https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
"""
import scipy.stats
import numpy as np


def wilson_score_continuity_correction(
    accuracy: float, num_inst: int, confidence: float = 0.95
):
    crit_point = scipy.stats.norm.ppf(1 - 0.5 * (1.0 - confidence))
    crit_point_sqr = crit_point ** 2

    denom = 2.0 * (num_inst + crit_point_sqr)
    mean_value = 2.0 * num_inst * accuracy

    dist = 1.0 + crit_point * np.sqrt(
        crit_point_sqr
        - 1 / num_inst
        + 4 * num_inst * accuracy * (1.0 - accuracy)
        + (4.0 * accuracy - 2.0)
    )

    lower_bound = (mean_value - dist) / denom
    upper_bound = (mean_value + dist) / denom

    lower_bound = max(0.0, lower_bound)
    upper_bound = min(1.0, upper_bound)

    return (lower_bound, upper_bound)


def wilson_score_interval(
    accuracy: float, num_inst: int, confidence: float = 0.95, conservative: bool = False
):
    crit_point = scipy.stats.norm.ppf(1 - 0.5 * (1.0 - confidence))
    crit_point_sqr_over_n = crit_point ** 2 / num_inst

    coeff_a = 1.0 / (1 + crit_point_sqr_over_n)
    coeff_b = accuracy * (1.0 - accuracy) if not conservative else 0.5

    mean_value = coeff_a * (accuracy + 0.5 * crit_point_sqr_over_n)
    dist = (
        crit_point
        * coeff_a
        * np.sqrt((coeff_b + 0.25 * crit_point_sqr_over_n) / num_inst)
    )

    lower_bound = mean_value - dist
    upper_bound = mean_value + dist

    if conservative:
        lower_bound = max(lower_bound, 0.0)
        upper_bound = min(upper_bound, 1.0)

    return (lower_bound, upper_bound)


def _test():
    accuracy = 0.9
    confidence = 0.975
    num_inst = 150

    print(wilson_score_interval(accuracy, num_inst, confidence, True))
    print(wilson_score_interval(accuracy, num_inst, confidence, False))
    print(wilson_score_continuity_correction(accuracy, num_inst, confidence))


if __name__ == "__main__":
    _test()
