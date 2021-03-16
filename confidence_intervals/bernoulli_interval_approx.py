"""Estimate the confidence interval of a Bernoulli dist. parameter.

It is a conservative interval i.e. it overestimates the confidence
interval bounds.

Useful to find confidence intervals for machine learning models
accuracy.
"""
import scipy.stats
import numpy as np


def bernoulli_confidence_interval(
    accuracy: float, num_inst: int, confidence: float = 0.95
):
    crit_point = scipy.stats.norm.isf(0.5 * (1.0 - confidence))

    dist = 0.5 * crit_point / np.sqrt(num_inst)

    lower_bound = max(0.0, accuracy - dist)
    upper_bound = min(1.0, accuracy + dist)

    return (lower_bound, upper_bound)


def _test():
    accuracy = 0.98
    confidence = 0.95
    num_inst = 128

    print(bernoulli_confidence_interval(accuracy, num_inst, confidence))


if __name__ == "__main__":
    _test()
