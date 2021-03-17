import typing as t

import numpy as np
import scipy.stats


def z_test(
    samples: t.Sequence[float],
    true_var: float,
    hypothesis_mean: float,
    tail: str = "both",
):
    assert tail in {"both", "left", "right"}
    assert true_var >= 0.0

    sample_mean = np.mean(samples)
    n = len(samples)

    statistic = (sample_mean - hypothesis_mean) * np.sqrt(n / true_var)
    null_dist = scipy.stats.norm(loc=0.0, scale=1.0)

    if tail == "left":
        p_value = null_dist.cdf(statistic)

    elif tail == "right":
        p_value = null_dist.sf(statistic)

    else:
        p_value = 2.0 * null_dist.cdf(-abs(statistic))

    return statistic, p_value


def _test():
    samples = 11 + 4 * np.random.randn(500)

    for tail in ("both", "left", "right"):
        print("tail:", tail)
        statistic, p_value = z_test(samples, true_var=16, hypothesis_mean=11, tail=tail)
        print("Test statistic:", statistic)
        print("Test p-value:", p_value)
        print("\n")


if __name__ == "__main__":
    _test()
