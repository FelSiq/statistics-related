import typing as t

import numpy as np
import scipy.stats


def t_test(
    samples: t.Sequence[float],
    hypothesis_mean: float,
    tail: str = "both",
):
    assert tail in {"both", "left", "right"}

    sample_mean = np.mean(samples)
    sample_var = np.var(samples, ddof=1)
    n = len(samples)

    statistic = (sample_mean - hypothesis_mean) * np.sqrt(n / sample_var)
    null_dist = scipy.stats.t(n - 1)

    if tail == "left":
        p_value = null_dist.cdf(statistic)

    elif tail == "right":
        p_value = null_dist.sf(statistic)

    else:
        p_value = 2.0 * null_dist.cdf(-abs(statistic))

    return statistic, p_value


def _test():
    for null_hypothesis_mean in (4, 6, 10, 11, 12, 13, 17):
        samples = 11 + 4 * np.random.randn(500)

        for tail, scipy_tail in zip(
            ("both", "left", "right"), ("two-sided", "less", "greater")
        ):
            statistic, p_value = t_test(
                samples, hypothesis_mean=null_hypothesis_mean, tail=tail
            )

            assert np.allclose(
                (statistic, p_value),
                scipy.stats.ttest_1samp(
                    samples, popmean=null_hypothesis_mean, alternative=scipy_tail
                ),
            )


if __name__ == "__main__":
    _test()
