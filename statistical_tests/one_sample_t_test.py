import typing as t

import numpy as np
import scipy.stats


def t_test(
    samples: t.Sequence[float],
    hypothesis_mean: float,
    tail: str = "both",
):
    """One sample t-test to check if a population have a hypothesized mean.

    Assumptions:
        i.i.d. x_{1}, ..., x_{n} ~ N(mu, sigma^{2}), where both mu and sigma
        are unknown values.

    Test statistic: t = (x_mean - hypothesis_mean) / x_mean_std
    where:
        x_mean_std = x_sample_std / sqrt(n)

    Null hypothesis: T ~ t(n - 1), where t is the t-student distribution.

    H0: population has mean = `hypothesis_mean`
    HA:
        if tail = `both` : population mean is different than `hypothesis_mean`;
        if tail = `left` : population mean < `hypothesis_mean`;
        if tail = `right`: population mean > `hypothesis_mean`.
    """
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
