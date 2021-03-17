import scipy.stats
import numpy as np


def two_sample_t_test_equal_var(
    samples_x, samples_y, diff: float = 0.0, tail: str = "both"
):
    """Two sample t-test for normal data to check if pop. mean differs by `diff`.

    Assumptions:
        i.i.d. x_{1}, ..., x_{n} ~ N(mu_{x}, sigma^{2})
        i.i.d. y_{1}, ..., y_{m} ~ N(mu_{y}, sigma^{2})
        Note 1: that the variance for all distributions are the same, sigma^{2}.
        Note 2: `n` may be different from `m`.

    Test statistic: t = (x_mean - y_mean - diff) / sqrt(pooled_var)
    where:
        pooled_var = ((n - 1) * sample_var_x + (m - 1) * sample_var_y) * norm_factor,
        norm_factor = (1 / n + 1 / m) / (n + m - 2)

    null distribution: T ~ t(n + m - 2), there t is the t-student distribution.

    H0: x_mean - y_mean = `diff`
    HA:
        if tail = `both` : x_mean - y_mean != `diff`
        if tail = `left` : x_mean - y_mean < `diff`
        if tail = `right`: x_mean - y_mean > `diff`
    """
    assert tail in {"both", "left", "right"}

    mean_x = np.mean(samples_x)
    mean_y = np.mean(samples_y)

    n_x = len(samples_x)
    n_y = len(samples_y)

    var_x = np.var(samples_x, ddof=n_x - 1)
    var_y = np.var(samples_y, ddof=n_y - 1)

    pooled_var = (var_x + var_y) / (n_x + n_y - 2) * (1.0 / n_x + 1.0 / n_y)

    test_statistic = (mean_x - mean_y - diff) / np.sqrt(pooled_var)

    null_dist = scipy.stats.t(n_x + n_y - 2)

    if tail == "both":
        p_value = 2.0 * null_dist.cdf(-abs(test_statistic))

    elif tail == "left":
        p_value = null_dist.cdf(test_statistic)

    else:
        p_value = null_dist.sf(test_statistic)

    return test_statistic, p_value


def _test():
    sample_std = 6
    for tail, tail_scipy in zip(
        ["both", "left", "right"], ["two-sided", "less", "greater"]
    ):
        for mean_x, mean_y in zip(
            [12, 10, -5, -5, 0, 0, 0], [12, 11, -5, -4, 0, 1, -1]
        ):
            sample_x = mean_x + sample_std * np.random.randn(200)
            sample_y = mean_y + sample_std * np.random.randn(100)
            res = two_sample_t_test_equal_var(sample_x, sample_y)
            print(tail, res)
            print(
                scipy.stats.ttest_ind(
                    sample_x, sample_y, equal_var=True, alternative=tail_scipy
                ),
            )
            assert np.allclose(
                res,
                scipy.stats.ttest_ind(
                    sample_x, sample_y, equal_var=True, alternative=tail_scipy
                ),
            )


if __name__ == "__main__":
    _test()
