import one_sample_t_test
import scipy.stats
import numpy as np


def paired_two_sample_t_test(
    samples_x, samples_y, diff: float = 0.0, tail: str = "both"
):
    """Paired two-sample t-test to check if two population means are different.

    Data is assumed to be paired, i.e. len(samples_x) = len(samples_y)
    and x_i has a direct relation with y_i, for all 1 <= i <= n. For
    example, each i can represent a different person, and x_i is the
    measure of its blood pressure before a medical treatment, and y_i is
    its blood pressure after the medical treatment.

    The paired two-sample t-test is just a regular one-sample t-test for
    the differences w_i = x_i - y_i ~ N(mu, sigma^{2}) of each sample group
    X and Y, where both mu and sigma are unknown parameters.

    Assumptions:
        i.i.d x_1, ..., x_n ~ N(mu_x, sigma_x^{2})
        i.i.d y_1, ..., y_n ~ N(mu_y, sigma_y^{2})
        x_i is paired (is related) to y_i.
        The differences w_i = x_i - y_i ~ N(mu, sigma^{2})
        Note 1: emphasizing that both samples has the same size.

    Test statistic: (w_mean - diff) / w_mean_std
    where:
        w_i = x_i - y_i;
        w_mean = mean(w);
        w_mean_std = w_std / sqrt(n);
        `diff` is given as a user argument.

    Null distribution: W ~ t(n - 1), where t is the t-student distribution.

    H0: x_mean - y_mean = diff
    HA:
        if tail = `both` : x_mean - y_mean != diff;
        if tail = `left` : x_mean - y_mean < diff;
        if tail = `right`: x_mean - y_mean > diff.
    """
    assert len(samples_x) == len(samples_y)

    samples_x = np.asfarray(samples_x)
    samples_y = np.asfarray(samples_y)

    return one_sample_t_test.t_test(
        samples_x - samples_y, hypothesis_mean=diff, tail=tail
    )


def _test():
    sample_std = 6
    test_x = [12, 10, -5, -5, 0, 0, 0]
    test_y = [12, 11, -5, -4, 0, 1, -1]
    diffs = [0, 0, 0, -1, 0, -1, 1]
    for tail, tail_scipy in zip(
        ["both", "left", "right"], ["two-sided", "less", "greater"]
    ):
        for mean_x, mean_y, diff in zip(test_x, test_y, diffs):
            sample_x = mean_x + sample_std * np.random.randn(200)
            sample_y = mean_y + sample_std * np.random.randn(200)
            res = paired_two_sample_t_test(sample_x, sample_y, diff=diff, tail=tail)
            print(tail, res)
            if np.isclose(0, diff):
                assert np.allclose(
                    res,
                    scipy.stats.ttest_rel(sample_x, sample_y, alternative=tail_scipy),
                )


if __name__ == "__main__":
    _test()
