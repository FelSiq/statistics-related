import scipy.stats
import numpy as np


def one_way_anova(*args):
    """One-way ANOVA to check if all groups have the same pop. mean.

    ANOVA stands for `Analysis of Variance`.

    Assumptions:
        i.i.d. x_{(1, 1)}, ..., x_{(1, m)} ~ N(mu_1, sigma^{2})
        i.i.d. x_{(2, 1)}, ..., x_{(2, m)} ~ N(mu_2, sigma^{2})
        ...
        i.i.d. x_{(n, 1)}, ..., x_{(n, m)} ~ N(mu_n, sigma^{2})

        Note 1: all groups (1 to n) has the same number of samples `m`.
        Note 2: all groups (1 to n) has the same pop. variance sigma^{2}.

    Test statistic: w = msb / msw
    where:
        msb = m * var(group_means);
        msw = mean(group_vars).

    Null distribution: W ~ F(n - 1, n * (m - 1)), where F is the F-distribution,
    `n` is the number of groups and `m` is the number of samples per group.

    H0: mu_1 = mu_2 = ... = mu_n
    HA: Exists at least a pair (i, j), 1 <= i, j <= n, such that mu_i != mu_j.
    """
    assert len(set(map(len, args))) == 1

    num_groups, num_inst_per_group = len(args), len(args[0])

    group_means = list(map(np.mean, args))
    group_vars = list(map(lambda arr: np.var(arr, ddof=1), args))

    msb = num_inst_per_group * np.var(group_means, ddof=1)
    msw = np.mean(group_vars)

    null_dist = scipy.stats.f(num_groups - 1, num_groups * (num_inst_per_group - 1))

    test_statistic = msb / msw
    p_value = null_dist.sf(test_statistic)

    return test_statistic, p_value


def f_test_for_equal_means(*args):
    """Also known as One-way ANOVA."""
    return one_way_anova(*args)


def _test():
    for n_groups in (2, 5, 10, 20):
        samples = [np.random.randn(50) for _ in range(n_groups)]
        res = one_way_anova(*samples)
        print(res)
        assert np.allclose(res, scipy.stats.f_oneway(*samples))

        samples[1] += 3 * np.random.random() - 1.5
        res = one_way_anova(*samples)
        print(res)
        assert np.allclose(res, scipy.stats.f_oneway(*samples))


if __name__ == "__main__":
    _test()
