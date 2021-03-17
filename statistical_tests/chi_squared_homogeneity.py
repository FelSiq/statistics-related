import scipy.stats
import numpy as np

import chi_squared_goodness_of_fit


def chi_squared_homogeneity(*args, statistic: str = "likelihood_ratio"):
    """Chi-squared test for homogeneity.

    Checks whether all data groups follows the same data distribution.

    Assumptions:
        None.

    Test statistic:
        The same statistics from the Chi-squared test for goodness of
        fit. Check `chi_squared_goodness_of_fit.py` documentation for
        more specific info.

    Null distribution: Y ~ Chi^2((group_num - 1) * (bins_per_group - 1))
    Where:
        Y can be either G or X^{2} test statistic (again, check Chi-
        squared goodness of fit documentation for more info.);
        Chi^2 is the Chi^2 distribution.

    H0: all groups follows the same data distribution.
    HA: At least one group does not follow the same data distribution
        as the remaining groups.
    """
    assert len(set(map(len, args))) == 1

    args = np.asfarray(args)

    num_groups, bins_per_group = args.shape

    group_size = np.sum(args, axis=1)
    null_dist_pmf = np.sum(args, axis=0) / np.sum(args)

    expected_counts = np.outer(group_size, null_dist_pmf)

    ddof = (
        num_groups + bins_per_group - 1
    )  # df = (num_groups - 1) * (bins_per_group - 1)

    return chi_squared_goodness_of_fit.chi_squared_goodness_of_fit(
        data_counts=args.ravel(),
        expected_counts=expected_counts.ravel(),
        ddof=ddof,
        statistic=statistic,
    )


def _test():
    n = 500
    p = 0.7
    group_num = 4
    x = np.arange(n * p - 5, n * p + 6)

    expected_counts = n * scipy.stats.binom(n, p).pmf(x)
    group_counts = [
        np.ceil(np.maximum(0.0, expected_counts + np.random.randint(-20, 20)))
        for _ in range(group_num)
    ]
    group_counts[0][1] *= 4
    group_counts[1][2] *= 1.2

    print(expected_counts)

    res = chi_squared_homogeneity(*group_counts, statistic="pearson_chi_square")
    res_scipy = scipy.stats.chi2_contingency(
        np.asfarray(group_counts), correction=False
    )[:2]
    print(res, res_scipy)
    assert np.allclose(res, res_scipy)

    res = chi_squared_homogeneity(*group_counts, statistic="likelihood_ratio")
    print(res)
    assert np.allclose(
        res,
        scipy.stats.chi2_contingency(
            np.asfarray(group_counts), correction=False, lambda_="log-likelihood"
        )[:2],
    )


if __name__ == "__main__":
    _test()
