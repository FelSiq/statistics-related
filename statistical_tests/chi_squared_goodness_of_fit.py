import scipy.stats
import numpy as np


def g_test(data_counts, expected_counts, ddof: int = 1):
    """Chi-squared test for goodness of fit using likelihood ratio statistic."""
    return chi_squared_goodness_of_fit(
        data_counts, expected_counts, ddof, statistic="llikelihood_ratio"
    )


def chi_squared_goodness_of_fit(
    data_counts, expected_counts, ddof: int = 1, statistic: str = "likelihood_ratio"
):
    """Chi-squared test for goodness of fit.

    Checks whether a given count matches the expected counts from some
    known distribution.

    Assumptions:
        None.

    Test statistic:
        If statistic=`likelihood_ratio`, G = 2 * dot(O, log(O / E))
        If statistic=`pearson_chi_square`, X^{2} = sum((O - E)^2 / E)
    Where:
        `O` is the data_counts;
        `E` is the expected_counts;
        log is the natural log.

    Null distribution: G ~ Chi^2(df) and X^{2} ~ Chi^2(df), where Chi^2
        is the Chi^2 distribution. Also, G is approximately X^{2} under
        the null hypothesis, and usually df = data_bins - 1, but may vary
        depending on the problem.

    H0: the data distribution matches the expected distribution.
    HA: the data distribution follows another latent distribution.
    """
    assert statistic in {"likelihood_ratio", "pearson_chi_square"}

    if statistic == "likelihood_ratio":
        test_statistic = 2.0 * np.dot(
            data_counts, np.log(1e-8 + data_counts / expected_counts)
        )

    else:
        test_statistic = np.sum(
            np.square(data_counts - expected_counts) / expected_counts
        )

    null_dist = scipy.stats.chi2(len(data_counts) - ddof)
    p_value = null_dist.sf(test_statistic)

    return test_statistic, p_value


def _test():
    tosses = 140
    p = 0.85
    x = np.arange(p * tosses - 4, p * tosses + 5)
    expected_counts = tosses * scipy.stats.binom(tosses, p).pmf(x)
    data_counts = np.ceil(
        np.maximum(
            0, expected_counts + np.random.randint(-3, 4, size=expected_counts.size)
        )
    )
    print(data_counts)
    print(expected_counts)

    res = chi_squared_goodness_of_fit(
        data_counts, expected_counts, statistic="likelihood_ratio"
    )
    print(res)

    res = chi_squared_goodness_of_fit(
        data_counts, expected_counts, statistic="pearson_chi_square"
    )
    print(res)
    assert np.allclose(res, scipy.stats.chisquare(data_counts, expected_counts))


if __name__ == "__main__":
    _test()
