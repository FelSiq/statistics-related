import scipy.stats
import numpy as np


def chi_squared_goodness_of_fit(
    data_counts, expected_counts, ddof: int = 1, statistic: str = "likelihood_ratio"
):
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
