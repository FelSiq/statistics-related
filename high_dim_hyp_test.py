"""Methods for testing a high number of statistical hypothesis."""
import typing as t

import numpy as np


def bonferroni(pvalues: np.ndarray,
               threshold: float = 0.05,
               return_array: bool = False) -> t.Union[int, np.ndarray]:
    """Uses the Bonferroni correction to hypothesis testing.

    Arguments
    ---------
    pvalues : :obj:`np.ndarray` shape = (test_num,), dtype = :obj:`float`
        Numpy array with of p-values extracted from the statistical
        test.

    threshold : :obj:`float`, optional
        Threshold, also known as ``significance level`` or ``Type I error
        rate``. The value used as threshold t to reject the null hypothesis
        of each p-value $p_{i}$ iff $p_{i} < t$.

    return_array: :obj:`bool`, optional
        If True, return a boolean numpy array with the null hypothesis
        rejection status (i.e., each rejected null hypothesis array
        position has a ``True`` value.)

    Returns
    -------
    If ``return_array`` is False:
        :obj:`int`
            Number of discoveries/statistically significant tests with
            the Bonferroni correction.
    Else:
        :obj:`np.ndarray` shape = (test_num,), dtype = :obj:`bool`
    """
    if not 0 <= threshold <= 1:
        raise ValueError(
            "'threshold' must be in [0, 1] range (got {}.)".format(threshold))

    tests = pvalues < threshold / pvalues.size

    if return_array:
        return tests

    return np.sum(tests)


def holm(pvalues: np.ndarray,
         threshold: float = 0.05,
         return_array: bool = False) -> t.Union[int, np.ndarray]:
    """Uses the Holm correction to hypothesis testing.

    Arguments
    ---------
    pvalues : :obj:`np.ndarray` shape = (test_num,), dtype = :obj:`float`
        Numpy array with of p-values extracted from the statistical
        test.

    threshold : :obj:`float`, optional
        Threshold, also known as ``significance level`` or ``Type I error
        rate``. The value used as threshold t to reject the null hypothesis
        of each p-value $p_{i}$ iff $p_{i} < t$.

    return_array: :obj:`bool`, optional
        If True, return a boolean numpy array with the null hypothesis
        rejection status (i.e., each rejected null hypothesis array
        position has a ``True`` value.)

    Returns
    -------
    If ``return_array`` is False:
        :obj:`int`
            Number of discoveries/statistically significant tests with
            the Holm correction.
    Else:
        :obj:`np.ndarray` shape = (test_num,), dtype = :obj:`bool`
    """
    if not 0 <= threshold <= 1:
        raise ValueError(
            "'threshold' must be in [0, 1] range (got {}.)".format(threshold))

    sorted_ind = np.argsort(pvalues)
    pvalues = pvalues[sorted_ind]

    tests = np.zeros(pvalues.size, dtype=bool)

    j = 0
    while j < pvalues.size:
        test_ind = sorted_ind[j]
        tests[test_ind] = True

        if pvalues[j] > threshold / (pvalues.size - j):
            tests[test_ind] = False
            j = pvalues.size

        j += 1

    if return_array:
        return tests

    return np.sum(tests)


def _test_bonf() -> None:
    import statsmodels.stats.multitest
    np.random.seed(16)
    num_tests = 100

    for i in np.arange(num_tests):
        pvals = np.random.exponential(scale=1 / 16, size=1000)
        res_a = statsmodels.stats.multitest.multipletests(
            pvals, alpha=0.05, method="bonferroni")[0]
        res_b = bonferroni(pvals, threshold=0.05, return_array=True)
        assert np.allclose(res_a, res_b)
        print("\rBonferroni test {}%...".format(100 * i / num_tests), end="")

    print("\rBonferroni test successfully.")


def _test_holm() -> None:
    import statsmodels.stats.multitest
    np.random.seed(16)
    num_tests = 200

    for i in np.arange(num_tests):
        pvals = np.random.exponential(scale=1 / 64, size=1000)
        res_a = statsmodels.stats.multitest.multipletests(
            pvals, alpha=0.05, method="holm")[0]
        res_b = holm(pvals, threshold=0.05, return_array=True)
        assert np.allclose(res_a, res_b)
        print("\rHolm test {}%...".format(100 * i / num_tests), end="")

    print("\rHolm test finished successfully.")


if __name__ == "__main__":
    _test_bonf()
    _test_holm()
