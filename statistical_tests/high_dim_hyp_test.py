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
        tests.

    threshold : :obj:`float`, optional
        Threshold, also known as ``significance level`` or ``Type I error
        rate``. The value used as threshold $t$ to reject the null hypothesis
        of each p-value $p_{i}$ iff $p_{i} < t$.

    return_array: :obj:`bool`, optional
        If True, return a boolean numpy array with the null hypothesis
        rejection status (i.e., each rejected null hypothesis array
        position has a ``True`` value.)

    Returns
    -------
    If ``return_array`` is False:
        :obj:`int`
            Number of discoveries (statistically significant) tests with
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
        tests.

    threshold : :obj:`float`, optional
        Threshold, also known as ``significance level`` or ``Type I error
        rate``. The value used as threshold $t$ to reject the null hypothesis
        of each p-value $p_{i}$ iff $p_{i} < t$.

    return_array: :obj:`bool`, optional
        If True, return a boolean numpy array with the null hypothesis
        rejection status (i.e., each rejected null hypothesis array
        position has a ``True`` value.)

    Returns
    -------
    If ``return_array`` is False:
        :obj:`int`
            Number of discoveries (statistically significant) tests with
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


def benjamini_hochberg(pvalues: np.ndarray,
                       threshold: float = 0.05,
                       return_array: bool = False) -> t.Union[int, np.ndarray]:
    """Uses the Benjamini-Hochberg correction to hypothesis testing.

    This is an False Discovery Rate (FDR) controlling algorithm, and
    works only if the test statistics related to the given p-values are
    independent. If this is not the case, use the ``Benjamini-Yekutieli``
    corrected algorithm.

    Arguments
    ---------
    pvalues : :obj:`np.ndarray` shape = (test_num,), dtype = :obj:`float`
        Numpy array with of p-values extracted from the statistical
        tests.

    threshold : :obj:`float`, optional
        Threshold, also known as ``significance level`` or ``Type I error
        rate``. The value used as threshold $t$ to reject the null hypothesis
        of each p-value $p_{i}$ iff $p_{i} < t$.

    return_array: :obj:`bool`, optional
        If True, return a boolean numpy array with the null hypothesis
        rejection status (i.e., each rejected null hypothesis array
        position has a ``True`` value.)

    Returns
    -------
    If ``return_array`` is False:
        :obj:`int`
            Number of discoveries (statistically significant) tests with
            the FDR rate controlled by Benjamini-Hochberg correction.
    Else:
        :obj:`np.ndarray` shape = (test_num,), dtype = :obj:`bool`
    """
    if not 0 <= threshold <= 1:
        raise ValueError(
            "'threshold' must be in [0, 1] range (got {}.)".format(threshold))

    sorted_ind = np.argsort(pvalues)
    pvalues = pvalues[sorted_ind]

    tests = np.zeros(pvalues.size, dtype=bool)

    try:
        max_ind = np.nonzero(
            pvalues <= threshold * np.arange(1, 1 + pvalues.size) /
            pvalues.size)[0][-1]

    except IndexError:
        return tests

    tests[sorted_ind[:(1 + max_ind)]] = True

    if return_array:
        return tests

    return np.sum(tests)


def benjamini_yekutieli(pvalues: np.ndarray,
                        threshold: float = 0.05,
                        return_array: bool = False
                        ) -> t.Union[int, np.ndarray]:
    """Uses the Benjamini-Yekutieli correction to hypothesis testing.

    This is an False Discovery Rate (FDR) controlling algorithm. This
    Benjamini-Yekutieli corrected version, unlike the Benjamini-Hochberg
    version, always controls de FDR at level of the ``threshold``,
    independently of independence of test statistics related to the given
    p-values.

    Arguments
    ---------
    pvalues : :obj:`np.ndarray` shape = (test_num,), dtype = :obj:`float`
        Numpy array with of p-values extracted from the statistical
        tests.

    threshold : :obj:`float`, optional
        Threshold, also known as ``significance level`` or ``Type I error
        rate``. The value used as threshold $t$ to reject the null hypothesis
        of each p-value $p_{i}$ iff $p_{i} < t$.

    return_array: :obj:`bool`, optional
        If True, return a boolean numpy array with the null hypothesis
        rejection status (i.e., each rejected null hypothesis array
        position has a ``True`` value.)

    Returns
    -------
    If ``return_array`` is False:
        :obj:`int`
            Number of discoveries (statistically significant) tests with
            the FDR rate controlled by Benjamini-Yekutieli correction.
    Else:
        :obj:`np.ndarray` shape = (test_num,), dtype = :obj:`bool`
    """
    if not 0 <= threshold <= 1:
        raise ValueError(
            "'threshold' must be in [0, 1] range (got {}.)".format(threshold))

    def _harmonic_number(n: int) -> float:
        r"""Calculate an approximation of the ``n``th harmonic number $H_{n}$.

        $H_{n}$ is calculated as $H_{n} = \sum_{i=1}^{n}\frac{1}{i}$, but
        can be approximated with
        $$$
        H_{n} \approx \log(n)
                      + \gamma
                      + \frac{1}{2n}
                      - \frac{1}{12n^{2}}
                      + \frac{1}{120n^{4}}
        $$$
        Where $\gamma$ is the Euler-Mascheroni constant, also known as
        ``Euler constant.``
        """
        harm_num = (np.euler_gamma + np.log(n) + 0.5 / n - 1 / (12 * n**2) +
                    1 / (120 * n**4))
        return harm_num

    mod_threshold = threshold / _harmonic_number(n=pvalues.size)

    return benjamini_hochberg(
        pvalues=pvalues, threshold=mod_threshold, return_array=return_array)


def fdr_estimation(pvalues: np.ndarray,
                   threshold: float = 0.05,
                   lambda_: float = 0.5) -> float:
    """Estimate the False Discovery Rate (FDR) of ``pvalues`` at ``threshold`` level.

    Arguments
    ---------
    pvalues : :obj:`np.ndarray`
        Numpy array with of p-values extracted from the statistical
        tests.

    threshold : :obj:`float`, optional
        Threshold, also known as ``significance level`` or ``Type I error
        rate``. The value used as threshold $t$ to reject the null hypothesis
        of each p-value $p_{i}$ iff $p_{i} < t$.

    lambda_ : :obj:`float`, optional

    Returns
    -------
    :obj:`float`
        False Discovery Rate (FDR) estimation in function of ``threshold``
        and ``lambda_``.
    """
    if not 0 <= threshold <= 1:
        raise ValueError(
            "'threshold' must be in [0, 1] range (got {}.)".format(threshold))

    if not 0 <= lambda_ <= 1:
        raise ValueError(
            "'lambda_' must be in [0, 1] range (got {}.)".format(lambda_))

    if lambda_ == 1.0:
        return 0.0

    disc_num = max(1, np.sum(pvalues <= threshold))

    return (threshold / (1 - lambda_)) * (np.sum(pvalues > lambda_) / disc_num)


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
        print(
            "\rBonferroni test {}%...".format(100 * (1 + i) / num_tests),
            end="")

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
        print("\rHolm test {}%...".format(100 * (1 + i) / num_tests), end="")

    print("\rHolm test finished successfully.")


def _test_bh() -> None:
    import statsmodels.stats.multitest
    np.random.seed(16)
    num_tests = 500

    for i in np.arange(num_tests):
        pvals = np.random.exponential(scale=1 / 64, size=1000)
        res_a = statsmodels.stats.multitest.multipletests(
            pvals, alpha=0.05, method="fdr_bh")[0]
        res_b = benjamini_hochberg(pvals, threshold=0.05, return_array=True)
        assert np.allclose(res_a, res_b)
        print(
            "\rBenjamini-Hochberg test {}%...".format(
                100 * (1 + i) / num_tests),
            end="")

    print("\rBenjamini-Hochberg test finished successfully.")


def _test_by() -> None:
    import statsmodels.stats.multitest
    np.random.seed(16)
    num_tests = 500

    for i in np.arange(num_tests):
        pvals = np.random.exponential(scale=1 / 64, size=1000)
        res_a = statsmodels.stats.multitest.multipletests(
            pvals, alpha=0.05, method="fdr_by")[0]
        res_b = benjamini_yekutieli(pvals, threshold=0.05, return_array=True)
        assert np.allclose(res_a, res_b)
        print(
            "\rBenjamini-Yekutieli test {}%...".format(
                100 * (1 + i) / num_tests),
            end="")

    print("\rBenjamini-Yekutieli test finished successfully.")


def _test_fdr() -> None:
    import matplotlib.pyplot as plt
    np.random.seed(16)

    threshold = 0.05

    num_null = 800
    num_alt = 200
    pvals = np.concatenate((
        np.random.exponential(scale=1 / 32, size=num_alt),
        np.random.random(size=num_null),
    ))

    lambda_vals = np.linspace(0, 1, 100)
    vals = np.array([
        fdr_estimation(pvalues=pvals, threshold=threshold, lambda_=l)
        for l in lambda_vals
    ])

    plt.subplot(121)
    plt.hist(pvals, bins=64)
    plt.title("P-values histogram")

    plt.subplot(122)
    plt.plot(lambda_vals, vals, linestyle="-")
    plt.title("Significance level: {}".format(threshold))
    plt.ylabel("False Discovery Rate (FDR)")
    plt.xlabel("Lambda parameter")
    plt.show()


if __name__ == "__main__":
    _test_bonf()
    _test_holm()
    _test_bh()
    _test_by()
    _test_fdr()
