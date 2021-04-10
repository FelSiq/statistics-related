"""ESS (Effective Sample Size) for MCMC sampling efficiency measurements."""
import typing as t

import numpy as np


def ess(samples: np.ndarray,
        mean: t.Union[float, int],
        var: t.Union[float, int],
        verbose: bool = False) -> float:
    r"""ESS (Effective sample size) of a MCMC chain.

    This statistics evaluate how many independent samples a given
    ``samples`` of size n is worth. If all samples are uncorrelated,
    then $ESS = n$, while highly correlated samples have very small
    $ESS \ll n$.

    Arguments
    ---------
    samples : :obj:`np.ndarray`, shape: (n,)
        Samples chain from a MCMC process.

    mean : :obj:`float` or :obj:`int`
        Estimate of the process mean.

    var : :obj:`float` or :obj:`int`
        Estimate of the process variance.

    verbose : :obj:`bool`, optional
        If True, enable message printing related to this process.

    Returns
    -------
    :obj:`float`
        ESS statistic for ``samples.``
    """
    lag_size = 1
    stop_condition = False
    autocorr = []  # type: t.List[float]

    if not isinstance(samples, np.ndarray):
        samples = np.array(samples)

    shifted_samples = samples - mean

    while not stop_condition:
        autocorr.append(
            np.mean([
                shifted_samples[i] * shifted_samples[i + lag_size]
                for i in np.arange(samples.size - lag_size)
            ]))

        if lag_size >= 3 and lag_size % 2 == 1:
            stop_condition = (
                autocorr[lag_size - 1] + autocorr[lag_size - 2]) < 0

        lag_size += 1

    ess_stat = samples.size / (1 + 2 / var * np.sum(autocorr[:-2]))

    if verbose:
        print("Chain size: {} - ESS: {}".format(samples.size, ess_stat))
        print("ESS / (chain size):", ess_stat / samples.size)
        print("Last lag size effectively used:", lag_size - 2)

    return ess_stat
