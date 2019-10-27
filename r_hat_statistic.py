"""R-hat and split R-hat statistic for MCMC convergence checking.

Usually, the best practice is to use always split R-hat instead
of traditional R-hat, as the later one can miss some cases where
the chains are non-stationary.
"""
import numpy as np


def r_hat_stat(chains: np.ndarray, verbose: bool = False) -> float:
    r"""Calculate R-hat statistic from $m$ different MCMC chains.

    The R-hat statistics is elaborated to verify if the MCMC
    converged using $m$ different MCMC chains starting from
    widely dispersed (distinct) initial states.

    R-hat approaches 1 as the number of samples in every MCMC chain
    increases.

    In general, the threshold 1.1 is commonly used to consider a
    chain having converged.

    Arguments
    ---------
    chains : :obj:`np.ndarray`, shape: (num_samples, num_chains)
        Samples for every MCMC chain. Every column represents a
        distinct and independent MCMC chain. Note that this function
        is only defined for scalar MCMC samples.

    verbose : :obj:`bool`, optional
        Enable message printing about the convergence status.

    Returns
    -------
    :obj:`float`
        R-hat statistic.

    Notes
    -----
    ``The similarity of the results is commonly measured by the potential
    scale reduction R-hat, that measures the factor by which the scale of
    the current distribution might be reduced if the simulations were
    continued until $n \rightarrow \infty$ (Gelman et al. (2013))``
    """
    if not isinstance(chains, np.ndarray):
        chains = np.array(chains)

    chains_num, _ = chains.shape
    mean_axis = chains.mean(axis=0)
    var_axis = chains.var(axis=0, ddof=1)

    between_chain_var_mod = mean_axis.var(ddof=1)
    within_chain_var = var_axis.mean()

    r_hat = np.sqrt((chains_num - 1) / chains_num +
                    between_chain_var_mod / within_chain_var)

    if verbose:
        print("Samples per chain: {} - Number of chains: {}".format(*chains.shape))
        print("R-hat statistic: {}".format(r_hat))
        print("Theoretically expected: 1.1 (chains may "
              "have {}.)".format(
                  "converged" if r_hat <= 1.1 else "not converged"))

    return r_hat


def split_r_hat_stat(chains: np.ndarray, verbose: bool = False) -> float:
    """Split R-hat statistic.

    Similar to the R-hat statistic. However, in this case, every chain
    is separated in half and considering each part as an independent
    chain, doubling the number of chains considered.

    Returns
    -------
    :obj:`float`
        R-hat statistic for splitted ``chains``.
    """
    return r_hat_stat(np.hstack(np.split(chains, 2, axis=0)), verbose=verbose)
