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
    chains : :obj:`np.ndarray`, shape: (num_samples, num_chains, [sample dimensions])
        Samples for every MCMC chain. Every column represents a
        distinct and independent MCMC chain. This function is defined
        for both scalar (unidimensional) and multidimensional
        samples of MCMC. In the later case, the R-hat statistic
        is calculated for each component of the samples, and the
        final R-hat statistic is the maximum value calculated
        among every component.

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

    def _r_hat_stat(chains_comp: np.ndarray) -> float:
        """Calculate the R-hat statistic for a single component of a vector."""
        chains_num, _ = chains_comp.shape
        mean_axis = chains_comp.mean(axis=0)
        var_axis = chains_comp.var(axis=0, ddof=1)

        between_chain_var_mod = mean_axis.var(ddof=1)
        within_chain_var = var_axis.mean()

        r_hat = np.sqrt((chains_num - 1) / chains_num +
                        between_chain_var_mod / within_chain_var)

        return r_hat

    if chains.ndim == 3:
        r_hats = np.array([
            _r_hat_stat(chains[:, :, ind_comp])
            for ind_comp in np.arange(chains.ndim)
        ])

        if verbose:
            print("R-hat for each vector component:", r_hats)

        r_hat = np.max(r_hats)

    else:
        r_hat = _r_hat_stat(chains_comp=chains)

    if verbose:
        print(
            "Samples per chain: {} - Number of chains: {} - Component dimension: {}"
            .format(*(chains.shape if chains.ndim == 3 else
                      (*chains.shape, 1))))
        print("R-hat statistic: {} {}".format(
            r_hat, "(maximum over all R-hats)" if chains.ndim == 3 else ""))
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


def _test() -> None:
    np.random.seed(16)

    num_samples = 40

    dataset_1 = np.random.randint(10, size=(num_samples, 4))
    dataset_2 = np.array(
        [[i, i + 1, 3 * i + 9, i * 2] for i in np.arange(num_samples)])
    dataset_3 = np.random.randint(10, size=(num_samples, 4))

    print(r_hat_stat(dataset_1, verbose=True), end="\n\n")
    print(r_hat_stat(dataset_2, verbose=True), end="\n\n")
    print(r_hat_stat(dataset_3, verbose=True), end="\n\n")

    dataset = np.stack((dataset_1, dataset_2, dataset_3), axis=2)

    print("Stacked dataset shape:", dataset.shape)
    print(r_hat_stat(dataset, verbose=True))


if __name__ == "__main__":
    _test()
