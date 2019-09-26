"""Implementation of Metropolis-Hasting algorithm."""
import typing as t

import numpy as np


def symm_parallel_metropolis_hasting(
        initial_theta: t.Union[float, np.ndarray],
        num_samples: int,
        log_target: t.Callable[
            [t.Union[float, np.ndarray], t.Union[float, np.ndarray]], float],
        proposal_sampler: t.
        Callable[[t.Union[float, np.ndarray], t.Union[float, np.ndarray]], t.
                 Union[float, np.ndarray]],
        betas: np.ndarray,
        discard_warm_up: bool = True,
        warm_up_frac: float = 0.5,
        verbose: bool = False,
        return_acceptance_rate: bool = False,
        random_state: t.Optional[int] = None) -> np.ndarray:
    """Symmetric case of Metropolis-Hasting algorithm."""
    if num_samples <= 0:
        raise ValueError("'num_samples' must be a positive value.")

    if discard_warm_up and not 0 <= warm_up_frac < 1:
        raise ValueError("'warm_up_frac' must be in [0.0, 1.0) range.")

    if random_state is not None:
        np.random.seed(random_state)

    theta = np.array([np.copy(initial_theta) for _ in np.arange(betas.size)],
                     dtype=np.float)

    theta_log_targ = np.array([
        log_target(cur_theta, cur_beta)
        for cur_theta, cur_beta in zip(theta, betas)
    ], dtype=np.float)

    if isinstance(initial_theta, (int, float, np.number)):
        thetas = np.zeros((num_samples, betas.size), dtype=np.float)

    else:
        thetas = np.zeros((num_samples, betas.size, initial_theta.size),
                          dtype=np.float)

    hits = np.zeros(betas.size)
    swaps = np.zeros(betas.size - 1)
    swaps_hits = np.zeros(betas.size - 1)

    for ind_inst in np.arange(num_samples):
        for ind_beta, cur_beta in enumerate(betas):
            theta_proposed = proposal_sampler(theta[ind_beta], cur_beta)
            log_theta_prop = log_target(theta_proposed, cur_beta)

            if np.log(np.random.uniform(
                    0, 1)) < log_theta_prop - theta_log_targ[ind_beta]:
                theta[ind_beta] = theta_proposed
                theta_log_targ[ind_beta] = log_theta_prop
                hits[ind_beta] += 1

        swap_ind = np.random.randint(betas.size - 1)
        swaps[swap_ind] += 1

        aux = log_target(theta[swap_ind], betas[swap_ind + 1]) + log_target(
            theta[swap_ind + 1], betas[swap_ind])
        aux -= theta_log_targ[swap_ind] + theta_log_targ[swap_ind + 1]

        if np.log(np.random.uniform(0, 1)) < aux:
            theta[swap_ind], theta[swap_ind + 1] = theta[swap_ind +
                                                         1], theta[swap_ind]
            theta_log_targ[swap_ind] = log_target(theta[swap_ind],
                                                  betas[swap_ind])
            theta_log_targ[swap_ind + 1] = log_target(theta[swap_ind + 1],
                                                      betas[swap_ind + 1])
            swaps_hits[swap_ind] += 1

        thetas[ind_inst] = theta

    acceptance_rate = hits / num_samples
    swap_acceptance_rate = np.zeros(swaps_hits.size)
    swap_acceptance_rate[swaps > 0] = swaps_hits / swaps

    if verbose:
        print("Acceptance rate: {}".format(acceptance_rate))
        print("Theoretically expected: [0.23, 0.50] - ",
              np.logical_and(acceptance_rate >= 0.23, acceptance_rate <= 0.50))
        print("Swap (chains j and j+1) acceptance rate: {}".format(
            swap_acceptance_rate))

    if discard_warm_up:
        ret_thetas = thetas[int(warm_up_frac * num_samples):]

    else:
        ret_thetas = thetas

    if return_acceptance_rate:
        return ret_thetas, acceptance_rate

    return ret_thetas


def metropolis_hasting(
        initial_theta: t.Union[float, np.ndarray],
        num_samples: int,
        log_target: t.Callable[[t.Union[float, np.ndarray]], float],
        proposal_sampler: t.Callable[[t.Union[float, np.ndarray]], t.
                                     Union[float, np.ndarray]],
        proposal_log_density: t.Optional[t.Callable[
            [t.Union[float, np.ndarray], t.Union[float, np.
                                                 ndarray]], float]] = None,
        discard_warm_up: bool = True,
        warm_up_frac: float = 0.5,
        verbose: bool = False,
        return_acceptance_rate: bool = False,
        random_state: t.Optional[int] = None) -> np.ndarray:
    """Symmetric case of Metropolis-Hasting algorithm."""
    if num_samples <= 0:
        raise ValueError("'num_samples' must be a positive value.")

    if discard_warm_up and not 0 <= warm_up_frac < 1:
        raise ValueError("'warm_up_frac' must be in [0.0, 1.0) range.")

    if random_state is not None:
        np.random.seed(random_state)

    theta = initial_theta
    theta_log_targ = log_target(theta)

    if isinstance(initial_theta, (int, float, np.number)):
        thetas = np.zeros(num_samples)

    else:
        thetas = np.zeros((num_samples, initial_theta.size))

    hits = 0

    for ind in np.arange(num_samples):
        theta_proposed = proposal_sampler(theta)
        log_theta_prop = log_target(theta_proposed)

        q_term = 0.0
        if proposal_log_density is not None:
            q_term = (proposal_log_density(theta, theta_proposed) -
                      proposal_log_density(theta_proposed, theta))

        if np.log(np.random.uniform(
                0, 1)) < log_theta_prop - theta_log_targ + q_term:
            theta = theta_proposed
            theta_log_targ = log_theta_prop
            hits += 1

        thetas[ind] = theta

    acceptance_rate = hits / num_samples

    if verbose:
        print("Acceptance rate: {}".format(acceptance_rate))
        print("Theoretically expected: [0.23, 0.50] (results is {}.)".format(
            "optimal" if 0.23 <= acceptance_rate <= 0.50 else "not optimal"))

    if discard_warm_up:
        ret_thetas = thetas[int(warm_up_frac * num_samples):]

    else:
        ret_thetas = thetas

    if return_acceptance_rate:
        return ret_thetas, acceptance_rate

    return ret_thetas


def symm_metropolis_hasting(
        initial_theta: float,
        num_samples: int,
        log_target: t.Callable[[float], float],
        proposal_sampler: t.Callable[[float], float],
        discard_warm_up: bool = True,
        warm_up_frac: float = 0.5,
        verbose: bool = False,
        return_acceptance_rate: bool = False,
        random_state: t.Optional[int] = None) -> np.ndarray:
    """Symmetric case of Metropolis-Hasting algorithm."""
    ret = metropolis_hasting(
        initial_theta=initial_theta,
        num_samples=num_samples,
        log_target=log_target,
        proposal_sampler=proposal_sampler,
        proposal_log_density=None,
        discard_warm_up=discard_warm_up,
        warm_up_frac=warm_up_frac,
        verbose=verbose,
        return_acceptance_rate=return_acceptance_rate,
        random_state=random_state)

    return ret


def _experiment_01() -> None:
    """Experiment 01."""
    import matplotlib.pyplot as plt
    import scipy.stats

    random_seed = 16
    np.random.seed(random_seed)

    laplace_dist = scipy.stats.laplace(loc=0.0, scale=1.0 / (2**0.5))
    test_vals = np.linspace(-5, 5, 500)

    for plot_id, scale in enumerate([0.1, 2.5, 10, 50]):
        thetas = symm_metropolis_hasting(
            initial_theta=0.0,
            num_samples=10000,
            log_target=lambda x: -np.abs(x),
            proposal_sampler=
            lambda theta: theta + np.random.normal(loc=0.0, scale=scale),
            discard_warm_up=True,
            warm_up_frac=0.5,
            verbose=True,
            random_state=random_seed)

        plt.subplot(4, 2, plot_id * 2 + 1)
        plt.plot(thetas[::10], label=str(scale))
        plt.legend()

        plt.subplot(4, 2, plot_id * 2 + 2)
        plt.plot(test_vals, laplace_dist.pdf(test_vals))
        plt.hist(thetas[::10], bins=128, density=True, label=str(scale))
        plt.legend()

    plt.show()


def _experiment_02() -> None:
    """Experiment 02."""
    import matplotlib.pyplot as plt

    def bimodal_dist(theta, gamma):
        return np.exp(-gamma * np.square(np.square(theta) - 1))

    def log_bimodal_dist(theta, gamma):
        return -gamma * np.square(np.square(theta) - 1)

    random_seed = 16
    np.random.seed(random_seed)

    betas = np.logspace(-3, 0, 5)
    print("Betas:", betas)

    gamma = 64

    samples = symm_parallel_metropolis_hasting(
        initial_theta=1,
        num_samples=10000,
        log_target=lambda x, beta: beta * log_bimodal_dist(x, gamma),
        proposal_sampler=
        lambda x, beta: x + 0.1 / np.sqrt(beta) * np.random.randn(),
        betas=betas,
        verbose=True,
        random_state=16)

    test_vals = np.linspace(-3, 3, 100)
    plt.hist(samples[:, -1], bins=64, density=True, label='MCMC MH samples')
    plt.plot(
        test_vals,
        bimodal_dist(test_vals, gamma),
        label='(Unnormalized) target')
    plt.legend()
    plt.show()


def _experiment_03():
    """3rd experiment."""
    import matplotlib.pyplot as plt

    def generic_target(x, mu_vec):
        aux_1 = x - mu_vec
        aux_2 = np.arange(1, 1 + mu_vec.shape[0])
        return np.sum(np.exp(-aux_2 / 3 * np.sum(aux_1 * aux_1, axis=1)))

    def generic_log_target(x, mu_vec):
        aux_1 = x - mu_vec
        aux_2 = np.arange(1, 1 + mu_vec.shape[0])
        return np.log(
            np.sum(np.exp(-aux_2 / 3 * np.sum(aux_1 * aux_1, axis=1))))

    target = lambda x: generic_target(x, mu_vec)
    log_target = lambda x: generic_log_target(x, mu_vec)

    betas = np.logspace(-2, 0, 5)
    print("Betas:", betas)

    R = 5
    mu_vec = np.array([
        [0, 0],
        [R, R],
        [-R, R],
    ])

    samples = symm_parallel_metropolis_hasting(
        initial_theta=np.array([0, 3]),
        num_samples=125000,
        log_target=lambda x, beta: beta * log_target(x),
        proposal_sampler=
        lambda x, beta: x + 1.5 / np.sqrt(beta) * np.random.randn(x.size),
        betas=betas,
        verbose=True,
        random_state=16)

    vals = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(vals, vals)
    Z_1 = np.zeros(X.shape)
    Z_2 = np.zeros(X.shape)
    for i in np.arange(vals.size):
        for j in np.arange(vals.size):
            aux = target([X[i, j], Y[i, j]])
            Z_1[i, j] = aux
            Z_2[i, j] = np.log(aux)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.contour(X, Y, Z_1)
    plt.plot(samples[:, -1, 0], samples[:, -1, 1], '.')
    plt.title('Target')
    plt.subplot(1, 2, 2)
    plt.contour(X, Y, Z_2)
    plt.title('Log-target')
    plt.plot(samples[:, -1, 0], samples[:, -1, 1], '.')
    plt.show()

    print("E_{0.01}[Theta_1]:", np.mean(samples[:, 0, 0]))
    print("E_{0.01}[Theta_2]:", np.mean(samples[:, 0, 1]))
    print("E_{1}[Theta_1]:", np.mean(samples[:, -1, 0]))
    print("E_{1}[Theta_2]:", np.mean(samples[:, -1, 1]))


if __name__ == "__main__":
    _experiment_01()
    _experiment_02()
    _experiment_03()
