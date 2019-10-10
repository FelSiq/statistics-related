"""Implementation for Hamiltonian Monte Carlo (HMC) sampling."""
# pylint: disable=E1101
import typing as t

import autograd
import autograd.numpy as np


def hmc(initial_theta: np.ndarray,
        log_target: t.Callable[[np.ndarray], float],
        num_samples: int = 200,
        num_leapfrog_steps: int = 50,
        step_size: float = 0.1,
        burnout_frac: float = 0.5,
        random_state: t.Optional[int] = None,
        verbose: bool = False) -> np.ndarray:
    """Hamiltonian Monte Carlo implementation.

    Arguments
    ---------
    initial_theta : :obj:`np.ndarray`
        Initial values for the Markov Chain.

    log_target : :obj:`t.Callable`
        Log-density target function. Must receive a :obj:`np.ndarray`
        as the first argument, and return the natural (base-e) logarithm
        of the probability density evaluated at the given point as the
        argument.

    num_samples : :obj:`int`, optional
        Number of instances to sample.

    num_leapfrog_steps : :obj:`int`, optional
        Number of ``leapfrog`` steps for every sample.

    step_size : :obj:`float`, optional
        Also known as ``epsilon`` parameter. Affects the step size of
        each ``leapfrog`` step.

    burnout_frac : :obj:`float`, optional
        Fraction of initial samples to ignore. Must be a :obj:`float`
        value in [0, 1) range. Used to minimize the influence of the
        ``initial_theta`` value on the samples. More precisely, the
        np.ceil(num_samples * burnout_frac) first instances will be
        ignored before returning the samples.

    random_state : :obj:`int`, optional
        If not None, set numpy random seed before the first HMC
        sampling.

    verbose : :obj:`bool`, optional
        Enable priting of additional information about the HMC process.

    Returns
    -------
    :obj:`np.ndarray`
        HMC samples.

    Notes
    -----
    Works only for unimodal targets. Need to implement parallel
    tempering for multimodal targets.
    """

    def calc_total_energy(momentum: np.ndarray,
                          theta_logtarget: float) -> float:
        """Calculates the total energy of the current position on HMC process."""
        energy_kinetic = 0.5 * np.dot(momentum, momentum)
        energy_potential = -theta_logtarget
        energy_total = energy_kinetic + energy_potential
        return energy_total

    if not 0 <= burnout_frac < 1:
        raise ValueError("'burnout_frac' must be in [0, 1) interval.")

    if not isinstance(initial_theta, np.ndarray):
        initial_theta = np.array(initial_theta)

    if random_state is not None:
        np.random.seed(random_state)

    thetas = np.zeros((num_samples, initial_theta.size))
    grad_fun = autograd.grad(log_target)

    theta_cur = np.copy(initial_theta)
    theta_cur_grad = grad_fun(theta_cur)
    theta_logtarget_cur = log_target(theta_cur)
    hits = 0

    for sample_id in np.arange(num_samples):
        momentum = np.random.randn(theta_cur.size)
        total_energy_initial = calc_total_energy(
            momentum=momentum, theta_logtarget=theta_logtarget_cur)

        theta_new = theta_cur
        grad_new = theta_cur_grad
        for _ in np.arange(num_leapfrog_steps):
            momentum += step_size * 0.5 * grad_new
            theta_new += step_size * momentum
            grad_new = grad_fun(theta_new)
            momentum += step_size * 0.5 * grad_new

        theta_logtarget_new = log_target(theta_new)
        total_energy_final = calc_total_energy(
            momentum=momentum, theta_logtarget=theta_logtarget_new)

        delta_energy = total_energy_final - total_energy_initial
        if np.log(np.random.uniform(0, 1)) < delta_energy:
            theta_cur = theta_new
            theta_cur_grad = grad_new
            theta_logtarget_cur = theta_logtarget_new
            hits += 1

        thetas[sample_id, :] = theta_cur

    if verbose:
        acc_rate = hits / num_samples
        print("Acceptance ratio: {:.4f}".format(acc_rate))

    burnout_ind = int(np.ceil(num_samples * burnout_frac))
    return thetas[burnout_ind:]


def _test():
    import matplotlib.pyplot as plt
    random_state = 16

    def log_sphere(theta):
        return -20 * np.square(np.linalg.norm(theta, ord=2) - 10)

    thetas = hmc(
        initial_theta=[3.0, 0.0],
        log_target=log_sphere,
        random_state=random_state,
        step_size=0.2,
        verbose=True)

    print("Sampled {} instances.".format(thetas.shape[0]))
    plt.plot(thetas[:, 0], thetas[:, 1])
    plt.show()


if __name__ == "__main__":
    _test()
