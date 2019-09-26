"""Tests with kernel density estimation."""
import typing as t

import numpy as np


def kernel_gaussian(inst: np.ndarray) -> np.ndarray:
    """Gaussian kernel."""
    if inst.ndim == 1:
        inst = inst.reshape(-1, 1)

    _, dimension = inst.shape

    aux = np.sum(np.square(inst), axis=1)
    return np.exp(-0.5 * aux) * np.power(2.0 * np.pi, -0.5 * dimension)


def infinity_norm(
        inst: t.Union[np.ndarray, float]) -> t.Union[np.ndarray, float]:
    """Calculates the infinity norm of ``inst``.

    The infinity norm is defined as norm_{inf}(inst) = max_{i}(|x_{i}|), i.e.,
    the maximum absolute value in the vector ``inst``.

    For instance, if inst = [0, 4, -7, 2, -1], then norm_{inf}(inst) = 7.
    """
    return np.max(np.abs(inst), axis=1)


def kernel_uniform(inst: np.ndarray) -> np.ndarray:
    """Uniform kernel."""
    if inst.ndim == 1:
        inst = inst.reshape(-1, 1)

    _, dimension = inst.shape

    return 0.5**dimension * (infinity_norm(inst) < 1.0)


def kernel_epanechnikov(inst: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel."""
    if inst.ndim != 1:
        raise ValueError("'inst' vector must be one-dimensional!")

    return 0.75 * (1.0 - np.square(inst)) * (np.abs(inst) < 1.0)


def kernel_density_est(unknown_points: np.ndarray,
                       kernel: t.Callable[[np.ndarray], np.ndarray],
                       known_points: np.ndarray,
                       kernel_bandwidth: t.Union[int, float, np.number], *args,
                       **kwargs) -> np.ndarray:
    """Calculate the density estimation for ``unknown points`` using ``kernel``.

    Arguments
    ---------
    unknown_points : :obj:`np.ndarray`
        Numpy array with coordinates to evaluate the density function value.

    kernel : :obj:`Callable`
        Callable which implements the kernel function. Must receive a
        numpy array as the first argument, and outputs a another numpy
        array. Can receive additional arguments via *args and **kwargs.

    known_points : :obj:`np.ndarray`
        Set of points given, which partly represents the underlying unknown
        distribution. Used to estimate the density function.

    kernel_bandwidth: :obj:`np.ndarray`
        Length of the ``bumps`` of the kernel function around each
        known point. This parameter must be optimized in order to the
        output density function really fits the ``known_points``
        distribution.

    *args:
        Additional arguments to the ``kernel`` function.

    **kwargs:
        Additional arguments to the ``kernel`` function.

    Returns
    -------
    :obj:`np.ndarray`
        Array containing each ``unknown_points`` evaluated with the kernel
        density estimation technique.
    """
    if isinstance(unknown_points, (float, int, np.number)):
        unknown_points = np.array([unknown_points])

    if known_points.ndim == 1:
        num_inst, dimension = known_points.size, 1

    else:
        num_inst, dimension = known_points.shape

    return np.array([
        np.sum(
            kernel((inst - known_points) / kernel_bandwidth, *args, **kwargs))
        for inst in unknown_points
    ]) / (num_inst * kernel_bandwidth**dimension)


def _experiment_01() -> None:
    """Kernel Density estimation experiment 01."""
    import matplotlib.pyplot as plt

    linear_vals = np.linspace(-3, 3, 100)
    plt.plot(linear_vals, kernel_gaussian(linear_vals), label="Gaussian")
    plt.plot(linear_vals, kernel_uniform(linear_vals), label="Uniform")
    plt.plot(
        linear_vals, kernel_epanechnikov(linear_vals), label="Epanechnikov")
    plt.title("Kernels")
    plt.legend()
    plt.show()


def _experiment_02() -> None:
    """Kernel Density estimation experiment 02."""
    import matplotlib.pyplot as plt

    linear_vals = np.linspace(-3, 3, 100)
    dim_1, dim_2 = np.meshgrid(linear_vals, linear_vals)

    z_gauss = np.zeros((linear_vals.size, linear_vals.size))
    z_uniform = np.zeros((linear_vals.size, linear_vals.size))

    for ind in np.arange(linear_vals.size):
        aux_1 = dim_1[:, ind].reshape(-1, 1)
        aux_2 = dim_2[:, ind].reshape(-1, 1)
        vals = np.hstack((aux_1, aux_2))
        z_gauss[:, ind] = kernel_gaussian(vals)
        z_uniform[:, ind] = kernel_uniform(vals)

    plt.title("Kernels")
    plt.subplot(121)
    plt.contour(dim_1, dim_2, z_gauss)
    plt.subplot(122)
    plt.contour(dim_1, dim_2, z_uniform)
    plt.show()


def _experiment_03() -> None:
    """Estimate Kernel bandwidth."""
    import matplotlib.pyplot as plt
    from cross_validation import loo_cv

    def loss_function(inst):
        """Mean of log-probabilities (0 <= inst <= 1)."""
        return -np.mean(np.log(inst))

    random_state = 16
    chosen_kernel = kernel_gaussian

    np.random.seed(random_state)

    candidate_bandwidths = np.linspace(0.01, 1.0, 25)
    known_points = np.random.normal(size=512)

    logls = np.zeros(candidate_bandwidths.size)

    for subset_test, subset_train in loo_cv(known_points):
        for ind_bandwidth in np.arange(candidate_bandwidths.size):
            kernel_est = kernel_density_est(
                unknown_points=subset_test,
                known_points=subset_train,
                kernel=chosen_kernel,
                kernel_bandwidth=candidate_bandwidths[ind_bandwidth])

            logls[ind_bandwidth] += loss_function(kernel_est)

    bandwidth_opt = candidate_bandwidths[logls.argmin()]

    random_ind = np.random.randint(candidate_bandwidths.size)
    bandwidth_random = candidate_bandwidths[random_ind]

    print("Optimal bandwidth: {} (loss: {})".format(bandwidth_opt,
                                                    logls.min()))
    print("Random bandwidth: {} (loss: {})".format(bandwidth_random,
                                                   logls[random_ind]))

    linear_vals = np.linspace(-3, 3, 100)

    plt.hist(known_points, 32, density=True)
    plt.plot(
        linear_vals,
        kernel_density_est(
            unknown_points=linear_vals,
            known_points=known_points,
            kernel=chosen_kernel,
            kernel_bandwidth=bandwidth_opt))
    plt.plot(
        linear_vals,
        kernel_density_est(
            unknown_points=linear_vals,
            known_points=known_points,
            kernel=chosen_kernel,
            kernel_bandwidth=bandwidth_random))
    plt.show()


if __name__ == "__main__":
    _experiment_01()
    _experiment_02()
    _experiment_03()
