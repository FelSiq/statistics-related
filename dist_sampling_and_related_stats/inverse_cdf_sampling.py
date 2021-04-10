"""Examples of how to samples from the inverse cdf function."""
import typing as t

import numpy as np


def sample_exp(lambda_: float,
               num_inst: int = 1,
               random_state: t.Optional[int] = None) -> np.ndarray:
    r"""Sample from the exponential distribution using the inverse cdf.

    The c.d.f. of exponential distribution is
    \[
    F(x; \lambda) =
        \begin{cases}
            1 - e^{-\lambda x} & \text{if } x \geq 0 \\
            0 & \text{if } x < 0
        \end{cases}
    \]
    Thus, its inverse is
    $$
    F^{-1}(x; \lambda) = -\frac{\log(1 - x)}{\lambda}
    $$
    if $x \sim \text{Uniform(0, 1)}$, then 1 - x and x
    has the same distribution and, therefore, we can
    replace 1 - x for x, which yields:
    $$
    F^{-1}(x; \lambda) = -\frac{\log(x)}{\lambda}
    $$

    Drawing random samples $x \sim \text{Uniform(0, 1)}$,
    we can use any version of the $F^{-1}$ to draw samples
    from the exponential function.
    """
    if lambda_ <= 0:
        raise ValueError("'lambda_' must be a positive value.")

    if random_state is not None:
        np.random.seed(random_state)

    uniform_samples = np.random.uniform(size=num_inst)

    return -np.log(uniform_samples) / lambda_


def sample_laplace(loc: float = 0,
                   scale: float = 1,
                   num_inst: int = 1,
                   random_state: t.Optional[int] = None) -> np.ndarray:
    """Draw samples from laplace distribution."""
    if random_state is not None:
        np.random.seed(random_state)

    uniform_samples = np.random.uniform(size=num_inst)
    _aux = uniform_samples - 0.5

    return loc - scale * np.sign(_aux) * np.log(1 - 2 * np.abs(_aux))


def _test_exp():
    import matplotlib.pyplot as plt
    import scipy.stats

    vals = np.linspace(-1, 6, 100)

    samples = sample_exp(0.5, num_inst=1000)
    plt.subplot(1, 2, 1)
    plt.plot(vals, scipy.stats.expon(scale=1 / 0.5).pdf(vals))
    plt.hist(samples, bins=64, density=True)

    samples = sample_exp(3.0, num_inst=1000)
    plt.subplot(1, 2, 2)
    plt.plot(vals, scipy.stats.expon(scale=1 / 3.0).pdf(vals))
    plt.hist(samples, bins=64, density=True)

    plt.show()


def _test_laplace():
    import matplotlib.pyplot as plt
    import scipy.stats

    vals = np.linspace(-4, 4, 100)
    samples = sample_laplace(loc=0, scale=1, num_inst=1000, random_state=16)
    plt.subplot(1, 2, 1)
    plt.plot(vals, scipy.stats.laplace(loc=0, scale=1).pdf(vals))
    plt.hist(samples, bins=64, density=True)

    vals = np.linspace(-10, 20, 100)
    samples = sample_laplace(loc=6, scale=3, num_inst=1000, random_state=32)
    plt.subplot(1, 2, 2)
    plt.plot(vals, scipy.stats.laplace(loc=6, scale=3).pdf(vals))
    plt.hist(samples, bins=64, density=True)

    plt.show()


if __name__ == "__main__":
    _test_exp()
    _test_laplace()
