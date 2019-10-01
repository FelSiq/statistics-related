r"""Importance sampling for evaluating expectations.

The expectation generally is over a difficult density f(x),
but with a trick we can evaluate it using an easy-to-sample
surrogate function g(x). Suppose a transformation function h.
Then, the expected value of samples $x_{i} \sim f$ (drawn from
the difficult density function) transformed by h (emphasizing
that h can be the identity function h(x) = x) is:

$$
E_{f(x)}[h(x)] = \int{ h(x)f(x) }
               = \int{ h(x) \frac{f(x)}{g(x)} g(x) }
               = E_{g(x)}[h(x) \frac{f(x)}{g(x)}]
$$

i.e., we can evaluate the samples $x_{i}$ transformed by h
using only the ratio of the difficult density function by
the easy-to-sample density function:

$$
E_{g(x)}[h(x) \frac{f(x)}{g(x)}] \approx
    \frac{1}{n} \sum_{i=1}^{n}(h(x_i) \frac{f(x_i)}{g(x_i)})
$$
"""
import typing as t

import numpy as np


def log_importance_sampling(
        fun_log_difficult: t.Callable[[np.ndarray], float],
        fun_log_surrogate: t.Callable[[np.ndarray], float],
        surrogate_sampler: t.Callable[[], t.Union[float, np.ndarray]],
        num_inst: int,
        fun_transf: t.Optional[t.Callable[[np.ndarray], float]] = None,
        random_state: t.Optional[int] = None):
    r"""Calculates E[h(x)] = \frac{1}{n} * \sum_{i}^{n}(h(x_{i}) * \frac{f_{i}}{g_{i}}).

    This log version of the importance sampling is more numerically stable than
    the standard version, and therefore should be preferred.
    """
    if random_state is not None:
        np.random.seed(random_state)

    samples = np.array([surrogate_sampler() for _ in np.arange(num_inst)])

    transf_samples = samples if fun_transf is None else fun_transf(samples)

    vals = transf_samples * np.exp(
        fun_log_difficult(samples) - fun_log_surrogate(samples))

    return vals.mean()


def importance_sampling(
        fun_difficult: t.Callable[[np.ndarray], float],
        fun_surrogate: t.Callable[[np.ndarray], float],
        surrogate_sampler: t.Callable[[], t.Union[float, np.ndarray]],
        num_inst: int,
        fun_transf: t.Optional[t.Callable[[np.ndarray], float]] = None,
        random_state: t.Optional[int] = None):
    r"""Calculates E[h(x)] = \frac{1}{n} * \sum_{i}^{n}(h(x_{i}) * \frac{f_{i}}{g_{i}}).

    This version is more numerically unstable than the log version of importance
    sampling.
    """
    if random_state is not None:
        np.random.seed(random_state)

    samples = np.array([surrogate_sampler() for _ in np.arange(num_inst)])

    transf_samples = samples if fun_transf is None else fun_transf(samples)

    vals = transf_samples * fun_difficult(samples) / fun_surrogate(samples)

    return vals.mean()


"""
# Inneficient, non-vectorized implementation. Kept just for reference.
TypeValue = t.Union[float, np.ndarray]

def importance_sampling(fun_transf: t.Callable[[TypeValue], float],
                        fun_difficult: t.Callable[[TypeValue], float],
                        fun_surrogate: t.Callable[[TypeValue], float],
                        proposal_sampler: t.Callable[[], TypeValue],
                        num_inst: int,
                        random_state: t.Optional[int] = None):
    if random_state is not None:
        np.random.seed(random_state)

    vals = np.zeros(num_inst)

    for ind in np.arange(num_inst):
        sample = proposal_sampler()
        vals[ind] = fun_transf(sample) * fun_difficult(sample) / fun_surrogate(
            sample)

    return vals.mean()
"""


def _experiment_01():
    """Experiment 01.

    Calculating some expectations.
    """

    def normal_pdf(x, mu, var):
        return np.exp(-np.square(x - mu) / (2 * var)) / np.sqrt(
            2 * np.pi * var)

    def laplace_pdf(x, mu, b):
        return np.exp(-np.abs(x - mu) / b) / (2 * b)

    b = 1

    proposal_sampler = lambda: np.random.laplace(0, b)
    fun_transf = lambda x: np.square(x - 1)
    fun_difficult = lambda x: 0.5 * (normal_pdf(x, mu=-1, var=0.5) + normal_pdf(x, mu=1, var=0.5))
    fun_surrogate = lambda x: laplace_pdf(x, 0, b)

    ans = importance_sampling(
        fun_transf=fun_transf,
        fun_difficult=fun_difficult,
        fun_surrogate=fun_surrogate,
        surrogate_sampler=proposal_sampler,
        num_inst=100000,
        random_state=16)

    print("First expectation:", ans)

    proposal_sampler = lambda: np.random.laplace(0, b)
    fun_transf = lambda x: np.square(x - 1)
    fun_difficult = lambda x: 0.5 * (normal_pdf(x, mu=-1, var=0.1) + normal_pdf(x, mu=1, var=0.1))
    fun_surrogate = lambda x: laplace_pdf(x, 0, b)

    ans = importance_sampling(
        fun_transf=fun_transf,
        fun_difficult=fun_difficult,
        fun_surrogate=fun_surrogate,
        surrogate_sampler=proposal_sampler,
        num_inst=100000,
        random_state=16)

    print("Second expectation:", ans)


def _experiment_02():
    """Second experiment."""
    import scipy.stats

    fun_transf_1 = np.square
    fun_transf_2 = lambda x: np.power(x, 4)

    fun_log_diff = scipy.stats.norm(loc=0, scale=1).logpdf
    fun_log_surrogate = scipy.stats.laplace(loc=0, scale=1).logpdf

    surrogate_sampler = lambda: np.random.laplace(loc=0, scale=1)

    is_1 = log_importance_sampling(
        fun_transf=fun_transf_1,
        fun_log_difficult=fun_log_diff,
        fun_log_surrogate=fun_log_surrogate,
        surrogate_sampler=surrogate_sampler,
        num_inst=4096,
        random_state=16)

    random_sample = np.random.normal(loc=0, scale=1, size=4096)

    print("Importance sampling E[x**2]:", is_1, "Random sample E[x**2]:",
          np.power(random_sample, 2).mean())

    is_2 = log_importance_sampling(
        fun_transf=fun_transf_2,
        fun_log_difficult=fun_log_diff,
        fun_log_surrogate=fun_log_surrogate,
        surrogate_sampler=surrogate_sampler,
        num_inst=4096,
        random_state=16)

    print("Importance sampling E[x**4]:", is_2, "Random sample E[x**4]:",
          np.power(random_sample, 4).mean())


if __name__ == "__main__":
    _experiment_01()
    _experiment_02()
