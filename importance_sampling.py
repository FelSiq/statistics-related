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

TypeValue = t.Union[float, np.ndarray]


def importance_sampling(fun_transf: t.Callable[[TypeValue], float],
                        fun_difficult: t.Callable[[TypeValue], float],
                        fun_surrogate: t.Callable[[TypeValue], float],
                        proposal_sampler: t.Callable[[], TypeValue],
                        num_inst: int,
                        random_state: t.Optional[int] = None):
    r"""Calculates E[h(x)] = \frac{1}{n} * \sum_{i}^{n}(h(x_{i}) * \frac{f_{i}}{g_{i}})."""

    if random_state is not None:
        np.random.seed(random_state)

    vals = np.zeros(num_inst)

    for ind in np.arange(num_inst):
        sample = proposal_sampler()
        vals[ind] = fun_transf(sample) * fun_difficult(sample) / fun_surrogate(
            sample)

    return vals.mean()


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
        proposal_sampler=proposal_sampler,
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
        proposal_sampler=proposal_sampler,
        num_inst=100000,
        random_state=16)

    print("Second expectation:", ans)


if __name__ == "__main__":
    _experiment_01()
