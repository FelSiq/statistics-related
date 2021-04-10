"""Test a uniform permutation algorithm against a non-uniform one."""
import typing as t

import matplotlib.pyplot as plt
import numpy as np


def uniform_permutation(num_elements: int,
                        random_state: t.Optional[int] = None) -> np.ndarray:
    r"""Uniformly distributed permutation algorithm.

    A permutation algorithm being uniformily distributed means that every
    permutation of ``n`` elements has exactly $\frac{1}{n!}$ probability to
    happen.
    """
    if random_state is not None:
        np.random.seed(random_state)

    vals = np.arange(num_elements)

    for i in np.arange(num_elements):
        j = np.random.randint(i, num_elements)
        vals[i], vals[j] = vals[j], vals[i]

    return vals


def non_uniform_permutation(
        num_elements: int, random_state: t.Optional[int] = None) -> np.ndarray:
    """Produces an non-uniform permutation."""
    if random_state is not None:
        np.random.seed(random_state)

    vals = np.arange(num_elements)

    for i in np.arange(num_elements):
        j = np.random.randint(num_elements)
        vals[i], vals[j] = vals[j], vals[i]

    return vals


def _uniformity_test(num_elements: int,
                     permutation_func: t.Callable[[int, int], t.Sequence[int]],
                     num_samples: int = int(1e+6),
                     random_state: int = 16) -> np.ndarray:
    """Count the frequencies of each permutation of ``num_elements`` distinct elements."""
    # This test quality is inversely proportional to num_elements!
    hash_counter = {}  # type: t.Dict[t.Tuple[int, ...], int]

    for i in np.arange(num_samples):
        config = tuple(
            permutation_func(num_elements, int(random_state * (1 + i))))

        if config not in hash_counter:
            hash_counter[config] = 0

        hash_counter[config] += 1

    print(hash_counter)

    return np.array(list(hash_counter.values()), dtype=int)


def _test():
    """Compare a non-uniform permutation algorithm against a uniform one."""

    def fac(n: int):
        fac = 1
        while n:
            fac *= n
            n -= 1

        return fac

    num_elements = 3
    n_fac = fac(num_elements)

    plt.subplot(1, 2, 1)
    vals = _uniformity_test(
        num_elements=num_elements, permutation_func=non_uniform_permutation)
    plt.plot(vals / vals.sum(), label="Relative frequency")
    plt.hlines(y=1 / n_fac, xmin=0, xmax=n_fac - 1, linestyle='--')
    plt.ylim(0, 1)
    plt.title("Non-uniform permutation")
    plt.legend()

    plt.subplot(1, 2, 2)
    vals = _uniformity_test(
        num_elements=num_elements, permutation_func=uniform_permutation)
    plt.plot(vals / vals.sum(), label="Relative frequency")
    plt.hlines(y=1 / n_fac, xmin=0, xmax=n_fac - 1, linestyle='--')
    plt.ylim(0, 1)
    plt.title("Uniform permutation")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    _test()
