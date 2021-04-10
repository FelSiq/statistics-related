import typing as t

import numpy as np
import scipy.stats


def horners(x: t.Union[int, float], coeffs: np.ndarray) -> t.Union[int, float]:
    """Evaluates a polynomial in ``x``.

    Operates with time complexity Theta(n) and space
    complexity Theta(1).
    """
    val = 0.0 if isinstance(x, float) else 0

    for c in coeffs:
        val = c + x * val

    return val


def normal_pdf(x: t.Union[int, float], mean: float = 0.0,
               var: float = 1.0) -> float:
    """Probability density of normal distribution."""
    return 1.0 / np.sqrt(2.0 * np.pi * var) * np.exp(-np.square(x - mean) /
                                                     (2.0 * var))


def rejection_sampling(func_a: t.Callable[[t.Union[int, float]], float],
                       func_b: t.Callable[[t.Union[int, float]], float],
                       func_b_sampler: t.Callable[[], float],
                       func_b_coeff: t.Union[float, int] = 1.0,
                       samples_num: int = 1000,
                       random_seed: t.Optional[int] = None) -> np.ndarray:
    """Uses rejection sampling method to sample from a complicated function ``func_a``."""
    if random_seed is not None:
        np.random.seed(random_seed)

    vals = np.zeros(samples_num)

    cur_ind = 0
    while cur_ind < samples_num:
        random_point = func_b_sampler()

        p = np.random.uniform(0, 1)

        if p * func_b_coeff * func_b(random_point) < func_a(random_point):
            vals[cur_ind] = random_point
            cur_ind += 1

    return vals


def _experiment_01():
    """Rejection sampling experiment 01."""

    _pdf_gen_1 = scipy.stats.norm(5, 1)
    _pdf_gen_2 = scipy.stats.norm(-2, 3)
    _pdf_gen_3 = scipy.stats.expon()

    def complex_fun(
            x: t.Union[int, float, np.ndarray]) -> t.Union[np.ndarray, float]:
        return (0.25 * _pdf_gen_1.pdf(x) + 0.35 * _pdf_gen_2.pdf(x) +
                0.4 * _pdf_gen_3.pdf(x))

    N = 5000
    INTERVAL = (-10, 10)
    B_COEFF = 10
    SEED = 1234

    vals = np.linspace(*INTERVAL, num=N)

    generator = scipy.stats.uniform(INTERVAL[0], np.ptp(INTERVAL))

    res1 = np.array([complex_fun(x) for x in vals])
    res2 = B_COEFF * generator.pdf(vals)

    samples = rejection_sampling(
        func_a=complex_fun,
        func_b=generator.pdf,
        func_b_sampler=lambda: np.random.uniform(*INTERVAL),
        func_b_coeff=B_COEFF,
        samples_num=N,
        random_seed=SEED)

    plt.plot(vals, res1)
    plt.plot(vals, res2)
    plt.hist(samples, bins=64, density=True)
    plt.show()


def _experiment_02() -> None:
    """Rejection sampling experiment 02."""

    def complex_fun(
            x: t.Union[int, float, np.ndarray]) -> t.Union[float, np.ndarray]:
        return (0.40 * normal_pdf(x, mean=-2.0, var=1.2) + 0.70 * normal_pdf(
            x, mean=2.0, var=0.8) - 0.10 * normal_pdf(x, mean=-2.0, var=0.35))

    INTERVAL = (-10, 10)
    N = 10000
    B_COEFF = 3
    SEED = 1444

    mean = 0
    var = 5

    vals = np.linspace(*INTERVAL, num=N)

    res1 = complex_fun(vals)  # type: np.ndarray
    res2 = B_COEFF * normal_pdf(vals, mean=mean, var=var)  # type: np.ndarray

    # The following plot must be always below the horizontal line y = 1.0.
    plt.plot(vals, res1 / res2)
    plt.hlines(1.0, *INTERVAL)
    plt.show()

    aux = rejection_sampling(
        func_a=complex_fun,
        func_b=lambda x: normal_pdf(x, mean=mean, var=var),
        func_b_sampler=lambda: mean + np.sqrt(var) * np.random.randn(),
        func_b_coeff=B_COEFF,
        samples_num=N,
        random_seed=SEED)

    plt.plot(vals, res1)
    plt.plot(vals, res2)
    plt.hist(aux, bins=64, density=True)
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    _experiment_01()
    _experiment_02()
