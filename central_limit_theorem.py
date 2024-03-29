"""Tests regarding the Central Limit Theorem (CLT).

The CLT states that the sum of $n$ set of i.i.d, samples
follow a normal distribution as $n$ goes to infinity.

More precisely, the distribution of the sum S of
n samples will converge to:

    S ~ N(n * mu, n * sigma^2)

as n -> +inf, where S = sum_{i}(x_i), and E[x_i] = mu
and Var[x_i] = sigma^2 for all i.

The corollary of this fact is that the mean value X'
will also follow a normal distribution:

    X' ~ N(mu, sigma^2 / n)

where X' = mean_{i}(x_i). This follows by the properties
of the definition of both Expectation and Variance:

    E[X'] = E[S / n] = E[S] / n = n * mu / n = mu
and:
    Var[X'] = Var[S / n] = Var[S] / n^2 = sigma^2 / n
"""
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np


def _test_01():
    """Varying sample size. Number of sample set is the control variable."""
    np.random.seed(16)

    random_var_num = 10000
    for samples_num in [10, 100, 1000]:

        samples = np.random.uniform(0, 1, size=(samples_num, random_var_num))
        means = samples.mean(axis=0)

        plt.hist(
            (means - means.mean()) / means.std(),
            bins=64,
            density=True,
            label=samples_num)
        plt.legend()

    vals = np.linspace(-5, 5, 100)
    plt.plot(
        vals, scipy.stats.norm(loc=0, scale=1).pdf(vals), '--', color="black")
    plt.show()


def _test_02():
    """Varying the distributions. Number of samples and sample sets kept constant."""
    np.random.seed(16)

    random_var_num = 10000
    samples_num = 500

    samplers = [
        lambda size: np.random.uniform(0, 5, size=size),
        lambda size: np.random.gamma(1, size=size),
        lambda size: np.random.poisson(5, size=size),
    ]

    noise = 5 * np.random.random(size=random_var_num)
    plt.hist((noise - noise.mean()) / noise.std(), density=True, label="noise")
    plt.legend()

    for sampler in samplers:
        samples = sampler((samples_num, random_var_num))
        means = samples.mean(axis=0)

        plt.hist(
            (means - means.mean()) / means.std(),
            bins=64,
            density=True,
            label=samples_num)
        plt.legend()

    vals = np.linspace(-5, 5, 100)
    plt.plot(
        vals, scipy.stats.norm(loc=0, scale=1).pdf(vals), '--', color="black")
    plt.show()


def _test_03():
    """Drawing random samples from pure random distributions."""
    np.random.seed(16)

    random_var_num = 5000
    samples_num = 200

    samplers = [
        lambda size: np.random.uniform(np.random.randint(100), np.random.randint(100, 201), size=size),
        lambda size: np.random.gamma(95 * np.random.random(), 95 * np.random.random(), size=size),
        lambda size: np.random.poisson(np.random.randint(75), size=size),
        lambda size: np.random.normal(loc=np.random.randint(-100, 101), scale=100 * np.random.random(), size=size),
        lambda size: np.random.laplace(loc=np.random.randint(-100, 101), scale=100 * np.random.random(), size=size),
    ]

    samples = np.array([
        samplers[np.random.randint(len(samplers))](size=1)
        for _ in np.arange(random_var_num * samples_num)
    ]).reshape((samples_num, random_var_num))

    means = samples.mean(axis=0)

    plt.hist(
        (means - means.mean()) / means.std(),
        bins=64,
        density=True,
        label=samples_num)
    plt.legend()

    vals = np.linspace(-5, 5, 100)
    plt.plot(
        vals, scipy.stats.norm(loc=0, scale=1).pdf(vals), '--', color="black")
    plt.show()


def _test_04():
    """Drawing random samples from mixture of random distributions."""
    np.random.seed(16)

    random_var_num = 5000
    samples_num = 200

    samplers = [
        lambda size: np.random.uniform(np.random.randint(100), np.random.randint(100, 201), size=size),
        lambda size: np.random.gamma(95 * np.random.random(), 95 * np.random.random(), size=size),
        lambda size: np.random.poisson(np.random.randint(75), size=size),
        lambda size: np.random.normal(loc=np.random.randint(-100, 101), scale=100 * np.random.random(), size=size),
        lambda size: np.random.laplace(loc=np.random.randint(-100, 101), scale=100 * np.random.random(), size=size),
    ]

    samples = np.array([
        samplers[np.random.randint(len(samplers))](size=1) +
        samplers[np.random.randint(len(samplers))](size=1)
        for _ in np.arange(random_var_num * samples_num)
    ]).reshape((samples_num, random_var_num))

    means = samples.mean(axis=0)

    plt.hist(
        (means - means.mean()) / means.std(),
        bins=64,
        density=True,
        label=samples_num)
    plt.legend()

    vals = np.linspace(-5, 5, 100)
    plt.plot(
        vals, scipy.stats.norm(loc=0, scale=1).pdf(vals), '--', color="black")
    plt.show()


if __name__ == "__main__":
    # _test_01()
    # _test_02()
    # _test_03()
    _test_04()
