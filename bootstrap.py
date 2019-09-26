"""Testing bootstrap."""
import typing as t

import numpy as np


def bootstrap(population: np.ndarray,
              num_samples: int = 10,
              prop: float = 1.0,
              random_state: t.Optional[int] = None) -> np.ndarray:
    """Generator of bootstraps ``population``.

    Arguments
    ---------
    prop : :obj:`float`
        Proportion between the size of ``population`` and the sampled
        population. Must be in (0.0, 1.0] interval.

    n : :obj:`int`
        Number of pseudo-datasets generated.

    random_state : :obj:`int`, optional
        If given, set the random seed before the first iteration.

    Returns
    -------
    :obj:`np.ndarray`
        Sample of ``population`` constructed via bootstrap technique.
    """
    if not isinstance(prop, (np.number, float, int)):
        raise TypeError("'prop' must be numberic type (got {}).".format(
            type(prop)))

    if not 0.0 < prop <= 1.0:
        raise ValueError("'prop' must be a number in (0.0, 1.0] interval.")

    if random_state is not None:
        np.random.seed(random_state)

    if population.ndim == 1:
        _pop_size = population.size

    else:
        _pop_size = population.shape[0]

    bootstrap_size = int(_pop_size * prop)

    for _ in np.arange(num_samples):
        cur_inds = np.random.randint(_pop_size, size=bootstrap_size)
        yield population[cur_inds]


def bootstrap_test(pop: np.ndarray,
                   test_statistic: t.Callable[[np.ndarray], t.
                                              Union[int, float, np.number]],
                   random_state: t.Optional[int] = None,
                   alpha: float = 0.05,
                   num_samples: int = 100,
                   bootstrap_sample_size: float = 1.0) -> np.ndarray:
    """Get a generic confidence interval via bootstrap technique."""
    if random_state is not None:
        np.random.seed(random_state)

    t_stat_pseudo_pops = np.zeros(num_samples)

    bootstrapper = bootstrap(
        population=pop, num_samples=num_samples, prop=bootstrap_sample_size)

    for ind, pseudo_pop in enumerate(bootstrapper):
        t_stat_pseudo_pops[ind] = test_statistic(pseudo_pop)

    interval = 100 * np.array([0.5 * alpha, 1.0 - 0.5 * alpha])

    return np.percentile(t_stat_pseudo_pops, interval)


def _experiment_01(random_state: t.Optional[int] = 16,
                   verbose: bool = True) -> bool:
    """Bootstrap experiment.

    To calculate the two-sided confidence interval (1.0 - alpha), for some
    parameter theta, can be obtained calculating the internal

        [h_{alpha/2}, h_{1-alpha/2)}]

    Where h_{x} denotes the x quantile of the bootstrap estimates for the
    parameter theta.

    To be more clear, we can calculate the some statistic theta for every
    bootstrapped pseudo-dataset, and then collect the two percentiles
    p_1 = h_{alpha/2} and p_2 = h_{1-alpha/2)} to form the two sided
    (1.0 - alpha) confidence internal [p_1, p_2].

    This experiment exemplifies this building the two-sided confidence
    interval for the mean of some dataset.
    """
    reps = 1000
    alpha = 0.05

    if random_state is not None:
        np.random.seed(random_state)

    pop = np.random.normal(size=100)

    pop_true_mean = pop.mean()

    if verbose:
        print("Population mean: {}".format(pop_true_mean))

    bootstrapper = bootstrap(pop, num_samples=reps, random_state=random_state)

    means = np.zeros(reps)
    for ind, pseudo_pop in enumerate(bootstrapper):
        means[ind] = pseudo_pop.mean()

    percentiles = 100 * np.array([0.5 * alpha, 1.0 - 0.5 * alpha])
    it_min, it_max = np.percentile(means, percentiles)

    in_conf_interval = it_min <= pop_true_mean <= it_max

    if verbose:
        print(
            "Confidence interval (alpha = {}/{}% confidence interval): [{}, {}]"
            .format(alpha, 100 * (1.0 - alpha), it_min, it_max))
        print("True mean in confidence internal: {}".format(in_conf_interval))

    return in_conf_interval


def _experiment_02():
    """Bootstrap experiment 02."""
    test_statistics = [np.mean, np.median]
    for t_stat in test_statistics:
        pop = np.random.normal(size=1000)
        conf_interval = bootstrap_test(pop=pop, test_statistic=t_stat)
        print(conf_interval, t_stat(pop))

    def corrcoef(pop):
        return np.corrcoef(pop[:, 0].T, pop[:, 1].T)[0, 1]

    pop_a = np.random.normal(size=1000).reshape(-1, 1)
    pop_b = np.random.normal(size=1000).reshape(-1, 1)
    pop = np.hstack((pop_a, pop_b))
    conf_interval = bootstrap_test(pop=pop, test_statistic=corrcoef)
    print(conf_interval, corrcoef(pop))


if __name__ == "__main__":
    print("Experiment 01")
    _experiment_01()

    print("Experiment 02")
    _experiment_02()
