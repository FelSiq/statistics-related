"""Testing bootstrap."""
import typing as t

import numpy as np


def bootstrap(
    samples: np.ndarray,
    num_resamples: int = 256,
    random_state: t.Optional[int] = None,
) -> np.ndarray:
    """Generator of bootstraps ``samples``.

    Arguments
    ---------
    samples : :obj:`np.ndarray`
        Population to draw samples from.

    num_resamples : :obj:`int`
        Number of pseudo-datasets generated.

    random_state : :obj:`int`, optional
        If given, set the random seed before the first iteration.

    Yields
    -------
    :obj:`np.ndarray`
        Sample of ``samples`` constructed via bootstrap technique.
    """
    if random_state is not None:
        np.random.seed(random_state)

    samples = np.asarray(samples)

    _sample_size = samples.shape[0]

    for _ in np.arange(num_resamples):
        cur_inds = np.random.randint(_sample_size, size=_sample_size)
        yield samples[cur_inds]


def ci_bootstrap_percentile(
    samples: np.ndarray,
    test_statistic: t.Callable[[np.ndarray], t.Union[int, float, np.number]],
    random_state: t.Optional[int] = None,
    alpha: float = 0.05,
    num_resamples: int = 1024,
) -> np.ndarray:
    """Get a generic confidence interval via bootstrap percentile method.

    Should not be used. Use empirical bootstrap instead.
    """
    if random_state is not None:
        np.random.seed(random_state)

    t_stat_bootstrap_samples = np.zeros(num_resamples)

    bootstrapper = bootstrap(samples=samples, num_resamples=num_resamples)

    for ind, bootstrap_samples in enumerate(bootstrapper):
        t_stat_bootstrap_samples[ind] = test_statistic(bootstrap_samples)

    interval = (0.5 * alpha, 1.0 - 0.5 * alpha)

    est_int = np.quantile(t_stat_bootstrap_samples, interval)

    return est_int


def ci_bootstrap_empirical(
    samples: np.ndarray,
    test_statistic: t.Callable[[np.ndarray], t.Union[int, float, np.number]],
    alpha: float = 0.05,
    num_resamples: int = 1024,
    random_state: t.Optional[int] = None,
) -> np.ndarray:
    """Get a generic confidence interval via empirical bootstrap method."""
    if random_state is not None:
        np.random.seed(random_state)

    bootstrapper = bootstrap(samples=samples, num_resamples=num_resamples)
    orig_stat = test_statistic(samples)
    diffs = np.full(num_resamples, fill_value=-orig_stat)

    for ind, bootstrap_samples in enumerate(bootstrapper):
        bootstrap_stat = test_statistic(bootstrap_samples)
        diffs[ind] += bootstrap_stat

    interval = (0.5 * alpha, 1.0 - 0.5 * alpha)
    upper_diff, lower_diff = np.quantile(diffs, interval)

    est_int = np.array([orig_stat - lower_diff, orig_stat - upper_diff])

    return est_int


def _test_parametric_bootstrap():
    """Parametric bootstrap using geometric distribution example."""
    import scipy.stats

    arr = np.array(
        [
            [1, 1, 2, 1, 5, 2, 6, 2, 1, 1, 2, 1],
            [1, 2, 3, 4, 1, 1, 2, 2, 1, 5, 1, 6],
            [2, 3, 3, 4, 1, 1, 2, 2, 1, 5, 3, 3],
            [4, 2, 4, 2, 1, 4, 1, 2, 4, 3, 1, 4],
            [4, 1, 1, 1, 3, 1, 2, 1, 1, 2, 2, 3],
            [3, 2, 2, 1, 3, 2, 1, 1, 2, 1, 7, 4],
            [3, 4, 6, 1, 5, 1, 2, 5, 6, 8, 1, 4],
        ]
    ).ravel()

    p = float(1.0 / np.mean(arr))
    dist = scipy.stats.geom(p)
    num_resamples = 3000
    diffs = np.full(num_resamples, fill_value=-p)

    for i in np.arange(num_resamples):
        resamples = dist.rvs(len(arr))
        p_bootstrap = float(1.0 / np.mean(resamples))
        diffs[i] += p_bootstrap

    halpha = 0.5 * 0.05
    upper_diff, lower_diff = np.quantile(diffs, (halpha, 1 - halpha))
    est_int = np.array([p - lower_diff, p - upper_diff])

    print(p, est_int)


def _test():
    test_statistics = {"mean": np.mean, "median": np.median}
    for name, t_stat in test_statistics.items():
        samples = np.random.normal(size=1024)
        conf_interval = ci_bootstrap_percentile(samples=samples, test_statistic=t_stat)
        print(name, "percentile", conf_interval, t_stat(samples))
        conf_interval = ci_bootstrap_empirical(samples=samples, test_statistic=t_stat)
        print(name, "empirical ", conf_interval, t_stat(samples))

    def corrcoef(samples):
        return np.corrcoef(samples[:, 0].T, samples[:, 1].T)[0, 1]

    samples_a = np.random.normal(size=1024).reshape(-1, 1)
    samples_b = np.random.normal(size=1024).reshape(-1, 1)
    samples = np.hstack((samples_a, samples_b))
    conf_interval = ci_bootstrap_percentile(samples=samples, test_statistic=corrcoef)
    print("corrcoef", "percentile", conf_interval, corrcoef(samples))
    conf_interval = ci_bootstrap_empirical(samples=samples, test_statistic=corrcoef)
    print("corrcoef", "empirical ", conf_interval, corrcoef(samples))


if __name__ == "__main__":
    _test()
    _test_parametric_bootstrap()
