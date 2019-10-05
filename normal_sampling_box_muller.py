"""Samples from normal distribution using Box-Muller transformation.

The transformation is applied to the bivariate normal distribution.
"""
import typing as t

import numpy as np


def sample_normal(num_inst: int, loc: float = 0.0, scale: float = 1.0, random_state: t.Optional[int] = None) -> np.ndarray:
    """Sample ``num_inst`` instances from normal distribution.

    Arguments
    ---------
    num_inst : :obj:`int`
        Number of samples to output.

    loc : :obj:`float`
       Mean of the normal distribution. 

    scale : :obj:`float`
        Standard deviation (not the variance!) of the normal distribution.

    random_state : :obj:`int`, optional
        If not None, set numpy random seed before the first sampling.

    Returns
    -------
    :obj:`np.ndarray`
        Samples of the normal distribution with ``loc`` mean and ``scale``
        standard deviation.

    Notes
    -----
    Uses the Box-Muller bivariate transformation, which maps two samples
    from the Uniform Distribution U(0, 1) into two samples of the Normal
    Distribution N(0, 1).
    """
    if random_state is not None:
        np.random.seed(random_state)

    remove_extra_inst = False

    if num_inst % 2:
        num_inst += 1
        remove_extra_inst = True

    uniform_samples = np.random.uniform(0, 1, size=(2, num_inst // 2))
    
    aux_1 = np.sqrt(-2 * np.log(uniform_samples[0, :]))
    aux_2 = 2 * np.pi * uniform_samples[1, :]

    samples = np.concatenate((aux_1 * np.cos(aux_2), aux_1 * np.sin(aux_2)))

    samples = loc + scale * samples

    if remove_extra_inst:
        return samples[1:]

    return samples


def _test():
    import matplotlib.pyplot as plt
    import scipy.stats


    plt.subplot(1, 2, 1)
    vals = np.linspace(-4, 4, 100)
    plt.plot(vals, scipy.stats.norm(loc=0, scale=1).pdf(vals))
    samples = sample_normal(num_inst=1000, random_state=16)
    plt.hist(samples, bins=64, density=True)
    plt.title("N(0, 1)")

    plt.subplot(1, 2, 2)
    vals = np.linspace(-20, 20, 100)
    plt.plot(vals, scipy.stats.norm(loc=6, scale=3).pdf(vals))
    samples = sample_normal(loc=6, scale=3, num_inst=1000, random_state=32)
    plt.hist(samples, bins=64, density=True)
    plt.title("N(6, 3)")

    plt.show()


if __name__ == "__main__":
    _test()
