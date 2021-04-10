import typing as t

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


def qq_plot(
    data: np.ndarray, dist: t.Union[scipy.stats.rv_discrete, scipy.stats.rv_continuous]
):
    sorted_data = np.sort(data)[:-1]
    probs = np.linspace(0, 1, data.size + 1)[1:-1]
    dist_quants = dist.ppf(probs)

    slope, intercept = scipy.stats.linregress(dist_quants, sorted_data)[:2]

    return dist_quants, sorted_data


def _test():
    data = -3 + 4 * np.random.randn(100)
    dist = scipy.stats.norm()

    dist_quant, data = qq_plot(data, dist)

    (osm, osr), _ = scipy.stats.probplot(x=data, dist=dist)

    plt.scatter(dist_quant, data, label="mine", color="blue")
    plt.scatter(osm, osr, label="ref", color="orange")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    _test()
