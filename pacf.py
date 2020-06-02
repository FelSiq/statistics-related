"""Estimate PACF using OLS.

PACF stands for `Partial Aucorrelation Function`.
"""
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools
import statsmodels.tsa.arima_model
import sklearn.linear_model
import numpy as np

import embed


def pacf(ts: np.ndarray, nlags: int, unbiased: bool = True) -> np.ndarray:
    """Calculate the PACF values of the given time-series."""
    pacf_vals = np.zeros(nlags, dtype=float)

    model = sklearn.linear_model.LinearRegression()

    for k in np.arange(nlags):
        ts_embed = embed.embed_ts(ts, dim=k + 1, lag=1, include_val=True)
        X, y = ts_embed[:, 1:], ts_embed[:, 0]
        model.fit(X=X, y=y)
        pacf_vals[k] = model.coef_[-1]

        if unbiased:
            pacf_vals[k] *= ts.size / ts_embed.shape[0]

    return pacf_vals


def plot_pacf(pacf_vals: np.ndarray, ts: np.ndarray):
    """Plot PACF."""
    nlags = pacf_vals.size

    plt.subplot(2, 1, 1)
    plt.title("Time series")
    plt.plot(ts)

    plt.subplot(2, 1, 2)
    crit_val = 1.96 / np.sqrt(ts.size)
    plt.title("PACF (Critical value: {:.4f})".format(crit_val))
    plt.hlines(y=[-crit_val, +crit_val],
               xmin=0,
               xmax=nlags + 1,
               linestyle="--",
               color="blue")
    plt.hlines(y=0, xmin=0, xmax=nlags + 1, linestyle="dotted", color="black")
    plt.vlines(x=np.arange(1, nlags + 1),
               ymin=0,
               ymax=pacf_vals,
               linestyle="-",
               color="black")
    plt.xlim((0.5, nlags + 0.5))
    plt.ylim((-1.1, 1.1))


def _test() -> None:
    ts_size = 64
    ts_period = 12
    np.random.seed(16)

    ts = np.array(
        [0.01 * i + 0.05 * (i % ts_period) for i in np.arange(ts_size)])
    ts += 0.2 * np.random.random(ts_size)

    nlags = 8
    pacf_a = pacf(ts, nlags=nlags)
    pacf_b = statsmodels.tsa.stattools.pacf(ts, nlags=8,
                                            method="ols-unbiased")[1:]
    print(pacf_a)
    print(pacf_b)
    assert np.allclose(pacf_a, pacf_b)

    plot_pacf(pacf_a, ts)
    plt.show()


if __name__ == "__main__":
    _test()
