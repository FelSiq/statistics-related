"""RANSAC linear fit.

http://www.cse.yorku.ca/~kosta/CompVis_Notes/ransac.pdf
http://www.cse.psu.edu/~rtc12/CSE486/lecture15_6pp.pdf
"""
import typing as t

import scipy.stats
import numpy as np
import warnings

import sklearn.linear_model


def _outlier_prob(y: np.ndarray) -> int:
    q_25, q_75 = np.quantile(y, (0.25, 0.75))
    iqr_whis = 1.5 * (q_75 - q_25)
    is_outlier = np.logical_or(y < q_25 - iqr_whis, y > q_75 + iqr_whis)
    p_outlier = np.sum(is_outlier) / y.size
    return p_outlier


def ransac(
        X: np.ndarray,
        y: np.ndarray,
        p: float = 0.99,
        fit_intercept: bool = True,
        random_state: t.Optional[int] = None,
        suppress_warnings: bool = False,
        verbose: bool = False,
) -> sklearn.linear_model.LinearRegression:
    """RANSAC linear fit."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    p_outlier = _outlier_prob(y)
    p_inliner = 1 - p_outlier
    num_param = int(fit_intercept) + X.shape[1]
    max_it = int(
        np.ceil(np.log(1. - p) / np.log(1. - (1. - p_outlier)**num_param)))

    tol = np.sqrt(scipy.stats.chi2.ppf(p_inliner, 1) * np.var(y))

    model = sklearn.linear_model.LinearRegression(fit_intercept=fit_intercept)

    if random_state is not None:
        np.random.seed(random_state)

    for it in np.arange(max_it):
        inds_train = np.random.choice(y.size, size=num_param, replace=False)

        X_sample = X[inds_train, :]
        y_sample = y[inds_train]

        model.fit(X=X_sample, y=y_sample)
        y_pred = model.predict(X)

        # Note: the right definition is actually the distance of the
        # point to the line.
        is_inliner = np.abs(y_pred - y) <= tol

        if is_inliner.size > 0 and np.mean(is_inliner) >= p_inliner:
            model.fit(X[is_inliner, :], y[is_inliner])

            if verbose:
                print("RANSAC convergence in iteration {} of {} ({:.2f}%)."
                      "".format(it + 1, max_it + 1,
                                100. * (it + 1) / (max_it + 1)))

            return model

    if not suppress_warnings:
        warnings.warn("Could not fit RANSAC model properly.", UserWarning)

    return model


def _test() -> None:
    import matplotlib.pyplot as plt
    import sklearn.linear_model

    size = 150
    p_outl = 0.15
    X = np.arange(size).reshape(-1, 1)
    y = np.arange(size) + np.random.randn(size)
    y[np.random.random(size) <= p_outl] *= 10

    model = ransac(X, y, random_state=16, verbose=True)
    modelr = sklearn.linear_model.RANSACRegressor().fit(X, y)

    plt.scatter(X, y)
    plt.plot(X, model.predict(X), color="red", label="fit line")
    plt.plot(X,
             modelr.predict(X),
             linestyle="--",
             color="purple",
             label="sklearn")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    _test()
