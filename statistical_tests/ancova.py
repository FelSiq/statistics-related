# Cool video explaining this test: https://www.youtube.com/watch?v=CqLGvwi-5Pc&list=PLblh5JKOoLUIzaEkCLIUxQFjPIlapw8nU&index=6
import typing as t

import numpy as np
import scipy.stats
import sklearn.linear_model


def _build_design_matrix(
    X_1: np.ndarray,
    X_2: np.ndarray,
    y_1: np.ndarray,
    y_2: np.ndarray,
):
    X_1 = np.asfarray(X_1)
    X_2 = np.asfarray(X_2)
    y_1 = np.asfarray(y_1)
    y_2 = np.asfarray(y_2)

    X_concat = np.vstack((X_1, X_2))

    X_design = np.column_stack(
        (
            np.hstack((np.zeros(y_1.size), np.ones(y_2.size))),
            X_concat,
        )
    )
    y_concat = np.concatenate((y_1, y_2))

    return X_design, y_concat


def _check_slopes(*args, threshold: float = 0.5):
    model = sklearn.linear_model.LinearRegression()
    slopes = []
    for X, y in zip(args[::2], args[1::2]):
        slope = model.fit(X, y).coef_[-1]
        slopes.append(slope)

    dist = np.ptp(slopes)
    assert dist < threshold, dist


def ancova(
    X_1: np.ndarray,
    X_2: np.ndarray,
    y_1: np.ndarray,
    y_2: np.ndarray,
    baseline: str = "full_regression",
    return_estimator: bool = False,
    check_slope_homogeneity: bool = True,
    max_slope_threshold: float = 0.5,
) -> t.Tuple[float, float]:
    assert baseline in {"full_regression", "global_average"}, baseline

    if check_slope_homogeneity:
        _check_slopes(X_1, y_1, X_2, y_2, threshold=max_slope_threshold)

    X_design, y_concat = _build_design_matrix(X_1, X_2, y_1, y_2)

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_design, y_concat)
    sqr_sum_resid = float(np.sum(np.square(y_concat - model.predict(X_design))))
    model_param_num = 3

    if baseline == "full_regression":
        model_baseline = sklearn.linear_model.LinearRegression()
        model_baseline.fit(X_concat, y_concat)
        sqr_sum_resid_baseline = float(
            np.sum(np.square(y_concat - model_baseline.predict(X_concat)))
        )
        baseline_param_num = 2

    else:
        sqr_sum_resid_baseline = np.var(y_concat, ddof=0)
        baseline_param_num = 1

    msb = sqr_sum_resid_baseline - sqr_sum_resid / (
        model_param_num - baseline_param_num
    )
    msw = sqr_sum_resid / (y_concat.size - model_param_num)

    F_stat = msb / msw

    null_dist = scipy.stats.f(1, 2 * (y_1.size - 1))
    p_val = null_dist.sf(F_stat)

    if return_estimator:
        return F_stat, p_val, model

    return F_stat, p_val


def _test():
    import matplotlib.pyplot as plt

    X_1 = 5 * np.random.random((15, 1))
    X_2 = 5 * np.random.random((15, 1))
    y_1 = X_1 * 3 + 4 + 1 * np.random.randn(X_1.shape[0], 1)
    y_2 = X_2 * 3.2 + 6 + 1 * np.random.randn(X_2.shape[0], 1)

    stat, p_val, model = ancova(
        X_1, X_2, y_1, y_2, return_estimator=True, baseline="global_average"
    )
    print(stat, p_val)

    y_preds_1 = model.predict(np.column_stack((np.zeros(15), X_1)))
    y_preds_2 = model.predict(np.column_stack((np.zeros(15), X_2)))

    min_p, max_p = np.quantile(np.vstack((X_1, X_2)), (0, 1))

    cols = y_1.size * ["r"] + y_2.size * ["b"]
    plt.plot(
        [min_p, max_p],
        model.predict([[0, min_p], [0, max_p]]),
        linestyle="--",
        color="red",
    )
    plt.plot(
        [min_p, max_p],
        model.predict([[1, min_p], [1, max_p]]),
        linestyle="--",
        color="blue",
    )
    plt.scatter(np.vstack((X_1, X_2)), np.vstack((y_1, y_2)), c=cols)
    plt.show()


if __name__ == "__main__":
    _test()
