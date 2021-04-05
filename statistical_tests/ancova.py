# Cool video explaining this test: https://www.youtube.com/watch?v=CqLGvwi-5Pc&list=PLblh5JKOoLUIzaEkCLIUxQFjPIlapw8nU&index=6
import typing as t

import numpy as np
import scipy.stats
import sklearn.linear_model


def _check_X_y(X, y):
    X = np.asfarray(X)
    y = np.asfarray(y)

    if X.ndim == 1:
        X = np.expand_dims(X, axis=1)

    return X, y


def _build_design_matrix(
    X_1: np.ndarray,
    X_2: np.ndarray,
    y_1: np.ndarray,
    y_2: np.ndarray,
):

    X_concat = np.vstack((X_1, X_2))

    X_design = np.column_stack(
        (
            np.ones(y_1.size + y_2.size),
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


def _calc_baseline(X_concat, y_concat, baseline):
    assert baseline in {"full_regression", "group_average", "global_average"}, baseline

    model_baseline = sklearn.linear_model.LinearRegression(fit_intercept=False)

    if baseline == "full_regression":
        X_slice = X_concat[:, [0, 2]]

    elif baseline == "group_average":
        X_slice = X_concat[:, [0, 1]]

    else:
        X_slice = X_concat[:, 0, np.newaxis]

    model_baseline.fit(X_slice, y_concat)
    sqr_sum_resid_baseline = float(
        np.sum(np.square(y_concat - model_baseline.predict(X_slice)))
    )

    baseline_param_num = model_baseline.coef_.size

    return sqr_sum_resid_baseline, baseline_param_num, model_baseline


def ancova(
    X_1: np.ndarray,
    X_2: np.ndarray,
    y_1: np.ndarray,
    y_2: np.ndarray,
    baseline: str = "full_regression",
    return_estimators: bool = False,
    check_slope_homogeneity: bool = True,
    max_slope_threshold: float = 0.5,
) -> t.Tuple[float, float]:
    X_1, y_1 = _check_X_y(X_1, y_1)
    X_2, y_2 = _check_X_y(X_2, y_2)

    if check_slope_homogeneity:
        _check_slopes(X_1, y_1, X_2, y_2, threshold=max_slope_threshold)

    X_design, y_concat = _build_design_matrix(X_1, X_2, y_1, y_2)

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_design, y_concat)
    sqr_sum_resid = float(np.sum(np.square(y_concat - model.predict(X_design))))
    model_param_num = 3

    sqr_sum_resid_baseline, baseline_param_num, model_baseline = _calc_baseline(
        X_design, y_concat, baseline
    )
    msb = sqr_sum_resid_baseline - sqr_sum_resid / (
        model_param_num - baseline_param_num
    )
    msw = sqr_sum_resid / (y_concat.size - model_param_num)

    F_stat = msb / msw

    null_dist = scipy.stats.f(1, 2 * (y_1.size - 1))
    p_val = null_dist.sf(F_stat)

    if return_estimators:
        return F_stat, p_val, model, model_baseline

    return F_stat, p_val


def _test():
    import matplotlib.pyplot as plt

    baseline = "full_regression"
    X_1 = 5 * np.random.random(15)
    X_2 = 5 * np.random.random(15)
    y_1 = X_1 * 3 + 4 + 1 * np.random.randn(X_1.shape[0])
    y_2 = X_2 * 3.2 + 4 + 1 * np.random.randn(X_2.shape[0])

    stat, p_val, model, model_baseline = ancova(
        X_1, X_2, y_1, y_2, return_estimators=True, baseline=baseline
    )

    min_p, max_p = np.quantile(np.vstack((X_1, X_2)), (0, 1))

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))

    cols = y_1.size * ["r"] + y_2.size * ["b"]
    ax1.plot(
        [min_p, max_p],
        model.predict([[1, 0, min_p], [1, 0, max_p]]),
        linestyle="--",
        color="red",
    )
    ax1.plot(
        [min_p, max_p],
        model.predict([[1, 1, min_p], [1, 1, max_p]]),
        linestyle="--",
        color="blue",
    )

    baseline_input = {
        "full_regression": ([[1, min_p], [1, max_p]], None),
        "group_average": ([[1, 0], [1, 0]], [[1, 1], [1, 1]]),
        "global_average": ([[1], [1]], None),
    }

    inp_1, inp_2 = baseline_input[baseline]

    ax2.plot(
        [min_p, max_p],
        model_baseline.predict(inp_1),
        linestyle="--",
        color="red" if baseline == "group_average" else "black",
    )

    if baseline == "group_average":
        ax2.plot(
            [min_p, max_p],
            model_baseline.predict(inp_2),
            linestyle="--",
            color="blue",
        )

    ax1.scatter(np.vstack((X_1, X_2)), np.vstack((y_1, y_2)), c=cols)
    ax1.set_title(f"Test statistic: {stat:.4}")
    ax2.scatter(np.vstack((X_1, X_2)), np.vstack((y_1, y_2)), c=cols)
    ax2.set_title(f"p-value: {p_val:.4}")

    plt.show()


if __name__ == "__main__":
    _test()
