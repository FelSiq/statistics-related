"""Simple implementation of Linear Regression."""
import numpy as np
import scipy.stats

import cross_validation


class LinRegressor:
    """Simple algorithm to fit a linear regression model.

    The strategy adopted is the Least Squares criterion.
    """

    def __init__(self):
        self.reg_coeff = None  # type: np.ndarray
        self.intercept = None  # type: np.ndarray
        self.residuals = None  # type: np.ndarray

        self.residual_sqr_sum = None  # type: float
        self.std_err_residual = None  # type: float
        self.sqr_err_residual = None  # type: float
        self.std_err_intercept = None  # type: float
        self.std_err_reg_coeff = None  # type: float

        self.conf_int_intercept = None  # type: np.ndarray
        self.conf_int_reg_coeff = None  # type: np.ndarray

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root mean squared error."""
        return np.sqrt(np.sum(np.square(y_true - y_pred)) / y_true.size)

    def _calc_errs(self, X: np.ndarray, x_mean: np.ndarray) -> None:
        """Calculate errors related to the fitted data."""
        self.residual_sqr_sum = np.sum(np.square(self.residuals))

        self.sqr_err_residual = self.residual_sqr_sum / (
            self.residuals.size - 2)
        self.std_err_residual = np.sqrt(self.sqr_err_residual)

        _aux = np.sum(np.square(X - x_mean))

        self.std_err_intercept = np.sqrt(
            self.sqr_err_residual *
            (1.0 / self.residuals.size + np.square(x_mean) / _aux))

        self.std_err_reg_coeff = np.sqrt(self.sqr_err_residual / _aux)

        _t_dist_int = np.asarray(
            scipy.stats.t.interval(alpha=0.975, df=self.residuals.size - 2),
            dtype=float)

        self.conf_int_intercept = self.intercept + _t_dist_int * self.std_err_intercept
        self.conf_int_reg_coeff = self.reg_coeff + _t_dist_int * self.std_err_reg_coeff

        self.t_stat_intercept = self.intercept / self.std_err_intercept
        self.t_stat_reg_coeff = self.reg_coeff / self.std_err_reg_coeff

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinRegressor":
        """Simple linear regression."""
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=float)

        if not isinstance(y, np.ndarray):
            y = np.asarray(y, dtype=float)

        _num_inst = X.size if X.ndim == 1 else X.shape[0]

        if _num_inst != y.size:
            raise ValueError("Number of instances (got {}) and 'y' "
                             "size (got {}) don't match!".format(
                                 _num_inst, y.size))

        x_mean = X.mean()
        y_mean = y.mean()

        _aux = X - x_mean

        self.reg_coeff = np.dot(_aux, y - y_mean) / np.dot(_aux, _aux)

        self.intercept = y_mean - self.reg_coeff * x_mean

        # Note: residuals are estimation of the true irredutible errors
        self.residuals = y - self.predict(X)

        self._calc_errs(X=X, x_mean=x_mean)

        return self

    def predict(self, vals: np.ndarray) -> np.ndarray:
        """Predict the fitted function values for ``vals``.

        Let a be the intercept coefficients and b the regression coefficiets.
        Then, this methods simply calculates y_{i} = f(vals_{i}) as follows:

            y_{i} = a + b * vals_{i}

        Arguments
        ---------
        vals : :obj:`np.ndarray`
            Points to evaluate model function.

        Returns
        -------
        :obj:`np.ndarray`
            Array such that every entry is y_{i} = f(vals_{i}).
        """
        return vals * self.reg_coeff + self.intercept


class MultivarLinRegressor:
    """Simple algorithm to fit multivariate linear regression model."""

    def __init__(self):
        """Fit multivariate linear regression models."""
        self.coeffs = None  # type: np.ndarray

    @staticmethod
    def _augment_x(X: np.ndarray) -> np.ndarray:
        """Append a column of 1's in ``X``.

        Useful to calculate both multivariate coefficients and intercept
        together.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        _num_inst, _ = X.shape

        return np.hstack((X, np.ones((_num_inst, 1))))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultivarLinRegressor":
        """Fit data into model, calculating the corresponding coefficients."""
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        X_aug = MultivarLinRegressor._augment_x(X)

        _M = np.matmul(X_aug.T, X_aug)
        _Y = np.matmul(X_aug.T, y)
        self.coeffs = np.linalg.solve(_M, _Y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Use the fitted model to predict unknown values."""
        return np.matmul(MultivarLinRegressor._augment_x(X), self.coeffs)


def _test_univar_lin_reg_01() -> None:
    import matplotlib.pyplot as plt
    random_state = 16

    np.random.seed(random_state)
    pop_size = 30
    num_folds = 9

    X = np.arange(pop_size) + np.random.normal(
        loc=0.0, scale=0.01, size=pop_size)
    y = np.arange(pop_size) + np.random.normal(
        loc=0.0, scale=1.0, size=pop_size)

    pop = np.hstack((X.reshape(-1, 1), y.reshape(-1, 1)))

    errors = np.zeros(num_folds)

    for fold_id, fold in enumerate(
            cross_validation.kfold_cv(
                pop, k=num_folds, random_state=random_state)):
        data_test, data_train = fold

        X_test, y_test = data_test[:, 0], data_test[:, 1]
        X_train, y_train = data_train[:, 0], data_train[:, 1]

        model = LinRegressor().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        errors[fold_id] = model.rmse(y_test, y_pred)

        plt.subplot(3, 3, 1 + fold_id)
        plt.plot(X, y, label="True data")
        plt.plot(
            X_test, y_pred, 'o', label="RMSE: {:.2f}".format(errors[fold_id]))
        plt.legend()
        plt.title(str(fold_id))

    total_error = LinRegressor.rmse(errors, np.zeros(num_folds))
    print("Total RMSE:", total_error)

    plt.show()


def _test_univar_lin_reg_02() -> None:
    import sklearn.datasets

    boston = sklearn.datasets.load_boston()

    X_boston = boston.data[:, boston.feature_names == "LSTAT"].ravel()
    y_boston = boston.target

    model = LinRegressor().fit(X=X_boston, y=y_boston)

    print("RSS:", model.residual_sqr_sum)
    print("RSE:", model.std_err_residual)
    print("RSE^2:", model.sqr_err_residual)
    print("Intercept:", model.intercept)
    print("ErrIntercept:", model.std_err_intercept)
    print("RegCoef:", model.reg_coeff)
    print("ErrCoef:", model.std_err_reg_coeff)
    print("95% intercept conf interval:", model.conf_int_intercept)
    print("95% regCoeff conf interval:", model.conf_int_reg_coeff)
    print("t-stat intercept", model.t_stat_intercept)
    print("t-stat reg_coeff", model.t_stat_reg_coeff)


if __name__ == "__main__":
    # _test_univar_lin_reg_01()
    _test_univar_lin_reg_02()
