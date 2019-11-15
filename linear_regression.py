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

        self.residual_sum_sqr = None  # type: float
        self.std_err_residual = None  # type: float
        self.sqr_err_residual = None  # type: float
        self.std_err_intercept = None  # type: float
        self.std_err_reg_coeff = None  # type: float

        self.conf_int_intercept = None  # type: np.ndarray
        self.conf_int_reg_coeff = None  # type: np.ndarray

        self.t_test_pval_intercept = None  # type: float
        self.t_test_pval_reg_coeff = None  # type: float

        self.r_sqr_stat = None  # type: float

        self.f_stat = None  # type: float
        self.f_stat_pval = None  # type: float

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root mean squared error."""
        return np.sqrt(np.sum(np.square(y_true - y_pred)) / y_true.size)

    @staticmethod
    def _ttest(t_stat_val: float, df: float) -> float:
        """Calculate the p-value from the t-student test."""
        return 2.0 * scipy.stats.t(df=df).sf(np.abs(t_stat_val))

    def _calc_rse(self) -> float:
        """Calculate the Residual Standard Error (RSE).

        Notes
        -----
        The RSE is a measure of ``lack of fit of the model to the data.``
        Therefore, the greater is the RSE, the worse the model fit to
        the training data. However, the RSE is not unit-less (more
        precisely, its units is the same as the dependent variable),
        which means that it is problem dependent, and it is not always
        easy to define which is a ``good/low RSE.``
        """
        self.sqr_err_residual = self.residual_sum_sqr / (
            self.residuals.size - 2)

        self.std_err_residual = np.sqrt(self.sqr_err_residual)

        return self.std_err_residual

    def _calc_f_stat(self) -> float:
        r"""Calculate the F-stat.

        In a simple linear regression (only a single independent variable)
        the F-stat is just the square of the t-stat of the
        regression angular coefficient.
        $$
        \text{F-stat} = \text{t-stat}^{2}
        $$
        Therefore, the F-statistic is not useful (given the t-statistic
        and the t-statistic test) in a simple linear regression model.

        However, in a multiple linear regression model, the F-statistic
        can be use to perform a hypothesis test of wether ANY of the
        model coefficients are different from zero, and it is not
        replaced by the individual p-values of each coefficient due to
        false positives given a significance level $\alpha$ (i.e., it is
        expected that $100 * \alpha$ zero coefficients are assumed
        significant, or non-zero, due to random chance.)
        """
        self.f_stat = np.square(self.t_stat_reg_coeff)

        # Note: this F-statistic p-value value will be *exactly* the same as
        # the t-statistic test p-value
        self.f_stat_pval = scipy.stats.f(
            dfn=1, dfd=self.residuals.size - 1).sf(np.abs(self.f_stat))

        return self.f_stat

    def _calc_r_sqr_stat(self) -> float:
        r"""Calculate the $R^{2}$ stat.

        The $R^{2}$ stat is calculated as
        $$
        R^{2} = \frac{\text{TSS} - \text{RSS}}{\text{TSS}}
              = 1.0 - \frac{\text{RSS}}{\text{TSS}}
        $$
        where ``TSS`` is the Total Sum of Squares defined as
        $$
        \text{TSS} = \sum_{i=1}^{n}(y_{i} - \bar{y})^{2}
        $$
        where $\bar{y} = \frac{1}{n}\sum_{i=1}^{n}y_{i}$ is the mean value of
        the dependent variable, and it calculates the total of variability
        inherently in the dependent variable (independently of the regression
        results), and ``RSS`` is the Residual Sum of Squares defined as
        $$
        \text{RSS} = \sum_{i=1}^{n}\epsilon^{2}
                   = \sum_{i=1}^{n}(y_{i} - \hat{y}_{i})^{2}
                   = \sum_{i=1}^{n}(y_{i} - (\beta_{0} + \beta_{1}x_{i}))^{2}
        $$
        where $y_{i}$ is the true value of the $i$th dependent variable, and
        $x_{i}$ is the value of the $i$th independent variable.

        It calculates the proportion of variability in the dependent variable
        can be explained using the independent variable.

        The range of this stat is [0, 1], where 1.0 means that all
        variance in the dependent variable $y$ was explained by the
        independent variable $x$, after regressing $y$ onto $x$. This
        stat is unit-less.

        Notes
        -----
        Just like the RSE (Residual Standard Error), it is difficult to
        select a ``good/high value for the $R^{2}$ stat.``, as it varies
        among different applications and subjects.

        In the scenario where there is a single independent variable, then
        $R^{2} = r^{2} = \text{COR(x, y)}^{2}$, where $r = COR(x, y)$ is the
        Person correlation coefficient.
        """
        self.r_sqr_stat = 1.0 - self.residual_sum_sqr / self.total_sum_sqr

        return self.r_sqr_stat

    def _calc_errs(self, X: np.ndarray, y: np.ndarray, x_mean: float,
                   y_mean: float) -> None:
        """Calculate errors related to the fitted data."""
        # TSS (Total Sum of Squares) measures the total variance in the
        # dependent variable. It is the variability inherent in y before
        # the regression is even performed.
        # Source: ISLR (7th ed.), page 70 (Chap.3 - Linear Regression)
        self.total_sum_sqr = np.sum(np.square(y - y_mean))

        # RSS (Residual Sum of Squares) measures the amount of variability
        # that is left unexplained after performing the regression.
        # Source: ISLR (7th ed.), page 70 (Chap.3 - Linear Regression)
        self.residual_sum_sqr = np.sum(np.square(self.residuals))

        self._calc_rse()
        self._calc_r_sqr_stat()

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

        self.t_test_pval_intercept = self._ttest(
            t_stat_val=self.t_stat_intercept, df=self.residuals.size - 1)
        self.t_test_pval_reg_coeff = self._ttest(
            t_stat_val=self.t_stat_reg_coeff, df=self.residuals.size - 1)

        self._calc_f_stat()

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

        self._calc_errs(X=X, y=y, x_mean=x_mean, y_mean=y_mean)

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


class MultipleLinRegressor:
    """Simple algorithm to fit Multiple Linear Regression model."""

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultipleLinRegressor":
        """Fit data into model, calculating the corresponding coefficients."""
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        X_aug = MultipleLinRegressor._augment_x(X)

        _M = np.matmul(X_aug.T, X_aug)
        _Y = np.matmul(X_aug.T, y)

        # Note: the interpretation of the coefficients related to some
        # dependent variable (i.e., all coefficients other than the
        # intercept) is 'the average effect on Y for a one unit increase in
        # the corresponding X, while kept all other independent variables
        # fixed.'
        self.coeffs = np.linalg.solve(_M, _Y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Use the fitted model to predict unknown values."""
        return np.matmul(MultipleLinRegressor._augment_x(X), self.coeffs)


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

    print("RSS:", model.residual_sum_sqr)
    print("RSE:", model.std_err_residual)
    print("RSE^2:", model.sqr_err_residual)
    print("Intercept:", model.intercept)
    print("RegCoef:", model.reg_coeff)
    print("ErrIntercept:", model.std_err_intercept)
    print("ErrCoef:", model.std_err_reg_coeff)
    print("95% intercept conf interval:", model.conf_int_intercept)
    print("95% regCoeff conf interval:", model.conf_int_reg_coeff)
    print("t-stat intercept", model.t_stat_intercept)
    print("t-stat regCoeff", model.t_stat_reg_coeff)
    print("t-stat p-value intercept", model.t_test_pval_intercept)
    print("t-stat p-value regCoeff", model.t_test_pval_reg_coeff)
    print("F-stat", model.f_stat)
    print("F-stat p-value", model.f_stat_pval)

    assert np.isclose(model.r_sqr_stat,
                      np.corrcoef(X_boston, y_boston)[1, 0]**2)


if __name__ == "__main__":
    # _test_univar_lin_reg_01()
    _test_univar_lin_reg_02()
