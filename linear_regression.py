"""Simple implementation of Linear Regression."""
# pylint: disable=C0103, E1101, R0902
import typing as t

import numpy as np
import scipy.stats
import scipy.linalg

import cross_validation


class LinRegressor:
    """Simple algorithm to fit a linear regression model.

    The strategy adopted is the Least Squares criterion.
    """

    def __init__(self):
        self.reg_coeff = None  # type: np.ndarray
        self.intercept = None  # type: np.ndarray
        self.residuals = None  # type: np.ndarray

        self._num_samples = 0

        self.total_sum_sqr = None  # type: float

        self.residual_sum_sqr = None  # type: float
        self.std_err_residual = None  # type: float
        self.sqr_err_residual = None  # type: float
        self.std_err_intercept = None  # type: float
        self.std_err_reg_coeff = None  # type: float

        self.conf_int_intercept = None  # type: np.ndarray
        self.conf_int_reg_coeff = None  # type: np.ndarray

        self.t_stat_intercept = None  # type: float
        self.t_stat_reg_coeff = None  # type: float

        self.t_test_pval_intercept = None  # type: float
        self.t_test_pval_reg_coeff = None  # type: float

        self.r_sqr_stat = None  # type: float
        self.r_sqr_adj_stat = None  # type: float

        self.f_stat = None  # type: float
        self.f_stat_pval = None  # type: float

        self.leverage = None  # type: np.ndarray

        self.loocv_err = None  # type: float

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
        self.sqr_err_residual = self.residual_sum_sqr / (self._num_samples - 2)

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
            dfn=1, dfd=self._num_samples - 2).sf(np.abs(self.f_stat))

        return self.f_stat

    def _calc_r_sqr_stat(self) -> t.Tuple[float, float]:
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
        $x_{i}$ is the value of the $i$th independent variable, and it express
        the amount of variability captured by the residuals (i.e., not
        explained by the model coefficients.)

        Therefore, $\text{TSS} - \text{RSS}$ is the amount of variability of
        the dependent variable captured by the model, and dividing it by
        $\text{TSS}$, we calculate the proportion of variability of the
        dependent variable that can be explained using the independent variable.

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
        if np.isclose(self.total_sum_sqr, 0.0):
            self.r_sqr_stat = np.nan
            self.r_sqr_adj_stat = np.nan
            return np.nan, np.nan

        self.r_sqr_stat = 1.0 - self.residual_sum_sqr / self.total_sum_sqr

        self.r_sqr_adj_stat = (
            1.0 - (self.residual_sum_sqr * (self._num_samples - 1)) /
            (self.total_sum_sqr * (self._num_samples - 2)))

        return self.r_sqr_stat, self.r_sqr_adj_stat

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

        _x_dist_sqr = np.square(X - x_mean)
        _x_dist_total = np.sum(_x_dist_sqr)

        self.leverage = 1 / self._num_samples + _x_dist_sqr / _x_dist_total

        self.std_err_intercept = np.sqrt(
            self.sqr_err_residual *
            (1.0 / self._num_samples + np.square(x_mean) / _x_dist_total))

        self.std_err_reg_coeff = np.sqrt(self.sqr_err_residual / _x_dist_total)

        _t_dist_int = np.asarray(
            scipy.stats.t.interval(alpha=0.975, df=self._num_samples - 2),
            dtype=float)

        self.conf_int_intercept = self.intercept + _t_dist_int * self.std_err_intercept
        self.conf_int_reg_coeff = self.reg_coeff + _t_dist_int * self.std_err_reg_coeff

        self.t_stat_intercept = self.intercept / self.std_err_intercept
        self.t_stat_reg_coeff = self.reg_coeff / self.std_err_reg_coeff

        self.t_test_pval_intercept = self._ttest(
            t_stat_val=self.t_stat_intercept, df=self._num_samples - 2)
        self.t_test_pval_reg_coeff = self._ttest(
            t_stat_val=self.t_stat_reg_coeff, df=self._num_samples - 2)

        self._calc_f_stat()

        # Leave-One-Out Cross Validation (LOOCV)
        self.loocv_err = np.sum(
            np.square(self.residuals /
                      (1.0 - self.leverage))) / self._num_samples

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinRegressor":
        """Simple linear regression."""
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=float)

        if not isinstance(y, np.ndarray):
            y = np.asarray(y, dtype=float)

        self._num_samples = X.size if X.ndim == 1 else X.shape[0]

        if self._num_samples != y.size:
            raise ValueError("Number of instances (got {}) and 'y' "
                             "size (got {}) don't match!".format(
                                 self._num_samples, y.size))

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

    def __init__(self, calc_stats: bool = True):
        """Fit multivariate linear regression models."""
        self.coeffs = None  # type: np.ndarray
        self.residuals = None  # type: np.ndarray
        self.calc_stats = calc_stats

        self._num_samples = 0

        self.add_intercept = False

        self.mallows_cp = None  # type: float

        self.leverage = None  # type: np.ndarray

        self.residual_sum_sqr = None  # type: float

        self.total_sum_sqr = None  # type: float

        self.std_err_residual = None  # type: float
        self.sqr_err_residual = None  # type: float
        self.std_err_coeffs = None  # type: float

        self.r_sqr_stat = None  # type: float
        self.r_sqr_adj_stat = None  # type: float

        self.t_stat = None  # type: np.ndarray
        self.t_test_pval = None  # type: np.ndarray
        self.t_stat_pval_coeffs = None  # type: np.ndarray

        self.f_stat = None  # type: float
        self.f_stat_pval = None  # type: float

        self.var_inflation_factor = None  # type: np.ndarray

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root mean squared error."""
        return np.sqrt(np.sum(np.square(y_true - y_pred)) / y_true.size)

    @staticmethod
    def _augment_x(X: np.ndarray) -> np.ndarray:
        """Append a column of 1's in ``X``.

        Useful to calculate both multivariate coefficients and intercept
        together.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        _num_inst, _ = X.shape

        return np.hstack((np.ones((_num_inst, 1)), X))

    def _calc_std_errs(self) -> np.ndarray:
        """Calculate the standard errors for every model coefficient."""
        self.sqr_err_residual = (
            self.residual_sum_sqr / (self._num_samples - self.coeffs.size))

        self.std_err_residual = np.sqrt(self.sqr_err_residual)

        self.std_err_coeffs = self.coeffs.ravel() / self.t_stat

    def _calc_t_stat(self, X_aug: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate the t-statistics related to every coefficient."""
        ang_coeffs_num = self.coeffs.size - 1

        self.t_stat = np.zeros(self.coeffs.size, dtype=float)
        self.t_test_pval = np.zeros(self.coeffs.size, dtype=float)

        _t_stat_model = scipy.stats.t(
            df=self._num_samples - ang_coeffs_num - 1)

        col_ind = np.arange(self.coeffs.size)

        model = MultipleLinRegressor(calc_stats=False)

        for i in np.arange(self.coeffs.size):
            X_aug_mod = X_aug[:, np.delete(col_ind, i)]

            model.fit(X=X_aug_mod, y=y, add_intercept=False)

            rss_model = model.residual_sum_sqr

            f_stat = (
                (self._num_samples - ang_coeffs_num - 1) *
                (rss_model - self.residual_sum_sqr)) / self.residual_sum_sqr

            self.t_stat[i] = np.sign(self.coeffs[i]) * np.sqrt(np.abs(f_stat))
            self.t_test_pval[i] = _t_stat_model.sf(np.abs(self.t_stat[i]))

        return self.t_stat

    def _calc_f_stat(self) -> float:
        """Calculate the F-statistic related to the model."""
        ang_coeffs_num = self.coeffs.size - 1

        if ang_coeffs_num == 0:
            self.f_stat = np.nan
            return self.f_stat

        self.f_stat = (((self._num_samples - ang_coeffs_num - 1) *
                        (self.total_sum_sqr - self.residual_sum_sqr)) /
                       (ang_coeffs_num * self.residual_sum_sqr))

        self.f_stat_pval = scipy.stats.f(
            dfn=ang_coeffs_num, dfd=self._num_samples - ang_coeffs_num + 1).sf(
                np.abs(self.f_stat))

        return self.f_stat

    def _calc_r_sqr_stat(self) -> t.Tuple[float, float]:
        """Calculate the $R^{2}$ statistic."""
        if np.isclose(self.total_sum_sqr, 0.0):
            self.r_sqr_stat = np.nan
            self.r_sqr_adj_stat = np.nan
            return np.nan, np.nan

        self.r_sqr_stat = 1.0 - self.residual_sum_sqr / self.total_sum_sqr

        self.r_sqr_adj_stat = (
            1.0 - (self.residual_sum_sqr * (self._num_samples - 1)) /
            (self.total_sum_sqr * (self._num_samples - self.coeffs.size)))

        return self.r_sqr_stat, self.r_sqr_adj_stat

    def _calc_vif(self, X_aug: np.ndarray) -> np.ndarray:
        """Calculate the Variance Inflation Factor (VIF).

        The VIF of each parameter $b_{j}$ is the ratio of the variance of
        $b_{j}$ when fitting the full model divided by the variance of
        $b_{j}$ if fit on its own.

        It is used to detect multicolinearity (i.e., collinearity among
        a indefinite amount of variables). The correlation matrix can be
        used to detect collinearity between two variables, but can not
        detect multicolinearity of three or more variables. For that
        matter, the VIF can be used.

        The smallest value for the VIF is 1, and ``high`` (typically more
        than 5 or 10 - although it may be problem dependent) values may
        represents high multicolinearity between that independent variable
        and a subset of other independent variables.

        The collinearity of independent variable is a problem in a linear
        model because it makes the values of parameters difficult to
        estimate, thus giving weak confidence intervals and statistical
        guarantees about the model. In other words, collinearity between
        independent variables may masks the importance of each variable.
        """
        self.var_inflation_factor = np.zeros(self.coeffs.size)

        var_inds = np.arange(self.coeffs.size)
        model = MultipleLinRegressor(calc_stats=False)

        for i in var_inds:
            model.fit(
                X=X_aug[:, np.delete(var_inds, i)],
                y=X_aug[:, i],
                add_intercept=False)

            self.var_inflation_factor[i] = 1.0 / (1.0 - model.r_sqr_stat)

        return self.var_inflation_factor

    def _calc_leverage(self, X: np.ndarray) -> np.ndarray:
        """Calculates leverage for all observations."""
        cho_factor = scipy.linalg.cholesky(np.matmul(X.T, X), lower=False)
        z_vector = scipy.linalg.solve(cho_factor.T, X.T)
        self.leverage = np.sum(np.square(z_vector), axis=0)
        return self.leverage

    def _calc_mallows_cp(self) -> float:
        self.mallows_cp = ((self.residual_sum_sqr + 2 *
                            (self.coeffs.size - 1) * np.var(self.residuals)) /
                           self._num_samples)

        return self.mallows_cp

    def _calc_errs(self, X_aug: np.ndarray, y: np.ndarray) -> None:
        """Calculate errors associated with the model and the fitted data."""
        y_mean = np.mean(y)
        self.total_sum_sqr = np.sum(np.square(y - y_mean))
        self.residual_sum_sqr = np.sum(np.square(self.residuals))
        self._calc_r_sqr_stat()

        if self.calc_stats:
            self._calc_f_stat()
            self._calc_t_stat(X_aug=X_aug, y=y)
            self._calc_std_errs()
            self._calc_vif(X_aug=X_aug)
            self._calc_leverage(X=X_aug[:, 1:])

    def fit(self,
            X: t.Optional[np.ndarray],
            y: np.ndarray,
            lambda_: float = 0.0,
            add_intercept: bool = True) -> "MultipleLinRegressor":
        """Fit data into model, calculating the corresponding coefficients.

        Arguments
        ---------
        X : None or :obj:`np.ndarray`, shape = [num_inst, num_attr]
            Train observations as a numpy array, where eahc line is a
            distinct observation and each column is a distinct independent
            attribute related to the correspondent observation. If
            None, then ``X`` is assumed to be a constant vector of ones of
            shape [num_inst, 1] (i.e., just related to the intercept term.)

        y : :obj:`np.ndarray`
            Dependent attribute. The multiple regression will be ``y`` onto
            ``X.``

        lambda_ : :obj:`float`, optional
            Shrinkage factor for Ridge Regression, which balances model
            accuracy with model complexity. If lambda_ == 0, then the
            ordinary linear regression will be performed. Note that if
            lambda_ > 0, then it is expected that both ``X`` and ``y``
            are standardized (mean 0 and variance 1.)

            The Ridge regression is useful in the presence of highly
            correlated variables and insufficient data to separate
            correctly the effect of each variable correctly.

            Note that this parameter is data-dependent and must be
            tunned (by cross-validation, for instance) to work effectively.

        add_intercept : :obj:`bool`, optional
            If True, add a constant column of ones to the ``X`` data
            in order to represent the intercept term of the regression.
            If your ``X`` data already have this constant column, you
            can set this argument to False. This argument is ignored
            if ``X`` is None.

        Return
        ------
        self
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.add_intercept = add_intercept

        if X is None or X.size == 0:
            X_aug = X = np.ones(shape=(y.size, 1))
            self.add_intercept = False

        elif add_intercept:
            X_aug = MultipleLinRegressor._augment_x(X)

        else:
            X_aug = X

        self._num_samples = X_aug.shape[0]

        _M = np.matmul(X_aug.T, X_aug) + np.diag(
            np.repeat(lambda_, X_aug.shape[1]))
        _Y = np.matmul(X_aug.T, y)

        # Note: the interpretation of the coefficients related to some
        # dependent variable (i.e., all coefficients other than the
        # intercept) is 'the average effect on Y for a one unit increase in
        # the corresponding X, while kept all other independent variables
        # fixed.'
        self.coeffs = np.linalg.solve(_M, _Y)

        self.residuals = y - self.predict(X)

        self._calc_errs(X_aug=X_aug, y=y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Use the fitted model to predict unknown values."""
        if self.add_intercept:
            X = MultipleLinRegressor._augment_x(X)

        if X is None or X.size == 0:
            X = np.ones((self.coeffs.size, 1))

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return np.matmul(X, self.coeffs)


class ModelSelection:
    """Select Multiple Linear regression models with different startegies."""

    def __init__(self):
        self.X = None  # type: np.ndarray
        self.y = None  # type: np.ndarray
        self.best_model = None  # type: t.Optional[MultipleLinRegressor]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ModelSelection":
        """."""
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=float)

        else:
            X = np.copy(X)

        if not isinstance(y, np.ndarray):
            y = np.asarray(y, dtype=float)

        else:
            y = np.copy(y)

        self.X = X
        self.y = y

        return self

    def _forward_get_all_models(self, verbose: bool = False) -> np.ndarray:
        predictors_num = self.X.shape[1]
        models = np.zeros(1 + predictors_num, dtype=object)
        models[0] = MultipleLinRegressor(calc_stats=False).fit(
            X=None, y=self.y)
        chosen_preds = []  # type: t.List[int]

        for pred_num in np.arange(predictors_num):
            best_cur_model = None
            best_cur_model_pred_ind = -1

            for cur_pred_ind in np.delete(
                    np.arange(predictors_num), chosen_preds):
                cur_preds = np.hstack((chosen_preds, cur_pred_ind)).astype(int)
                X_cur = self.X[:, cur_preds]
                cur_model = MultipleLinRegressor(calc_stats=False).fit(
                    X=X_cur, y=self.y)

                if best_cur_model is None or cur_model.r_sqr_stat >= best_cur_model.r_sqr_stat:
                    best_cur_model = cur_model
                    best_cur_model_pred_ind = cur_pred_ind

            models[pred_num + 1] = best_cur_model
            chosen_preds.append(best_cur_model_pred_ind)

            if verbose:
                r_sqr_cur = models[pred_num + 1].r_sqr_stat
                r_sqr_prev = models[pred_num].r_sqr_stat
                print(
                    "Predictors: {} - Current R^2: {:.4f} (relative increase of {:.2f}%)"
                    .format(pred_num + 1, r_sqr_cur,
                            100 * (1.0 - r_sqr_prev / r_sqr_cur)))

        return chosen_preds

    def _backward_get_all_models(self, verbose: bool = False) -> np.ndarray:
        predictors_num = self.X.shape[1]
        models = np.zeros(1 + predictors_num, dtype=object)
        models[0] = MultipleLinRegressor(calc_stats=False).fit(
            X=self.X, y=self.y)
        deleted_preds = np.zeros(predictors_num, dtype=int)
        active_preds = np.arange(predictors_num)

        for pred_num in np.arange(predictors_num):
            best_cur_model = None
            best_cur_model_pred_ind = -1
            best_cur_del_ind = -1

            for cur_del_ind, cur_pred_ind in enumerate(active_preds):
                cur_preds = np.delete(active_preds, cur_del_ind)
                X_cur = self.X[:, cur_preds]
                cur_model = MultipleLinRegressor(calc_stats=False).fit(
                    X=X_cur, y=self.y)

                if best_cur_model is None or cur_model.r_sqr_stat >= best_cur_model.r_sqr_stat:
                    best_cur_model = cur_model
                    best_cur_model_pred_ind = cur_pred_ind
                    best_cur_del_ind = cur_del_ind

            models[pred_num + 1] = best_cur_model
            deleted_preds[pred_num] = best_cur_model_pred_ind
            active_preds = np.delete(active_preds, best_cur_del_ind)

            if verbose:
                r_sqr_cur = models[pred_num + 1].r_sqr_stat
                r_sqr_prev = models[pred_num].r_sqr_stat
                print(
                    "Predictors: {} - Current R^2: {:.4f} (relative decrease of {:.2f}%)"
                    .format(predictors_num - pred_num - 1, r_sqr_cur,
                            100 * (1.0 - r_sqr_prev / r_sqr_cur)))

        return deleted_preds

    def _calc_model_errs(self,
                         chosen_preds: t.Optional[np.ndarray] = None,
                         deleted_preds: t.Optional[np.ndarray] = None,
                         k_fold_num: int = 10,
                         verbose: bool = False) -> np.ndarray:
        predictors_num = self.X.shape[1]
        model_test_errs = np.zeros(1 + predictors_num, dtype=float)
        model_test = MultipleLinRegressor(calc_stats=False)

        X_aug = model_test._augment_x(self.X)
        for model_ind in np.arange(1 + predictors_num):
            if chosen_preds is not None:
                cur_preds = chosen_preds[:model_ind]

            else:
                cur_preds = np.delete(
                    np.arange(predictors_num), deleted_preds[:model_ind])

            X_cur = X_aug[:, cur_preds]
            for inds_test, inds_train in cross_validation.kfold_cv(
                    X=X_cur, k=k_fold_num, return_inds=True):
                X_train, X_test = X_cur[inds_train, :], X_cur[inds_test, :]
                y_train, y_test = self.y[inds_train], self.y[inds_test]

                model_test.fit(X=X_train, y=y_train, add_intercept=False)

                preds = model_test.predict(X_test)
                model_test_errs[model_ind] += model_test.rmse(y_test, preds)

        if verbose:
            print(
                "{}-fold Cross validation mean test RMSE:".format(k_fold_num),
                model_test_errs / k_fold_num)

        return model_test_errs

    def _choose_best_model(self,
                           chosen_preds: t.Optional[np.ndarray] = None,
                           deleted_preds: t.Optional[np.ndarray] = None,
                           k_fold_num: int = 10,
                           verbose: bool = False) -> MultipleLinRegressor:
        if chosen_preds is None and deleted_preds is None:
            raise RuntimeError(
                "Both 'chosen_preds' and 'deleted_preds' are None.")

        model_test_errs = self._calc_model_errs(
            chosen_preds=chosen_preds,
            deleted_preds=deleted_preds,
            k_fold_num=k_fold_num,
            verbose=verbose)

        best_model_ind = np.argmin(model_test_errs)

        if chosen_preds is not None:
            best_model_preds = chosen_preds[:(1 + best_model_ind)]

        else:
            best_model_preds = np.delete(
                np.arange(self.X.shape[1]),
                deleted_preds[:(best_model_ind - 1)])

        if verbose:
            print(
                "Optimal (with {} selection strategy) number of predictors:".
                format("backward" if chosen_preds is None else "forward"),
                len(best_model_preds))
            print("Predictor indices:", best_model_preds)

        self.best_model = MultipleLinRegressor(calc_stats=True).fit(
            X=self.X[:, best_model_preds], y=self.y)

        return self.best_model

    def forward_selection(self, k_fold_num: int = 10,
                          verbose: bool = False) -> MultipleLinRegressor:
        """."""
        if self.X is None or self.y is None:
            raise TypeError(
                "Please 'fit' data into ModelSelection before model selection."
            )

        chosen_preds = self._forward_get_all_models(verbose=verbose)

        return self._choose_best_model(
            chosen_preds=chosen_preds, k_fold_num=k_fold_num, verbose=verbose)

    def backward_selection(self, k_fold_num: int = 10,
                           verbose: bool = False) -> MultipleLinRegressor:
        """Feature selection using the stepwise backward selection.

        In the stepwise backward feature selection strategy, a linear
        model is fitted using all independent variables. For each iteration,
        the least significant variable (based on the $R^{2}$ value) is
        removed from the model. The best model of each iteration is kept,
        and the final model is chosen based on the test error using k-fold
        cross validations.
        """
        if self.X is None or self.y is None:
            raise TypeError(
                "Please 'fit' data into ModelSelection before model selection."
            )

        deleted_preds = self._backward_get_all_models(verbose=verbose)

        return self._choose_best_model(
            deleted_preds=deleted_preds,
            k_fold_num=k_fold_num,
            verbose=verbose)


def _test_univar_lin_reg_01() -> None:
    # pylint: disable=R0914
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
            X_test, y_pred, "o", label="RMSE: {:.2f}".format(errors[fold_id]))
        plt.legend()
        plt.title(str(fold_id))

    total_error = LinRegressor.rmse(errors, np.zeros(num_folds))
    print("Total RMSE:", total_error)

    plt.show()


def _leverage_plot(model: t.Union[LinRegressor, MultipleLinRegressor],
                   y: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    studentized_res = model.residuals / model.std_err_residual
    plt.hlines(
        y=0, xmin=1 / y.size, xmax=np.max(model.leverage), linestyle="--")
    lev_threshold = 2 / y.size
    inds = model.leverage <= lev_threshold
    plt.scatter(
        model.leverage[inds],
        studentized_res[inds],
        marker="o",
        color="green",
        label="Leverage <=t")
    plt.scatter(
        model.leverage[~inds],
        studentized_res[~inds],
        marker="x",
        color="red",
        label="Leverage > t")
    plt.title(
        "Leverage x Studentized residuals (t = {:.8f})".format(lev_threshold))
    plt.xlabel("Leverage")
    plt.ylabel("Studentized residuals")
    plt.legend()
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
    print("R^2:", model.r_sqr_stat)
    print("Adjusted R^2:", model.r_sqr_adj_stat)
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

    _leverage_plot(model=model, y=y_boston)

    assert np.isclose(np.mean(model.leverage), 2 / y_boston.size)

    loocv_err = 0
    for ind_test, ind_train in cross_validation.loo_cv(
            X=X_boston, return_inds=True):
        cur_model = MultipleLinRegressor().fit(
            X=X_boston[ind_train], y=y_boston[ind_train])
        loocv_err += np.square(
            cur_model.predict(X_boston[ind_test]) - y_boston[ind_test])

    loocv_err /= y_boston.size

    assert np.isclose(model.loocv_err, loocv_err)


def _test_multi_lin_reg_01() -> None:
    import sklearn.datasets

    boston = sklearn.datasets.load_boston()

    X_boston = boston.data[:,
                           np.isin(
                               boston.feature_names, ["LSTAT", "TAX", "AGE"],
                               assume_unique=True)]
    y_boston = boston.target

    model = MultipleLinRegressor().fit(X=X_boston, y=y_boston)

    print("RSS:", model.residual_sum_sqr)
    print("RSE:", model.std_err_residual)
    print("R^2:", model.r_sqr_stat)
    print("Adjusted R^2:", model.r_sqr_adj_stat)
    print("Coeff SE:", model.std_err_coeffs)
    print("coeffs:", model.coeffs)
    print("t-stat", model.t_stat)
    print("t-stat p-value", model.t_test_pval)
    print("F-stat", model.f_stat)
    print("F-stat p-value", model.f_stat_pval)
    print("VIF", model.var_inflation_factor)

    _leverage_plot(model=model, y=y_boston)


def _test_model_selection() -> None:
    import sklearn.datasets

    boston = sklearn.datasets.load_boston()
    ModelSelection().fit(
        X=boston.data, y=boston.target).forward_selection(verbose=True)

    ModelSelection().fit(
        X=boston.data, y=boston.target).backward_selection(verbose=True)


if __name__ == "__main__":
    _test_univar_lin_reg_01()
    _test_univar_lin_reg_02()
    _test_multi_lin_reg_01()
    _test_model_selection()
