"""Calculus of AIC and BIC following the R implementation."""
import typing as t

import numpy as np
import scipy.stats


def aic(residuals: np.ndarray, num_params: int, num_obs: int,
        k: int = 2) -> float:
    r"""Akaike Information Criterion (AIC) estimate.

    Information criteria used to penalize the number of parameters in a model.

    Arguments
    ---------
    residuals : :obj:`np.ndarray`, shape = [num_obs,]
        Residuals $\hat{e}_{i}$ are calculated by the true value of a dependent
        variable used to fit the model $y_{i}$ minus its predicted value $\hat{y}_{i}$
        by the model. In other words, the residual is the difference of some
        instance label from the train set and the predicted value by the model.
        $$
            \hat{e}_{i} = y_{i} - \hat{y}_{i}
        $$

    num_params : :obj:`int`
        Number of parameters in the model. If it is a simple linear regression model,
        then $\text{num_params} = 2$ (the independent variable + the intercept). In
        the Multiple Linear Regression case, $\text{num_params} = \text{intercept} +
        \text{number of independent variables in the train data}$.

        In fact, the ``number of parameters`` must count also an additional parameter
        $\sigma^{2}$, which is the variance of the standard errors of the model.
        This parameter is automatically accounted for in this function, so it does
        not need to be considered in ``num_params`` argument.

    num_obs : :obj:`int`
        Number of instances in the dataset.

    k : :obj:`int`, optional
        Weight for the number of parameters. Note that the AIC is calculated with
        $k=2$, while the BIC is calculated with $k=\log(n)$, where $\log$ is the
        natural logarithm.

    Return
    ------
    :obj:`float`
        AIC (if k = 2), BIF (if k = ln(num_obs)).
    """
    sqr_std_err = np.sum(np.square(residuals)) / num_obs
    aic_val = k * (1 + num_params) + num_obs * (
        1 + np.log(2 * np.pi) + np.log(sqr_std_err))
    return aic_val


def bic(residuals: np.ndarray, num_params: int, num_obs: int) -> float:
    """Calculate the Bayesian Information Criterion (BIC).

    Also known as ``Schwarz criterion``.
    """
    return aic(
        residuals=residuals,
        num_params=num_params,
        num_obs=num_obs,
        k=np.log(num_obs))


def likelihood_test_ratio(
        val_1: float,
        val_2: float,
        num_param_1: int,
        num_param_2: int,
        k: int = 2,
        return_pval: bool = True) -> t.Union[float, t.Tuple[float, float]]:
    r"""Tests whether model M_{1} is better than M_{2}, when M_{1} is a submodel of M_{2}.

    A model M_{1} being a submodel of M_{2} means that the model M_{1} is fitted
    using a subset of parameters of M_{2}. In this sense, M_{2} is a more complex
    model than M_{1}.

    Arguments
    ---------
    val_1 : :obj:`float`
        AIC or BIC of model M_{1} (the simpler one.)

    val_2 : :obj:`float`
        The same as ``val_1``, but for model M_{2} (the complex one.) Note
        that both ``val_1`` and ``val_2`` must be or AIC or BIC.

    num_param_1 : :obj:`int`
        Number of parameters in the simpler model M_{1}.

    num_param_2 : :obj:`int`
        Number of parameters in the more complex model M_{2}.

    k : :obj:`int`, optional
        Weight of the parameters. Must be $k = 2$ if ``val_1`` and ``val_2``
        are AICs, and must be $k = log(n)$, where $n$ is the number of
        observations used to train the models, if ``val_1`` and ``val_2``
        are BICs.

    return_pval : :obj:`bool`, optional
        If true, return the P-value calculated from a Chi-Squared distribution
        with $\text{num_param_2} - \text{num_param_1}$ degrees of freedom, of
        the likelihood test ratio value. Note that this P-value is only valid
        under the test assumptions, i.e., the M_{1} model is a submodel of the
        M_{2} model.

    Returns
    -------
    :obj:`float`
        Likelihood test ratio value.
    """
    diff_param_num = num_param_2 - num_param_1
    ltr = val_1 - val_2 + k * diff_param_num

    if return_pval:
        pval = scipy.stats.chi2.sf(x=ltr, df=diff_param_num)
        return ltr, pval

    return ltr


def _test_01() -> None:
    # pylint: disable=E1101
    import sklearn.datasets
    import linear_regression
    boston = sklearn.datasets.load_boston()

    model = linear_regression.MultipleLinRegressor().fit(
        X=boston.data, y=boston.target)

    aic_val = aic(
        residuals=model.residuals,
        num_params=model.coeffs.size,
        num_obs=boston.target.size,
        k=2)

    bic_val = bic(
        residuals=model.residuals,
        num_params=model.coeffs.size,
        num_obs=boston.target.size)

    print(aic_val)
    print(bic_val)


def _test_02() -> None:
    # pylint: disable=E1101
    import sklearn.datasets
    import linear_regression
    boston = sklearn.datasets.load_boston()

    num_obs = boston.target.size
    num_dep_vars = boston.data.shape[1]

    model_1 = linear_regression.MultipleLinRegressor().fit(
        X=boston.data, y=boston.target)

    aic_1 = aic(model_1.residuals, model_1.coeffs.size, num_obs)
    bic_1 = bic(model_1.residuals, model_1.coeffs.size, num_obs)

    ind_age = np.where(boston.feature_names == "AGE")[0]
    model_2 = linear_regression.MultipleLinRegressor().fit(
        X=boston.data[:, np.delete(np.arange(num_dep_vars), ind_age)],
        y=boston.target)

    aic_2 = aic(model_2.residuals, model_2.coeffs.size, num_obs)
    bic_2 = bic(model_2.residuals, model_2.coeffs.size, num_obs)

    ind_rm = np.where(boston.feature_names == "RM")[0]
    model_3 = linear_regression.MultipleLinRegressor().fit(
        X=boston.data[:, np.delete(np.arange(num_dep_vars), ind_rm)],
        y=boston.target)

    aic_3 = aic(model_3.residuals, model_3.coeffs.size, num_obs)
    bic_3 = bic(model_3.residuals, model_3.coeffs.size, num_obs)

    print(likelihood_test_ratio(aic_2, aic_1, num_dep_vars, num_dep_vars + 1))
    print(
        likelihood_test_ratio(
            bic_2, bic_1, num_dep_vars, num_dep_vars + 1, k=np.log(num_obs)))

    print(likelihood_test_ratio(aic_3, aic_1, num_dep_vars, num_dep_vars + 1))
    print(
        likelihood_test_ratio(
            bic_3, bic_1, num_dep_vars, num_dep_vars + 1, k=np.log(num_obs)))


if __name__ == "__main__":
    _test_01()
    _test_02()
