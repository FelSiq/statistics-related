"""Calculus of AIC and BIC following the R implementation."""
import numpy as np


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
    aic = k * (1 + num_params) + num_obs * (
        1 + np.log(2 * np.pi) + np.log(sqr_std_err))
    return aic


def bic(residuals: np.ndarray, num_params: int, num_obs: int) -> float:
    """Calculate the Bayesian Information Criterion (BIC).

    Also known as ``Schwarz criterion``.
    """
    return aic(
        residuals=residuals,
        num_params=num_params,
        num_obs=num_obs,
        k=np.log(num_obs))


def _test() -> None:
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


if __name__ == "__main__":
    _test()
