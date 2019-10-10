"""Estimate the log-density function of a Multivariate Normal distribution."""
import typing as t

import numpy as np
import scipy.linalg


def multivariate_normal_logpf(
        x: np.ndarray,
        mean_vec: t.Optional[np.ndarray] = None,
        cov_mat: t.Optional[np.ndarray] = None,
        cov_mat_cholesky: t.Optional[np.ndarray] = None) -> float:
    """Estimate the log-density function of a Multivariate Normal distribution.

    Arguments
    ---------
    x : :obj:`np.ndarray`

    mean_vec : :obj:`np.ndarray`, optional

    cov_mat : :obj:`np.ndarray`, optional

    cov_mat_cholesky : :obj:`np.ndarray, optional

    Returns
    -------
    :obj:`float`
        Log-density (natural logarithm of the probability density function) of the
        defined Multivariate Normal distribution estimated at ``x``.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if mean_vec is None:
        mean_vec = np.zeros(x.size)

    if cov_mat_cholesky is None:
        if cov_mat is None:
            cov_mat = np.diag(np.ones(mean_vec.size))

        cov_mat_cholesky = scipy.linalg.cholesky(cov_mat, lower=True)

    num_dim = mean_vec.size

    _cov_shape = cov_mat_cholesky.shape
    if not num_dim == _cov_shape[0] == _cov_shape[1]:
        raise ValueError("Dimensions of given mean vector and covariance "
                         "matrix does not match! (Got length {} mean vector "
                         "and covariance matrix with {} shape.".format(
                             mean_vec.size, _cov_shape))

    log_det = 2.0 * np.sum(np.log(np.diag(cov_mat_cholesky)))

    z_vector = scipy.linalg.solve_triangular(
        cov_mat_cholesky, x - mean_vec, lower=True)
    aux = np.dot(z_vector, z_vector)

    return -0.5 * (num_dim * np.log(2.0 * np.pi) + log_det + aux)


def _test():
    import scipy.stats
    def gen_positive_definite_matrix(dim: int) -> np.ndarray:
        """Generate a random positive definite matrix with dimension ``dim``."""
        pdmatrix = np.random.random(size=(dim, dim))
        pdmatrix = np.dot(pdmatrix, pdmatrix.T)
        pdmatrix /= pdmatrix.max()
        return pdmatrix

    np.random.seed(32)
    for it in np.arange(100):
        dim = np.random.randint(2, 32)

        mean_vec = np.random.randint(-64, 64, size=dim) + np.random.random(dim)

        cov_mat = gen_positive_definite_matrix(dim=dim)
        cov_mat[np.diag_indices(dim)] = 1.0
        cho_factor = scipy.linalg.cholesky(cov_mat, lower=True)

        for _ in np.arange(1000):
            x = np.random.randint(-10, 10, size=dim) + np.random.random(size=dim)
            res = multivariate_normal_logpf(x=x, mean_vec=mean_vec, cov_mat_cholesky=cho_factor)
            scipy_res = scipy.stats.multivariate_normal(mean=mean_vec, cov=cov_mat).logpdf(x)
            assert np.allclose(res, scipy_res)

        print("\rTest progress: {}%".format(it), end="")

    print("Done.")


if __name__ == "__main__":
    _test()
