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

    The strategy adopted is numeric stable.

    Arguments
    ---------
    x : :obj:`np.ndarray`
        Point to estimate the log-pdf of the multivariate normal distribution.

    mean_vec : :obj:`np.ndarray`, optional
        Mean vector of the normal distribution. Size must match with ``x``.
        If not given (:obj:`NoneType`), then the zero-mean vector will be
        used by default.

    cov_mat : :obj:`np.ndarray`, optional
        Covariance matrix of the normal distribution. Dimensions must match
        with both ``x`` and ``mean_vec``. If not given (:obj:`NoneType`), then
        the identity matrix $I_{mean_vec.size}$ will be used by default. This
        argument is used only if ``cov_mat_cholesky`` is :obj:`NoneType`, and
        is ignored otherwise.

    cov_mat_cholesky : :obj:`np.ndarray`, optional
        Lower triangle of the Cholesky Decomposition factor matrix of the
        covariance matrix. If not given, then this cholesky factor will be
        calculated inside this function.

    Returns
    -------
    :obj:`float`
        Log-density (natural logarithm of the probability density function) of the
        defined Multivariate Normal distribution estimated at ``x``.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if mean_vec is None:
        mean_vec = np.zeros(x.size, dtype=np.float64)

    if cov_mat_cholesky is None:
        if cov_mat is None:
            cov_mat = np.diag(np.ones(mean_vec.size, dtype=np.float64))

        cov_mat_cholesky = scipy.linalg.cholesky(cov_mat, lower=True)

    num_dim = mean_vec.size

    _cov_shape = cov_mat_cholesky.shape
    if not x.size == num_dim == _cov_shape[0] == _cov_shape[1]:
        raise ValueError("Dimensions of given 'x', mean vector and covariance "
                         "matrix does not match! (Got length {} mean vector "
                         "and covariance matrix with {} shape and {} length 'x'."
                        .format(mean_vec.size, _cov_shape, x.size))

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
    num_it = 1000
    for it in np.arange(num_it):
        dim = np.random.randint(2, 32)

        mean_vec = np.random.randint(-64, 64, size=dim) + np.random.random(dim)

        cov_mat = gen_positive_definite_matrix(dim=dim)
        cov_mat[np.diag_indices(dim)] = 1.0
        cho_factor = scipy.linalg.cholesky(cov_mat, lower=True)

        _multivar_normal = scipy.stats.multivariate_normal(
            mean=mean_vec, cov=cov_mat)
        for _ in np.arange(50):
            x = np.random.randint(
                -10, 10, size=dim) + np.random.random(size=dim)
            res = multivariate_normal_logpf(
                x=x, mean_vec=mean_vec, cov_mat_cholesky=cho_factor)
            scipy_res = _multivar_normal.logpdf(x)
            assert np.allclose(res, scipy_res)

        print("\rTest 1 progress: {}%".format(100 * it / num_it), end="")

    print("\rRandom normals test done.")

    for it in np.arange(num_it):
        dim = np.random.randint(2, 32)

        _multivar_normal = scipy.stats.multivariate_normal(
            mean=np.zeros(dim), cov=np.diag(np.ones(dim)))
        for _ in np.arange(50):
            x = np.random.randint(
                -10, 10, size=dim) + np.random.random(size=dim)
            res = multivariate_normal_logpf(x=x)
            scipy_res = _multivar_normal.logpdf(x)
            assert np.allclose(res, scipy_res)

        print("\rTest 2 progress: {}%".format(100 * it / num_it), end="")

    print("\rDone.")


if __name__ == "__main__":
    _test()
