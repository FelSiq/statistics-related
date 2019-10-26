r"""Implementation of the log-sum-exp trick.

The goal is to calculate \log\sum_{i}e^{x_{i}} in a numeric
stable manner.
"""
import typing as t

import numpy as np


def log_sum_exp(vals: np.ndarray,
                shift_coeff: t.Optional[t.Union[int, float]] = None) -> float:
    r"""Calculate \log_{e}\sum_{i}e^{vals_{i}} in a numeric stable manner."""
    if shift_coeff is None:
        shift_coeff = np.max(vals)

    if np.isinf(shift_coeff):
        return np.inf

    return shift_coeff + np.log(np.sum(np.exp(vals - shift_coeff)))


def _test():
    import scipy.special

    np.random.seed(16)
    for _ in np.arange(1024):
        size = np.random.randint(2, 16384)
        vals = np.random.randint(
            -1e+16, 1e+16, size=size) + np.random.random(size=size)
        assert np.allclose(log_sum_exp(vals), scipy.special.logsumexp(vals))


if __name__ == "__main__":
    _test()
