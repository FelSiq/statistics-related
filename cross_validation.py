"""Tests with cross-validation.

General and simple method used for estimating unknown parameters
from data.

General algorithm:
    1. Randomly partition the data X of size n into X_{train} and X_{test}
        Let m = X_{test}.size
        Therefore, X_{train}.size = n - m
    2. Fit the model using X_{train}
    3. Test the fitted model using X_{test}
    4. Repeat t times and average the results

Some of the most known Cross-validation procedures:

k-fold CV: partition the data X into k (approximately) equal-sized
    subsets. t = k and m = n/k (tests of every subset once.)

Leave-one-out (LOO) CV: m = 1, t = n, testing on every sample once.
    (The same as K-fold CV with k = n).

Monte Carlo CV: randomly sample subsets of suitable size for the
    desired number of times.
"""
import typing as t

import numpy as np


def kfold_cv(
        X: np.ndarray,
        k: int = 10,
        shuffle: bool = True,
        return_inds: bool = False,
        random_state: t.Optional[int] = None,
) -> t.Iterator[t.Tuple[np.ndarray, np.ndarray]]:
    """K-fold Cross Validation."""
    if not isinstance(k, (int, np.int, np.int32, np.int64)):
        raise TypeError("'k' must be an integer (got {}.)".format(type(k)))

    if k <= 1:
        raise ValueError("'k' must be a greater than 1 (got {}.)".format(k))

    n_samples = X.size if X.ndim == 1 else X.shape[0]

    if n_samples < max(2, k):
        raise ValueError("Insufficient number of instances ({}). "
                         "Required num_inst >= max(2, k)".format(n_samples))

    test_size = int(n_samples / k)
    uneven_extra_inds = n_samples - k * test_size

    indices = np.arange(n_samples)

    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)

        np.random.shuffle(indices)

    for _ in np.arange(k):
        split_index = test_size + int(uneven_extra_inds > 0)
        uneven_extra_inds -= 1

        if return_inds:
            yield indices[:split_index], indices[split_index:]

        else:
            yield X[indices[:split_index]], X[indices[split_index:]]

        indices = np.roll(indices, -split_index)


def loo_cv(
        X: np.ndarray,
        shuffle: bool = True,
        return_inds: bool = False,
) -> t.Iterator[t.Tuple[np.ndarray, np.ndarray]]:
    """LOOCV (Leave-one-out Cross Validation).

    This is the same as n-fold Cross Validation (k = n).
    """
    n_samples = X.size if X.ndim == 1 else X.shape[0]

    for fold in kfold_cv(
            X=X, k=n_samples, shuffle=shuffle, return_inds=return_inds):
        yield fold


def monte_carlo_cv(X: np.ndarray,
                   test_frac: float = 0.2,
                   n: int = 10,
                   return_inds: bool = False,
                   random_state: t.Optional[int] = None
                   ) -> t.Iterator[t.Tuple[np.ndarray, np.ndarray]]:
    """Monte Carlo Cross Validation."""
    if not isinstance(test_frac, float):
        raise ValueError("'test_frac' must be float type (got {}.)".format(
            type(test_frac)))

    if not isinstance(n, int):
        raise TypeError("'n' must be an integer (got {}.)".format(type(n)))

    if n <= 0:
        raise ValueError("'n' must be a positive value (got {}.)".format(n))

    if not 0 < test_frac < 1:
        raise ValueError(
            "'test_frac' must be in (0.0, 1.0) interval (got {}.)".format(
                test_frac))

    n_samples = X.size if X.ndim == 1 else X.shape[0]

    if n_samples < 2:
        raise ValueError("Number of samples must be greater than 1 "
                         "(got {}.)".format(n_samples))

    test_size = int(test_frac * n_samples)

    if test_size == 0:
        raise ValueError(
            "Test subset with 0 instances. Please choose a higher 'test_frac' (got {}.)"
            .format(test_frac))

    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(n_samples)

    for _ in np.arange(n):
        np.random.shuffle(indices)
        inds_test, inds_train = np.split(indices, [test_size])

        if return_inds:
            yield inds_test, inds_train

        else:
            yield X[inds_test], X[inds_train]


if __name__ == "__main__":
    for fold in monte_carlo_cv(np.arange(2), test_frac=0.99, random_state=1):
        print(fold)
