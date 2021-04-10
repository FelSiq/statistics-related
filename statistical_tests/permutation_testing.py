"""Permutation testing experiments."""
import typing as t

import numpy as np


def _shuffle(pop_a: np.ndarray,
             pop_b: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:
    """Shuffle (not in-place) arrays ``pop_a`` and ``pop_b`` values."""
    aux = np.random.permutation(np.concatenate((pop_a, pop_b)))
    return aux[:pop_a.size], aux[pop_a.size:]


def perm_test(pop_a: np.ndarray,
              pop_b: np.ndarray,
              stat_test: t.Callable[[np.ndarray, np.ndarray], t.
                                    Union[float, int, np.number]],
              perm_num: int = 1000,
              random_seed: t.Optional[int] = None) -> float:
    """Tests whether the ``stat_test`` values of ``pop_a`` and ``pop_b`` match.

    The test works as following:
    H_0 (null hypothesis): the ``stat_test`` value of ``pop_a`` and ``pop_b`` are equal.
    H_1 (alt hypothesis): the ``stat_test`` of ``pop_a`` and ``pop_b`` are different.

    Arguments
    ---------
    stat_test : :obj:`Callable`
        Statistical test. Must be a function that receives two numpy
        arrays and return some numeric value.

    perm_num : :obj:`int`, optional
        Number of permutations.

    random_seed : :obj:`int`, optional
        If given, set the random seed before the first permutation.

    Returns
    -------
    :obj:`float`
        p-value of the permutation test.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if pop_a.ndim != 1:
        pop_a = pop_a.ravel()

    if pop_b.ndim != 1:
        pop_b = pop_b.ravel()

    truediff = stat_test(pop_a, pop_b)

    stat_test_vals = np.zeros(perm_num)
    for ind in np.arange(perm_num):
        sh_a, sh_b = _shuffle(pop_a, pop_b)
        stat_test_vals[ind] = stat_test(sh_a, sh_b)

    return (np.sum(truediff <= stat_test_vals) + 1) / (perm_num + 1)


def stratified_perm_test(pop_a_1: np.ndarray,
                         pop_b_1: np.ndarray,
                         pop_a_2: np.ndarray,
                         pop_b_2: np.ndarray,
                         stat_test: t.Callable[[np.ndarray, np.ndarray], t.
                                               Union[float, int, np.number]],
                         perm_num: int = 1000,
                         random_seed: t.Optional[int] = None) -> float:
    """Tests whether the stratified ``test_values`` values of ``pop_a`` and ``pop_b`` match.

    The stratification is made between groups ``1`` and ``2``.

    Arguments
    ---------
    stat_test : :obj:`Callable`
        Statistical test. Must be a function that receives two numpy
        arrays and return some numeric value.

    perm_num : :obj:`int`, optional
        Number of permutations.

    random_seed : :obj:`int`, optional
        If given, set the random seed before the first permutation.

    Returns
    -------
    :obj:`float`
        p-value of the permutation test.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if pop_a_1.ndim != 1:
        pop_a_1 = pop_a_1.ravel()

    if pop_b_1.ndim != 1:
        pop_b_1 = pop_b_1.ravel()

    if pop_a_2.ndim != 2:
        pop_a_2 = pop_a_2.ravel()

    if pop_b_2.ndim != 2:
        pop_b_2 = pop_b_2.ravel()

    truediff = stat_test(
        np.concatenate((pop_a_1, pop_a_2)), np.concatenate((pop_b_1, pop_b_2)))

    stat_test_vals = np.zeros(perm_num)
    for ind in np.arange(perm_num):
        sh_a_1, sh_b_1 = _shuffle(pop_a_1, pop_b_1)
        sh_a_2, sh_b_2 = _shuffle(pop_a_2, pop_b_2)
        stat_test_vals[ind] = stat_test(
            np.concatenate((sh_a_1, sh_a_2)), np.concatenate((sh_b_1, sh_b_2)))

    return (np.sum(truediff <= stat_test_vals) + 1) / (perm_num + 1)


def _experiment_01():
    """Permutation testing experiment 01."""

    def mean_test(pop_a: np.ndarray, pop_b: np.ndarray) -> float:
        return np.abs(pop_a.mean() - pop_b.mean())

    random_seed = 16
    np.random.seed(random_seed)

    pop_a = np.random.normal(size=500)
    pop_b = np.random.normal(size=500) + 0.5
    print("Mean test")
    print("pop_a.mean = {}\npop_b.mean = {}".format(pop_a.mean(),
                                                    pop_b.mean()))

    p_val = perm_test(
        pop_a,
        pop_b,
        stat_test=mean_test,
        perm_num=1000,
        random_seed=random_seed)
    print("permutation test p-value: {}".format(p_val))


if __name__ == "__main__":
    _experiment_01()
