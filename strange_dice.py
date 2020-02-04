"""Counter-intuitive collection of dice for tournament.

There are 3 different dice: A, B, and C.

Die A has faces 2, 6 and 7.
Die B has faces 1, 5 and 9.
Die C has faces 3, 4 and 8.

If each die is thrown once, and we define "N > M" as the
die "N" got a higher value than die "M", it can be shown that
P[A > B] > P[B > C] > P[C > A].

In other words, those dice are a counter-example of the
transitive property of a tournament using an probabilistic
approach.

More than that, if we throw the dice twice, and sum the
scores of each die, and compare to the sum of all other
dice, it can be shown that:

P[A_2 > B_2] < P[B_2 > C_2] < P[C_2 > A_2]

In other words, the tournament result is reversed when we
throw each die twice, and sum their scores as the new die
score.

Other weird behaviours happens if we throw the dice more
than 2 times. This code simulates this experiment for any
number of throws for each die, and compares the result in
a tournament-like structure.
"""
import typing as t

import numpy as np

DICE_VALS = np.array([
    [2, 6, 7],
    [1, 5, 9],
    [3, 4, 8],
])
"""Each row represents a different die."""


def _verbose(res: np.ndarray, num_rolls: int, num_reps: int) -> None:
    """Print relevant information about the results."""
    print("Results: (rolling dice {} times, with {} repetitions)".format(
        num_rolls, num_reps))
    print("Die A beats B {:.4f} of times.".format(res[0]))
    print("Die B beats C {:.4f} of times.".format(res[1]))
    print("Die C beats A {:.4f} of times.".format(res[2]))


def roll_dice(num_rolls: int = 1,
              num_reps: int = 8192,
              random_state: t.Optional[int] = None,
              verbose: bool = True) -> np.ndarray:
    """Monte-carlo approach to the dice experiment.

    Arguments
    ---------
    num_rolls : :obj:`int`, optional
        Number of throws for each dice. Let `N > M` means that the die `N`
        has a higher probability than having a higher `final score` han die
        `M`, and defining the `final score` as the sum of all individual
        throws of the same die (i.e., the `final score` of a die `N` is the
        sum of the scores of all individual throws of die `N`.) Then:
            If ``num_rols`` = 1, then A > B > C > A.
            If ``num_rols`` = 2, then A < B < C < A.
            Other relationships between A, B and C appear when ``num_rolls``
            assumes different values.

    num_reps : :obj:`int`, optional
        Number of experiment repetitions.

    random_state : :obj:`int`, optional
        If not None, set the random seed before the experiments to keep
        the results reproducible.

    verbose : :obj:`bool`, optional
        If True, print messages regarding the experiment results.

    Returns
    -------
    :obj:`np.ndarray`
        A numpy array with the average values (of every experiment repetition)
        of 3 probabilities, P[A > B], P[B > C], and P[C > A], in this order.
    """
    if random_state is not None:
        np.random.seed(random_state)

    res = np.zeros(3, dtype=float)
    for _ in np.arange(num_reps):
        die_a, die_b, die_c = np.sum(
            np.apply_along_axis(
                arr=DICE_VALS,
                axis=1,
                func1d=np.random.choice,
                size=num_rolls,
                replace=True),
            axis=1)

        res += die_a > die_b, die_b > die_c, die_c > die_a

    res /= num_reps

    if verbose:
        _verbose(res, num_rolls, num_reps)

    return res


def _test() -> None:
    num_reps = 10
    for rolls in np.arange(1, 11):
        res = np.zeros(3, dtype=float)

        for _ in np.arange(num_reps):
            res += roll_dice(num_rolls=rolls)

        res /= num_reps
        _verbose(res, rolls, 8192 * num_reps)


if __name__ == "__main__":
    _test()
