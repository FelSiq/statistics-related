import numpy as np


def markov_bound(bound: np.number, expected: np.number) -> float:
    """Markov's bound for a Random Variable.

    Let R be a nonnegative Random Variable. Then, for any
    bound > 0 the Markov's bound is calculated as:

        P[R >= bound] <= Ex[R] / bound

    Where Ex[R] is the expected value (weighted mean) of R,
    and P[R >= bound] is the probability of R value be at least
    ``bound``.
    """
    if bound <= 0:
        raise ValueError("'bound' must be positive. Got '{}'.".format(bound))

    if expected < 0:
        raise ValueError("'expected' must be nonnegative. "
                         "Got '{}'.".format(expected))

    return expected / bound


def markov_bound_from_mean(deviations: np.number) -> float:
    """Prob. of a Random Variable has a factor of ``deviations`` from ``expected``.

    Let R be a nonnegative Random Variable. Then, for any
    deviations >= 1 the Markov's bound based on the mean is
    calculated as:

        P[R >= deviations * Ex[R]] <= 1 / deviations

    Where Ex[R] is the expected value (weighted mean) of R,
    and P[R >= x] is the probability of R value be at least
    ``x``, and ``deviations`` is some positive the number of
    deviations from the expected value.
    """
    if deviations <= 0:
        raise ValueError("'deviations' must be positive. "
                         "Got '{}'.".format(deviations))

    return 1.0 / deviations


def chebyshev_bound(bound: np.number, variance: np.number) -> float:
    """Chebyshev's Bound of a Random Variable deviates from expected value.

    Let R be any Random Variable, Ex[R] be its expected value (weighted
    mean), Var[R] be its variance, calculated as follows:

        Var[R] := Ex[(R - Ex[R]) ** 2.0]

    And let ``bound`` be some real positive value. Then, by Chebyshev's
    bound:

        P[|R - Ex[R]| >= bound] <= Var[R] / (bound ** 2.0)

    Where |.| is the absolute value operator and P[x] is the probability
    of x.
    """
    if bound <= 0:
        raise ValueError("'bound' must be positive. Got '{}'.".format(bound))

    if variance < 0:
        raise ValueError("'variance' must be nonnegative. "
                         "Got '{}'.".format(variance))

    return variance / (bound**2.0)


def chebyshev_bound_for_std(deviations: np.number,
                            two_sided: bool = True) -> float:
    """Chebyshev bound to ``deviations`` standard deviations from mean.

    Let R be any Random Variable, Ex[R] its expected value, and
    Sigma[R] its standard deviation calculated as follows:

        Sigma[R] = sqrt(Var[R]), where Var[R] = Ex[(R - Ex[R]) ** 2.0]

    Then, by the Chebyshev's bound:

        P[|R - Ex[R]| >= c * Sigma[R]] <= 1.0 / (c ** 2.0 + k)

    For any c > 0, where c is the number of standard deviations between
    R and Ex[R], and k = 0 if considering the deviation from both sides
    of the distribution, and k = 1 otherwise (one-sided deviation).
    """
    if deviations <= 0:
        raise ValueError("'deviations' must be positive. "
                         "Got '{}'.".format(deviations))

    if not isinstance(two_sided, bool):
        raise TypeError("'two_sided' must be a boolean value.")

    return 1.0 / (deviations**2.0 + int(two_sided))


def chernoff_bound(expected: np.number, deviations: np.number) -> float:
    """Chernoff's bound for sums of some mutually independent Random Variables.

    Let T be a function of Random Variables defined as:

        T = T_1 + T_2 + ... + T_n = sum_i(T_i)

    Where T_i are mutually independent random variables such that
    0 <= T_i <= 1 for all i. Then, for any ``deviations`` >= 1:

        P[T >= deviations * Ex[T]] <= exp(-k * Ex[T])

    Where Ex[T] is the expected value (weighted mean) of T, P[x] is
    the probability of ``x`` and k = (deviations * (ln(deviations) - 1) + 1),
    where ``ln`` is the natural logarithm.

    Hence, the probability of some function T of random variables
    mutually independent, all in the range [0, 1], be at least ``c``
    times its expected value is bounded by a exponential function
    of ``c`` and the expected value.
    """
    k = deviations * (np.log(deviations) - 1) + 1
    return np.exp(-k * expected)


def murphy_law(expected: np.number) -> float:
    """Probability of no event occur given the expected number of events to occur.

    The events must be mutually independent.
    """
    if expected < 1:
        raise ValueError("'expected' must be greater or equal than 1. "
                         "Got '{}'".format(expected))

    return np.exp(-expected)


if __name__ == "__main__":
    print("markov_bound:", markov_bound(bound=300, expected=100))
    print("markov_bound_from_mean:", markov_bound_from_mean(deviations=3))
    print("chebyshev_bound:", chebyshev_bound(bound=100, variance=10))
    print("chebyshev_bound_for_std (two sided):",
          chebyshev_bound_for_std(deviations=3, two_sided=True))
    print("chebyshev_bound_for_std (one sided):",
          chebyshev_bound_for_std(deviations=3, two_sided=False))
    print("chernoff_bound:", chernoff_bound(expected=500, deviations=1.2))
    print("murphy_lay:", murphy_law(3))
