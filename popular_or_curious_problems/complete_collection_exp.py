"""
The expected number of items to complete a collection of n
distinct elements, in a random uniform and independent
gathering process, is n * H_n, where H_n is the nth harmonic
number.
"""
import numpy as np


def harmonic_number(n):
    return np.sum([1 / (1 + i) for i in np.arange(n)])


def general_experiment(n: int, name: str):
    expected_value = n * harmonic_number(n)
    counter = 0

    bdays = set()
    while len(bdays) < n:
        bdays.update({np.random.randint(1, n + 1)})
        counter += 1

    print("Gathering all {}".format(name))
    print("Steps to complete collection :", counter)
    print("Expected number of steps     :", expected_value)
    print("Difference                   :", abs(counter - expected_value))
    print()


general_experiment(365, "birthdays")
general_experiment(6, "dice values")
general_experiment(2, "coin faces (or getting a baby girl and baby boy)")
