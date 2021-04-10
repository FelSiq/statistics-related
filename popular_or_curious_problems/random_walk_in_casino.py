import matplotlib.pyplot as plt
import numpy as np


def random_walking(initial, upper=-1, lower=0, p=0.5):
    if upper < lower:
        upper = np.inf
    val = initial
    vals = []
    while lower < val < upper:
        vals.append(val)
        val += np.random.choice((-1, 1), p=(1.0 - p, p))
    vals.append(val)
    return np.array(vals)


def gamblers_ruin(x, p=0.5):
    return x * (1 - 2 * p)


def plot(ans, p=0.5):
    plt.figure()
    plt.plot(np.arange(ans.size), ans)
    plt.axhline(
        y=start, xmin=0, xmax=ans.size, linestyle="dotted", color="red")
    plt.axhline(
        y=lower, xmin=0, xmax=ans.size, linestyle="dashed", color="purple")
    plt.axhline(
        y=upper, xmin=0, xmax=ans.size, linestyle="dashed", color="purple")
    if p != 0.5:
        plt.plot(
            np.arange(ans.size),
            ans[0] - np.fromfunction(gamblers_ruin, [ans.size], p=p),
            linestyle="dashed",
            color="orange")
    plt.show()


start = 1000
lower = 0
upper = start * 1.1
p = 0.499

ans = random_walking(start, upper=upper, lower=lower, p=p)
plot(ans, p=p)
