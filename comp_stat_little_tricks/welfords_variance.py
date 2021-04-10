import typing as t


def welfords_var(vals: t.Sequence[t.Union[int, float]]) -> float:
    """Powerful one-pass method for computing array variance."""
    M, S = 0, 0

    for k, x in enumerate(vals, 1):
        oldM = M
        M += (x - M) / k
        S += (x - M) * (x - oldM)

    return S / (len(vals) - 1)


if __name__ == "__main__":
    import numpy as np
    np.random.seed(1444)

    for i in range(500):
        vals = (np.random.randint(-999999, 999999, size=1000) +
                2.0 * np.random.random(size=1000) - 1.0)

        var_wf = welfords_var(vals)
        var_np = vals.var(ddof=1)

        assert np.allclose(var_wf, var_np)

    for i in range(500):
        vals = 2.0 * np.random.random(size=1000) - 1.0

        var_wf = welfords_var(vals)
        var_np = vals.var(ddof=1)

        assert np.allclose(var_wf, var_np)
