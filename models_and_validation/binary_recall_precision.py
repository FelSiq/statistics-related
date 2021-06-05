import numpy as np
import sklearn.metrics


def recall(y_true, y_pred):
    return float(np.dot(y_true, y_pred) / np.sum(y_true))


def precision(y_true, y_pred):
    return float(np.dot(y_true, y_pred) / np.sum(y_pred))


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r)


def _test():
    np.random.seed(16)
    n = 300

    for _ in np.arange(n):
        a, b = np.random.randint(2, size=(2, n))
        rec = recall(a, b)
        rec_ref = sklearn.metrics.recall_score(a, b)

        prec = precision(a, b)
        prec_ref = sklearn.metrics.precision_score(a, b)

        f1 = f1_score(a, b)
        f1_ref = sklearn.metrics.f1_score(a, b)

        assert np.isclose(rec, rec_ref)
        assert np.isclose(prec, prec_ref)
        assert np.isclose(f1, f1_ref)


if __name__ == "__main__":
    _test()
