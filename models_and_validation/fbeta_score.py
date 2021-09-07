import numpy as np


def _build_confusion_matrix(y_true, y_preds):
    cls, inds_all = np.unique(np.concatenate((y_true, y_preds)), return_inverse=True)
    n = len(y_true)
    inds_true, inds_preds = inds_all[:n], inds_all[n:]
    confusion_mat = np.zeros((cls.size, cls.size), dtype=np.uint)
    np.add.at(confusion_mat, (inds_true, inds_preds), 1)
    return confusion_mat


def _fb(precision, recall, beta=1.0):
    b_sqr = beta ** 2
    return (1.0 + b_sqr) * precision * recall / (1e-8 + b_sqr * precision + recall)


def macro_precision(confusion_matrix):
    # binary precision = TP / (TP + FP)
    true_positives = np.diag(confusion_matrix)
    per_class_precision = true_positives / (1e-8 + np.sum(confusion_matrix, axis=0))
    return float(np.mean(per_class_precision))


def macro_recall(confusion_matrix):
    # binary recall = TP / (TP + FN)
    true_positives = np.diag(confusion_matrix)
    per_class_recall = true_positives / (1e-8 + np.sum(confusion_matrix, axis=1))
    return float(np.mean(per_class_recall))


def macro_fb(y_true, y_preds, beta=1.0):
    confusion_mat = _build_confusion_matrix(y_true, y_preds)
    macp = macro_precision(confusion_mat)
    macr = macro_recall(confusion_mat)
    res = _fb(macp, macr, beta)
    return res


def micro_fb(y_true, y_preds, beta=1.0):
    # Micro Fb = Accuracy
    return np.mean(np.asarray(y_true) == np.asarray(y_preds))


def _test():
    import sklearn.metrics

    np.random.seed(16)

    n = 1000

    for _ in np.arange(n):
        n_cls = np.random.randint(2, 6)
        beta = 5 * np.random.random()
        size = np.random.randint(100, 1000)
        print(n_cls, beta, size)

        a, b = np.random.randint(0, n_cls, size=(2, size))

        micro_score = micro_fb(a, b, beta=beta)
        micro_score_ref = sklearn.metrics.fbeta_score(a, b, beta=beta, average="micro")
        assert np.isclose(micro_score, micro_score_ref), (micro_score, micro_score_ref)

        macro_score = macro_fb(a, b, beta=beta)
        macro_score_ref = sklearn.metrics.fbeta_score(a, b, beta=beta, average="macro")
        assert np.isclose(macro_score, macro_score_ref, rtol=0.10), (macro_score, macro_score_ref)


if __name__ == "__main__":
    _test()
