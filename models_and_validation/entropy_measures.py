import collections
import typing as t

import numpy as np


class _BaseEntropy:
    def __init__(self, eps: float = 1e-12):
        eps = float(eps)

        assert eps > 0.0

        self.eps = eps
        self._cache: t.Dict[str, float] = {}

    @staticmethod
    def cached_property(method):
        def wrapper(self):
            res = self._cache.get(method)

            if res is not None:
                return res

            self._cache[method] = res = method(self)
            return res

        return property(wrapper)

    def cross_entropy(self, p, q=None):
        q = q if q is not None else p

        p = np.asfarray(p)
        q = np.asfarray(q)

        return -float(np.dot(p, np.log(self.eps + q)))


class EntropyMetrics(_BaseEntropy):
    def __init__(self, eps: float = 1e-12):
        super(EntropyMetrics, self).__init__(eps=eps)
        self.x = np.empty(0, dtype=int)
        self.cls = np.empty(0, dtype=int)
        self.p = np.empty(0, dtype=float)

    def fit(self, x):
        x = np.asarray(x, dtype=int)

        assert x.size

        self.x = x
        self.n_cls = 1 + int(np.max(self.x))
        self.p = np.zeros(self.n_cls, dtype=float)
        np.add.at(self.p, self.x, 1.0 / self.x.size)

        self._cache = {}

        return self

    @_BaseEntropy.cached_property
    def entropy(self):
        return super(EntropyMetrics, self).cross_entropy(self.p)


class JointEntropyMetrics(_BaseEntropy):
    def __init__(self, eps: float = 1e-12):
        super(JointEntropyMetrics, self).__init__(eps=eps)

        self._entropy_x = EntropyMetrics(eps=eps)
        self._entropy_y = EntropyMetrics(eps=eps)

        self.joint_p: t.Dict[t.Tuple[int, int], float] = {}

    def fit(self, x, y):
        self._entropy_x.fit(x)
        self._entropy_y.fit(y)

        self.joint_p = collections.Counter(zip(self.x, self.y))
        self.joint_p = {k: v / self.x.size for k, v in self.joint_p.items()}

        self._cache = {}

        return self

    @property
    def entropy_x(self):
        return self._entropy_x.entropy

    @property
    def entropy_y(self):
        return self._entropy_y.entropy

    @property
    def cls_x(self):
        return self._entropy_x.cls

    @property
    def cls_y(self):
        return self._entropy_y.cls

    @property
    def x(self):
        return self._entropy_x.x

    @property
    def y(self):
        return self._entropy_y.x

    @property
    def p_x(self):
        return self._entropy_x.p

    @property
    def p_y(self):
        return self._entropy_y.p

    @_BaseEntropy.cached_property
    def cross_entropy(self):
        return -float(np.dot(self.p_x, np.log(self.eps + self.p_y)))

    @_BaseEntropy.cached_property
    def joint_entropy(self):
        pj = list(self.joint_p.values())
        return super(JointEntropyMetrics, self).cross_entropy(pj)

    @_BaseEntropy.cached_property
    def kl_div(self):
        """Kullback-Leibler divergence."""
        res = self.cross_entropy - self.entropy_x

        # This is equivalent to:
        aux = -float(
            np.dot(self.p_x, np.log(self.eps + self.p_y / (self.eps + self.p_x)))
        )
        assert np.isclose(aux, res), (aux, res)

        return res

    @_BaseEntropy.cached_property
    def jensen_shannon_div(self):
        p_avg = 0.5 * (self.p_x + self.p_y)

        f_ce = super(JointEntropyMetrics, self).cross_entropy

        kl_div_x_avg = f_ce(self.p_x, p_avg) - self.entropy_x
        kl_div_y_avg = f_ce(self.p_y, p_avg) - self.entropy_y

        js_div = 0.5 * (kl_div_x_avg + kl_div_y_avg)

        return js_div

    @_BaseEntropy.cached_property
    def jensen_shannon_dist(self):
        return self.jensen_shannon_div ** 0.5

    @_BaseEntropy.cached_property
    def mutual_info(self):
        return self.entropy_x + self.entropy_y - self.joint_entropy

    @_BaseEntropy.cached_property
    def cond_entropy_y_given_x(self):
        # H(X|Y)
        pj_norm = np.asfarray(
            [pj / (self.eps + self.p_x[i_x]) for (i_x, _), pj in self.joint_p.items()]
        )
        pj = list(self.joint_p.values())
        return super(JointEntropyMetrics, self).cross_entropy(pj, pj_norm)

    @_BaseEntropy.cached_property
    def cond_entropy_x_given_y(self):
        # H(X|Y)
        pj_norm = np.asfarray(
            [pj / (self.eps + self.p_y[i_y]) for (_, i_y), pj in self.joint_p.items()]
        )
        pj = list(self.joint_p.values())
        return super(JointEntropyMetrics, self).cross_entropy(pj, pj_norm)

    @_BaseEntropy.cached_property
    def homogeneity(self):
        return 1.0 - self.cond_entropy_x_given_y / self.entropy_x

    @_BaseEntropy.cached_property
    def completeness(self):
        return 1.0 - self.cond_entropy_y_given_x / self.entropy_y

    @_BaseEntropy.cached_property
    def v_measure(self):
        """
        Harmonic mean between completeness and homogeneity.

        Completeness: Instances with the same true label are grouped together as much as possible.
        Homogeneity: Groupings build must have the least amount of distinct labels as possible.
        """
        vc = self.completeness
        vh = self.homogeneity
        res = 2.0 * vc * vh / (self.eps + vc + vh)

        # This is equivalent to (NMI):
        aux = 2.0 * self.mutual_info / (self.eps + self.entropy_x + self.entropy_y)
        assert np.isclose(res, aux), (res, aux)

        return res

    @property
    def normalized_mutual_info(self):
        return self.v_measure


def _test():
    import scipy.stats
    import scipy.spatial
    import sklearn.metrics

    np.random.seed(32)
    n = 32
    x, y = np.random.randint(4, size=(2, n))

    model = JointEntropyMetrics().fit(x, y)

    print(model.entropy_x, scipy.stats.entropy(model.p_x))
    print(model.entropy_y, scipy.stats.entropy(model.p_y))
    print(model.kl_div, scipy.stats.entropy(model.p_x, model.p_y))
    print(model.mutual_info, sklearn.metrics.mutual_info_score(x, y))
    print(
        model.homogeneity,
        sklearn.metrics.homogeneity_score(x, y),
        sklearn.metrics.completeness_score(y, x),
    )
    print(
        model.completeness,
        sklearn.metrics.completeness_score(x, y),
        sklearn.metrics.homogeneity_score(y, x),
    )
    print(
        model.v_measure,
        sklearn.metrics.v_measure_score(x, y),
        sklearn.metrics.v_measure_score(y, x),
    )
    print(
        model.normalized_mutual_info,
        sklearn.metrics.normalized_mutual_info_score(x, y),
        sklearn.metrics.normalized_mutual_info_score(y, x),
    )
    print(
        model.jensen_shannon_dist,
        scipy.spatial.distance.jensenshannon(model.p_x, model.p_y),
        scipy.spatial.distance.jensenshannon(model.p_y, model.p_x),
    )


if __name__ == "__main__":
    _test()
