import numpy as np
import scipy as sci


def mean(values):
    return np.nanmean(values, axis=1)


def median(values):
    return np.nanmedian(values, axis=1)


def MAP(values):
    return sci.stats.mode(values, axis=1, keepdims=True, nan_policy="omit").mode


class WeightedAverage:
    def __init__(self, normalize=False, weights=None, force_resample=True):
        self.normalize = normalize
        self.weights = weights
        self.force_resample = force_resample

    def __call__(self, values):
        s = values.shape[1]
        rng = np.random.default_rng()
        if self.weights is None or self.force_resample:
            self.weights = rng.dirichlet([1] * s)
        m_values = np.ma.array(values, mask=np.isnan(values))
        return np.ma.average(m_values, axis=1, weights=self.weights)


class Percentile:
    def __init__(self, q=75):
        self.q = q

    def __call__(self, values):
        return np.nanpercentile(values, self.q, axis=1)

    def __repr__(self):
        return f"percentile({self.q})"
