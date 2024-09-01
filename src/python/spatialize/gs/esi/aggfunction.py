import numpy as np
import scipy as sci

from spatialize._math_util import BilateralFilteringFusion


def mean(values):
    return np.nanmean(values, axis=1)


def median(values):
    return np.nanmedian(values, axis=1)


def MAP(values):
    return sci.stats.mode(values, axis=1, keepdims=True, nan_policy="omit").mode


class Percentile:
    def __init__(self, q=75):
        self.q = q

    def __call__(self, values):
        return np.nanpercentile(values, self.q, axis=1)

    def __repr__(self):
        return f"percentile({self.q})"


class WeightedAverage:
    def __init__(self, normalize=False, weights=None, force_resample=True):
        self.normalize = normalize
        self.weights = weights
        self.force_resample = force_resample

    def __call__(self, values):
        s = values.shape[1]
        if self.weights is None or self.force_resample:
            rng = np.random.default_rng()
            self.weights = rng.dirichlet([1] * s)
        m_values = np.ma.array(values, mask=np.isnan(values))
        estimation = np.ma.getdata(np.ma.average(m_values, axis=1, weights=self.weights))
        if self.normalize:
            zscore_estimation = (estimation - np.mean(estimation)) / np.std(estimation)
            return zscore_estimation * np.nanstd(values) + np.nanmean(values)
        else:
            return estimation


def identity(values):
    return values


# Bilateral filter
def bilateral_filter(values):

    bff = BilateralFilteringFusion(cube=values)
    fusion = bff.eval()
    two_dims_fusion = np.flip(fusion.reshape(fusion.shape[0], fusion.shape[1]), 1)

    return two_dims_fusion
