import numpy as np
import scipy as sci

from spatialize._math_util import BilateralFilteringFusion


def mean(esi_samples):
    return np.nanmean(esi_samples, axis=1)


def median(esi_samples):
    return np.nanmedian(esi_samples, axis=1)


def MAP(esi_samples):
    return sci.stats.mode(esi_samples, axis=1, keepdims=True, nan_policy="omit").mode


class Percentile:
    def __init__(self, q=75):
        self.q = q

    def __call__(self, esi_samples):
        return np.nanpercentile(esi_samples, self.q, axis=1)

    def __repr__(self):
        return f"percentile({self.q})"


class WeightedAverage:
    def __init__(self, normalize=False, weights=None, force_resample=True):
        self.normalize = normalize
        self.weights = weights
        self.force_resample = force_resample

    def __call__(self, esi_samples):
        s = esi_samples.shape[1]
        if self.weights is None or self.force_resample:
            rng = np.random.default_rng()
            self.weights = rng.dirichlet([1] * s)
        m_esi_samples = np.ma.array(esi_samples, mask=np.isnan(esi_samples))
        estimation = np.ma.getdata(np.ma.average(m_esi_samples, axis=1, weights=self.weights))
        if self.normalize:
            zscore_estimation = (estimation - np.mean(estimation)) / np.std(estimation)
            return zscore_estimation * np.nanstd(esi_samples) + np.nanmean(esi_samples)
        else:
            return estimation


def identity(esi_samples):
    return esi_samples


# Bilateral filter
def bilateral_filter(esi_samples):

    bff = BilateralFilteringFusion(cube=esi_samples)
    fusion = bff.eval()
    two_dims_fusion = np.flip(fusion.reshape(fusion.shape[0], fusion.shape[1]), 1)

    return two_dims_fusion
