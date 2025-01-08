import numpy as np
import scipy as sci

from spatialize._math_util import BilateralFilteringFusion


def mean(samples):
    """
    Computes the mean of the samples.
    """
    return np.nanmean(samples, axis=1)


def median(samples):
    """
    Computes the median of the samples.
    """
    return np.nanmedian(samples, axis=1)


def MAP(samples):
    """
    Computes the mode of the samples.
    """
    return sci.stats.mode(samples, axis=1, keepdims=True, nan_policy="omit").mode


class Percentile:
    """
    Computes the percentile of the samples.
    """
    def __init__(self, q=75):
        self.q = q

    def __call__(self, samples):
        return np.nanpercentile(samples, self.q, axis=1)

    def __repr__(self):
        return f"percentile({self.q})"


class WeightedAverage:
    """
    Computes the weighted average of the samples.
    """
    def __init__(self, normalize=False, weights=None, force_resample=True):
        self.normalize = normalize
        self.weights = weights
        self.force_resample = force_resample

    def __call__(self, samples):
        s = samples.shape[1]
        if self.weights is None or self.force_resample:
            rng = np.random.default_rng()
            self.weights = rng.dirichlet([1] * s)
        m_samples = np.ma.array(samples, mask=np.isnan(samples))
        estimation = np.ma.getdata(np.ma.average(m_samples, axis=1, weights=self.weights))
        if self.normalize:
            zscore_estimation = (estimation - np.mean(estimation)) / np.std(estimation)
            return zscore_estimation * np.nanstd(samples) + np.nanmean(samples)
        else:
            return estimation


def identity(samples):
    """
    Returns the samples as they are
    """
    return samples


# Bilateral filter
def bilateral_filter(samples):
    """
    Applies a bilateral filter to the samples.
    """
    bff = BilateralFilteringFusion(cube=samples)
    fusion = bff.eval()
    two_dims_fusion = np.flip(fusion.reshape(fusion.shape[0], fusion.shape[1]), 1)

    return two_dims_fusion
