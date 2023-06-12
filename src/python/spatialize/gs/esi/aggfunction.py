import numpy as np
import scipy as sci


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

