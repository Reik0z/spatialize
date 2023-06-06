import numpy as np
import scipy as sci


def mean(values):
    return np.nanmean(values, axis=1)


def median(values):
    return np.nanmedian(values, axis=1)


def MAP(values):
    return sci.stats.mode(values, axis=1, keepdims=True, nan_policy="omit").mode