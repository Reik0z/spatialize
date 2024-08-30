import numpy as np
import scipy as sci
from cv2 import bilateralFilter


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
def Bilateral_Filter(hsi):
    class BilateralFilteringFusion:
        def __init__(self, cube, final_bands=1, c=0.1, c1=1. / 16., c2=0.05, m=10):
            self.cube = np.ma.array(cube, fill_value=0, mask=np.isnan(cube)).filled().astype(np.float32)
            self.final_bands = final_bands
            self.c = c
            self.c1 = c1
            self.c2 = c2
            self.m = m

            self.weights = None

        def filter_cube(self):
            local_cube = self.cube
            filtered_cube = np.empty(local_cube.shape, dtype=np.float32)
            m, n, k = local_cube.shape
            sigma_s = self.c1 * min(m, n)
            sigma_r = np.array(self.c2 * (local_cube.max(axis=(0, 1)) - local_cube.min(axis=(0, 1))))
            for i, img in enumerate(local_cube.swapaxes(0, 2).swapaxes(1, 2)):
                filtered_cube[:, :, i] = bilateralFilter(img, 5, sigma_r[i], sigma_s)
            return filtered_cube

        def eval(self):
            local_cube = self.cube
            m, n, l = local_cube.shape
            p = int(np.ceil(local_cube.shape[2] / self.final_bands))
            filtered_cube = self.filter_cube()

            weights = np.empty_like(local_cube)
            for i in range(self.final_bands):
                local_cube_range = local_cube[:, :, i * p:(i + 1) * p]
                filtered_cube_range = filtered_cube[:, :, i * p:(i + 1) * p]
                abs_difference = np.abs(local_cube_range - filtered_cube_range) + self.c
                abs_difference_sum = abs_difference.sum(axis=2)
                weights[:, :, i * p:(i + 1) * p] = abs_difference / abs_difference_sum[:, :, np.newaxis]

            self.weights = weights
            lin_comb = np.zeros(shape=(m, n, self.final_bands))
            for i in range(self.final_bands):
                lin_comb[:, :, i] = (self.weights[:, :, i * p:(i + 1) * p] * local_cube[:, :, i * p:(i + 1) * p]).sum(
                    axis=2)
            lin_comb[lin_comb > 65535.0] = 65535.0
            lin_comb[lin_comb < 0.0] = 0.0
            return lin_comb

    bff = BilateralFilteringFusion(cube=hsi)
    fusioned = bff.eval()
    two_dims_fusioned = np.flip(fusioned.reshape(fusioned.shape[0], fusioned.shape[1]), 1)

    return two_dims_fusioned
