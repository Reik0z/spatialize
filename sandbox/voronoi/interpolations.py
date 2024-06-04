import numpy as np
from numba import njit
from scipy.optimize import minimize
from .geometry_transformations import get_tras_rot_2d


# Classic IDW interpolations 2-D
# @njit
def idw_interpolation_old(point, data, exp_dist=1.0, smooth=0.0, dists=None, pos_cols=['X', 'Y'], value_col='grade'):
    if len(data.shape) < 2:  # only one data
        return np.sum(data[value_col])
    if dists is None:
        dists = np.linalg.norm(data[pos_cols].values - point, axis=1)

    mask_zero_dist = dists == 0
    # mask for data in the same point position
    if smooth == 0 and np.sum(mask_zero_dist) > 0:
        weights = np.zeros(len(dists))
        weights[mask_zero_dist] = 1.
    else:
        weights = 1. / (smooth + dists ** exp_dist)

    w_norm = weights / np.sum(weights)
    return np.sum(w_norm * data[value_col])

def idw_interpolation(point, data, value, exp_dist=1.0, smooth=0.0, dists=None):
    if len(data.shape) < 2:  # only one data
        return np.sum(value)

    if dists is None:
        # dists = data[pos_cols].apply(lambda x: np.linalg.norm(x - point), engine="numba", axis=1)
        dists = np.linalg.norm(data - point)

    mask_zero_dist = (dists == 0) * 1
    # mask for data in the same point position
    if smooth == 0 and np.sum(mask_zero_dist) > 0:
        weights = np.zeros(len(dists))
        weights[mask_zero_dist] = 1.
    else:
        weights = 1. / (smooth + dists ** exp_dist)

    w_norm = weights / np.sum(weights)
    return np.sum(w_norm * value)


# ----------------------
# Anisotropic IDW interpolations 2-D (exponent, anisotropy factor and azimuth angle)
def get_distances(orig, points, af, azm):
    new_points = get_tras_rot_2d(points, azm, orig) * np.array([[af, 1]])
    dists = np.sqrt(np.sum(new_points ** 2, axis=1))
    return dists


def idw_anisotropic_interpolation(point, data, exp_dist=1, anis_factor=1.0, azimuth=0.0, smooth=0.0,
                                  pos_cols=['X', 'Y'], value_col='grade'):
    if len(data.shape) < 2:  # only one data
        return np.sum(data[value_col])
    dists = get_distances(point, data[pos_cols].values, anis_factor, azimuth)

    mask_zero_dist = dists == 0
    # mask for data in the same point position
    if smooth == 0 and np.sum(mask_zero_dist) > 0:
        weights = np.zeros(len(dists))
        weights[mask_zero_dist] = 1.
    else:
        weights = 1. / (smooth + dists ** exp_dist)

    w_norm = weights / np.sum(weights)
    return np.sum(w_norm * data[value_col])


def optim_anisotropic_idw(data, x0=None, metric='mae', pos_cols=['X', 'Y'], value_col='grade'):
    # x0 = [exponent, af, azm]  -> exponent of IDW, anisotropy factor (applied to rotated x axis), azimuth angle (rads)
    if x0 is None:
        x0 = np.array([1.0, 1.0, 0.0])
    if len(data.shape) < 2 or data.shape[0] < 2:
        return x0

    if metric == 'mae':
        res = minimize(full_dist_mae_anisotropic_idw, x0, args=(data, pos_cols, value_col),
                       options={'maxiter': 3, 'disp': False})
    elif metric == 'mse':
        res = minimize(full_dist_mse_anisotropic_idw, x0, args=(data, pos_cols, value_col),
                       options={'maxiter': 3, 'disp': False})
    else:
        raise Exception('Wrong metric for optimization')
    return res.x


def full_dist_mse_anisotropic_idw(x, data, pos_cols, value_col):
    exponent, af, azm = x[0], x[1], x[2]
    errors = [(est_data_anisotropic_idw(idx, data, exponent, af, azm, pos_cols, value_col) -
               data[value_col].iloc[idx]) ** 2 for idx in range(len(data))]
    return sum(errors) / len(data)


def full_dist_mae_anisotropic_idw(x, data, pos_cols, value_col):
    exponent, af, azm = x[0], x[1], x[2]
    errors = [np.abs(est_data_anisotropic_idw(idx, data, exponent, af, azm, pos_cols, value_col) -
                     data[value_col].iloc[idx]) for idx in range(len(data))]
    return sum(errors) / len(data)


def est_data_anisotropic_idw(idx, data, exponent, af, azm, pos_cols, value_col):
    point = data[pos_cols].iloc[idx].values
    filtered_data = data.drop(data.index[idx], axis=0)
    return idw_anisotropic_interpolation(point, filtered_data, exp_dist=exponent, anis_factor=af, azimuth=azm,
                                         pos_cols=pos_cols, value_col=value_col)
