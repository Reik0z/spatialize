import time, numpy as np, libspatialize as lsp
from numba import njit

from spatialize.data import load_drill_holes_andes_2D
from spatialize.logging import default_singleton_callback


@njit
def idw_local_interpolator_numba(smp: np.ndarray, val: np.ndarray, qry: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Perform Inverse Distance Weighting (IDW) interpolation for a set of query points.

    :param smp: Coordinates of known sample points.
    :type smp: np.ndarray (shape: [n_samples, n_dimensions])

    :param val: Values associated with the sample points.
    :type val: np.ndarray (shape: [n_samples])

    :param qry: Coordinates of query points where interpolation is desired.
    :type qry: np.ndarray (shape: [n_queries, n_dimensions])

    :param params: Additional parameters for interpolation (currently unused).
    :type params: np.ndarray

    :return: Interpolated values at each query point.
    :rtype: np.ndarray (shape: [n_queries])
    """
    power = params[0] if params.shape[0] > 0 else 2.0  # fallback if empty

    result = []
    for q in qry:
        dists = np.array([np.sqrt(np.sum(np.power(s - q, power * np.ones((smp.shape[1],))))) for s in smp])
        weights = 1.0 / (1 + np.power(dists, power * np.ones((len(dists),))))
        norm = np.sum(weights)
        result.append(np.sum(val * weights) / norm)
    return (np.array(result))


@njit
def idw_local_interpolator_numba2(smp: np.ndarray, val: np.ndarray, qry: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Perform fast Inverse Distance Weighting (IDW) interpolation for a set of query points using Numba.

    :param smp: Coordinates of known sample points.
    :type smp: np.ndarray (shape: [n_samples, n_dimensions])

    :param val: Values associated with the sample points.
    :type val: np.ndarray (shape: [n_samples])

    :param qry: Coordinates of query points where interpolation is desired.
    :type qry: np.ndarray (shape: [n_queries, n_dimensions])

    :param params: Interpolation parameters (expects power as the first element).
                   Example: params = np.array([2.0]) for inverse squared distance.
    :type params: np.ndarray

    :return: Interpolated values at each query point.
    :rtype: np.ndarray (shape: [n_queries])
    """
    n_qry = qry.shape[0]
    n_smp = smp.shape[0]
    power = params[0] if params.shape[0] > 0 else 2.0  # fallback if empty
    result = np.empty(n_qry, dtype=np.float64)

    for i in range(n_qry):
        q = qry[i]
        weights_sum = 0.0
        weighted_val_sum = 0.0

        for j in range(n_smp):
            s = smp[j]
            d = 0.0
            for k in range(smp.shape[1]):
                d += (s[k] - q[k]) ** 2
            d = np.sqrt(d)
            w = 1.0 / (1.0 + d ** power)
            weights_sum += w
            weighted_val_sum += val[j] * w

        result[i] = weighted_val_sum / weights_sum if weights_sum > 0 else 0.0

    return result


@njit
def idw_local_interpolator_anisotropic(
    smp: np.ndarray,
    val: np.ndarray,
    qry: np.ndarray,
    params: np.ndarray
) -> np.ndarray:
    """
    Anisotropic Inverse Distance Weighting interpolation with fixed numerical stability.

    params format:
        [power, scale_dim_0, scale_dim_1, ..., scale_dim_(n_dimensions-1)]
    If no scaling factors provided (params length == 1), isotropic scaling of 1.0 per dimension is assumed.

    :param smp: Sample points, shape (n_samples, n_dimensions).
    :param val: Values at sample points, shape (n_samples,).
    :param qry: Query points, shape (n_queries, n_dimensions).
    :param params: [power, scale_0, scale_1, ..., scale_n]
    :return: Interpolated values at query points, shape (n_queries,).
    """
    n_qry = qry.shape[0]
    n_smp = smp.shape[0]
    n_dim = smp.shape[1]

    power = params[0]

    # Use scaling if provided, else default to ones
    if params.size > 1:
        scaling = params[1:]
    else:
        scaling = np.ones(n_dim, dtype=params.dtype)

    result = np.empty(n_qry, dtype=np.float64)
    eps = 1e-12  # small epsilon for numerical stability

    for i in range(n_qry):
        q = qry[i]
        weights_sum = 0.0
        weighted_val_sum = 0.0

        for j in range(n_smp):
            s = smp[j]
            dist_sq = 0.0
            for k in range(n_dim):
                diff = (s[k] - q[k]) * scaling[k]
                dist_sq += diff * diff
            dist = np.sqrt(dist_sq)
            w = 1.0 / (eps + dist ** power)
            weights_sum += w
            weighted_val_sum += val[j] * w

        result[i] = weighted_val_sum / weights_sum if weights_sum > 0 else 0.0

    return result

# the samples included in the spatialize package
samples, locations, krig, _ = load_drill_holes_andes_2D()

# estimation data and result shape
w, h = 300, 200

# input variables for non gridded estimation spatialize functions
points = samples[['x', 'y']].values
values = samples[['cu']].values[:, 0]
xi = locations[['x', 'y']].values

power = 2.0
scaling = np.array([1.0, 1.0])  # Emphasize second dimension
params = np.concatenate(([power], scaling))

power = 2.0
params_iso = np.array([power])              # isotropic (no scaling specified)
params_aniso = np.array([power, 1.0, 1.0])  # anisotropic with all ones scaling

t = time.time()
est1 = lsp.estimation_esi_idw(
    points,
    values,
    100, 0.7, 2.0, 206936,
    xi,
    default_singleton_callback
)
t1 = time.time() - t

t = time.time()
est2 = lsp.estimation_custom_esi(
    points,
    values,
    100, 0.7, 206936,
    xi,
    None,
    idw_local_interpolator_anisotropic,
    default_singleton_callback
)
t2 = time.time() - t

d = est1[1] - est2[1]
d = d[~np.isnan(d)]

r = sorted(list(d.flatten()))
print('%d:%f' % (t1 // 60, t1 % 60), '%d:%f' % (t2 // 60, t2 % 60), r[-1])
