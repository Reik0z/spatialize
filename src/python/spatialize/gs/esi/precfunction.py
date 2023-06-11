import numpy as np

from spatialize.gs.esi.aggfunction import mean


def mse_precision(estimation, esi_samples):
    return apply_loss_function(estimation, esi_samples,
                               lambda x, y: (x - y) ** 2,
                               mean)


def mae_precision(estimation, esi_samples):
    return apply_loss_function(estimation, esi_samples,
                               lambda x, y: np.abs(x - y),
                               mean)


class OpErrorPrecision:
    def __init__(self, dyn_range=None):
        self.dyn_range = dyn_range

    def __call__(self, estimation, esi_samples):
        dyn_range = self.dyn_range
        if dyn_range is None:
            dyn_range = np.abs(np.min(estimation) - np.max(estimation))

        def op_error(x, y):
            return np.abs(x - y) / dyn_range

        return apply_loss_function(estimation, esi_samples, op_error, mean)


def apply_loss_function(estimation, esi_samples, loss_function, agg_function):
    loss = np.empty(esi_samples.shape)
    for i in range(loss.shape[1]):
        loss[:, i] = loss_function(esi_samples[:, i], estimation)

    return agg_function(loss)
