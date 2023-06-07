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


def apply_loss_function(estimation, esi_samples, loss_function, agg_function):

    loss = np.empty(esi_samples.shape)
    for i in range(loss.shape[1]):
        loss[:, i] = loss_function(esi_samples[:, i], estimation)

    return agg_function(loss)
