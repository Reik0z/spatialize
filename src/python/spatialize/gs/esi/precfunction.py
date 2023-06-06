import numpy as np


def mse_precision(estimation, esi_samples):
    return apply_loss_function(estimation, esi_samples, lambda x, y: (x - y) ** 2)


def mae_precision(estimation, esi_samples):
    return apply_loss_function(estimation, esi_samples, lambda x, y: np.abs(x - y))


def apply_loss_function(estimation, esi_samples, loss_function):

    loss = np.empty(esi_samples.shape)
    for i in range(loss.shape[1]):
        loss[:, i] = loss_function(esi_samples[:, i], estimation)

    return np.nanmean(loss, axis=1)
