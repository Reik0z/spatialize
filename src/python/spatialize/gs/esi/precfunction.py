import numpy as np

from spatialize.gs.esi.aggfunction import mean


def loss(function):
    function_name = function.__name__
    module_name = function.__module__

    class inner_function:
        def __call__(self, estimation, esi_samples):
            return _apply_loss_function(estimation, esi_samples,
                                        function,
                                        mean)

        def __repr__(self):
            return f"<decorated--{module_name}.{function_name}>"

    return inner_function()


@loss
def mse_precision(x, y):
    return (x - y) ** 2


@loss
def mae_precision(x, y):
    return np.abs(x - y)


class OperationalErrorPrecision:
    def __init__(self, dyn_range=None):
        self.dyn_range = dyn_range

    def __call__(self, estimation, esi_samples):
        dyn_range = self.dyn_range
        if dyn_range is None:
            dyn_range = np.abs(np.min(estimation) - np.max(estimation))

        @loss
        def _op_error(x, y):
            return np.abs(x - y) / dyn_range

        return _op_error(estimation, esi_samples)


def _apply_loss_function(estimation, esi_samples, loss_function, agg_function):
    loss = np.empty(esi_samples.shape)
    for i in range(loss.shape[1]):
        loss[:, i] = loss_function(esi_samples[:, i], estimation)

    return agg_function(loss)
