import numpy as np
from spatialize.gs.esi.aggfunction import mean, identity


def loss(agg_function):
    def outer_function(function):
        function_name = function.__name__
        module_name = function.__module__

        class inner_function:
            def __call__(self, estimation, esi_samples):
                return _apply_loss_function(estimation, esi_samples,
                                            function,
                                            agg_function)

            def __repr__(self):
                return f"<decorated--{module_name}.{function_name}>"

        return inner_function()

    return outer_function


@loss(mean)
def mse_loss(x, y):
    return (x - y) ** 2


@loss(mean)
def mae_loss(x, y):
    return np.abs(x - y)


@loss(identity)
def mse_cube(x, y):
    return mse_loss(x, y)


@loss(identity)
def mae_cube(x, y):
    return mae_loss(x, y)


class OperationalErrorLoss:
    def __init__(self, dyn_range=None, use_cube=False):
        self.dyn_range = dyn_range

    def __call__(self, estimation, esi_samples):
        dyn_range = self.dyn_range
        if dyn_range is None:
            dyn_range = np.abs(np.min(estimation) - np.max(estimation))

        @loss(identity)
        def _op_error_cube(x, y):
            return np.abs(x - y) / dyn_range

        @loss(mean)
        def _op_error_aggregated(x, y):
            return np.abs(x - y) / dyn_range

        _op_error = _op_error_aggregated
        if self.use_cube:
            _op_error = _op_error_cube

        return _op_error(estimation, esi_samples)


def _apply_loss_function(estimation, esi_samples, loss_function, agg_function):
    loss = np.empty(esi_samples.shape)
    for i in range(loss.shape[1]):
        loss[:, i] = loss_function(esi_samples[:, i], estimation)

    return agg_function(loss)
