import numpy as np
from sklearn.model_selection import ParameterGrid

import spatialize.gs
from spatialize import SpatializeError, logging
from spatialize._math_util import flatten_grid_data
from spatialize.logging import default_singleton_callback, singleton_null_callback
from spatialize.gs import LibSpatializeFacade


def idw_hparams_search(points, values, xi,
                       k=10,
                       griddata=False,
                       radius=(10, 100, 1000, 10000, 100000, np.inf),
                       exponent=tuple(np.arange(1.0, 15.0, 1.0)),
                       seed=np.random.randint(1000, 10000),
                       folding_seed=np.random.randint(1000, 10000),
                       callback=default_singleton_callback
                       ):
    method = "kfold"
    if k == points.shape[0] or k == -1:
        method = "loo"

    # get the cross validation function
    cross_validate = LibSpatializeFacade.get_operator(points,
                                                      spatialize.gs.PLAINIDW,
                                                      method,
                                                      LibSpatializeFacade.BackendOptions.IN_MEMORY)

    grid = {"radius": radius,
            "exponent": exponent}

    # get the actual parameter grid
    param_grid = ParameterGrid(grid)

    p_xi = xi
    if griddata:
        p_xi, _ = flatten_grid_data(xi)

    # run the scenarios
    results = {}
    it = range(len(param_grid))

    def run_scenario(i):
        param_set = param_grid[i].copy()

        l_args = [np.float32(points),
                  np.float32(values),
                  param_set["radius"],
                  param_set["exponent"],
                  singleton_null_callback]

        if method == "kfold":
            l_args.insert(-1, k)
            l_args.insert(-1, folding_seed)

        cv = cross_validate(*l_args)
        results[i] = np.nanmean(np.abs(values - cv))

    callback(logging.progress.init(len(param_grid), 1))
    for i in it:
        run_scenario(i)
        callback(logging.progress.inform())
    callback(logging.progress.stop())

    # sort results
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=False))

    # look for the best params combination
    best_key = list(results.keys())[0]
    best_params = param_grid[best_key]

    # this is just a reminder that a better way to return
    # search results needs to be implemented
    #
    # for k, v in results.items():
    #     print(f"({k[0], param_grid[k[1]]}) ---> {v}")
    return best_params


def idw_griddata(points, values, xi, **kwargs):
    ng_xi, original_shape = flatten_grid_data(xi)
    estimation = idw_nongriddata(points, values, ng_xi, **kwargs)
    return estimation.reshape(original_shape)


def idw_nongriddata(points, values, xi, radius=np.inf, exponent=1.0,
                    callback=default_singleton_callback):
    # get the estimator function
    estimate = LibSpatializeFacade.get_operator(points,
                                                spatialize.gs.PLAINIDW,
                                                "estimate", LibSpatializeFacade.BackendOptions.IN_MEMORY)

    # get the argument list
    l_args = [np.float32(points), np.float32(values),
              radius, exponent, np.float32(xi), callback]

    # run
    try:
        estimation = estimate(*l_args)
    except Exception as e:
        raise SpatializeError(e)

    return estimation
