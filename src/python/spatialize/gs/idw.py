import multiprocessing

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

import spatialize.gs
from spatialize import SpatializeError, logging
from spatialize._math_util import flatten_grid_data
from spatialize.logging import default_singleton_callback, singleton_null_callback, log_message
from spatialize.gs import lib_spatialize_facade
from spatialize._util import GridSearchResult, EstimationResult


class IDWGridSearchResult(GridSearchResult):
    def __init__(self, search_result_data):
        super().__init__(search_result_data)

    def best_result(self, optimize_data_usage=False, **kwargs):
        b_param = self.best_params.sort_values(by='radius', ascending=optimize_data_usage)
        row = pd.DataFrame(b_param.iloc[0]).to_dict(index=True)
        index = list(row.keys())[0]
        result = row[index]
        result.update({"result_data_index": index})
        return result


class IDWResult(EstimationResult):
    pass


def idw_hparams_search(points, values, xi,
                       k=10,
                       griddata=False,
                       radius=(10, 100, 1000, 10000, 100000, np.inf),
                       exponent=tuple(np.arange(0.1, 5.0, 0.1)),
                       folding_seed=np.random.randint(1000, 10000),
                       callback=default_singleton_callback
                       ):
    log_message(logging.logger.debug(f"searching best params ..."))

    method = "kfold"
    if k == points.shape[0] or k == -1:
        method = "loo"

    # get the cross validation function
    cross_validate = lib_spatialize_facade.get_operator(points,
                                                        spatialize.gs.local_interpolator.IDW,
                                                        method,
                                                        spatialize.gs.PLAIN_INTERPOLATOR,
                                                        lib_spatialize_facade.backend_options.IN_MEMORY)

    grid = {"radius": radius,
            "exponent": exponent}

    # get the actual parameter grid
    param_grid = ParameterGrid(grid)

    p_xi = xi
    if griddata:
        p_xi, _ = flatten_grid_data(xi)

    # run the scenarios
    results = {}

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
        callback(logging.progress.inform())

    callback(logging.progress.init(len(param_grid), 1))
    it = range(len(param_grid))
    for i in it:
        run_scenario(i)
    callback(logging.progress.stop())

    # create a dataframe with all results
    result_data = pd.DataFrame(columns=list(grid.keys()) + ["cv_error"])
    for k, v in results.items():
        d = {"cv_error": v}
        d.update(param_grid[k])
        if not result_data.empty:
            result_data = pd.concat([result_data, pd.DataFrame(d, index=[k])])
        else:
            result_data = pd.DataFrame(d, index=[k])

    return IDWGridSearchResult(result_data)


def idw_griddata(points, values, xi, **kwargs):
    ng_xi, original_shape = flatten_grid_data(xi)
    estimation = _call_libspatialize(points, values, ng_xi, **kwargs)
    return IDWResult(estimation, True, original_shape)


def idw_nongriddata(points, values, xi, **kwargs):
    estimation = _call_libspatialize(points, values, xi, **kwargs)
    return IDWResult(estimation)


def _call_libspatialize(points, values, xi, radius=np.inf, exponent=1.0,
                        callback=default_singleton_callback,
                        best_params_found=None):
    log_message(logging.logger.debug("running idw"))

    if best_params_found is None:
        rad = radius
        exp = exponent
    else:
        log_message(logging.logger.debug(f"using best params found: {best_params_found}"))
        rad, exp = best_params_found["radius"], best_params_found["exponent"]

    # get the estimator function
    estimate = lib_spatialize_facade.get_operator(points,
                                                  spatialize.gs.local_interpolator.IDW,
                                                  "estimate",
                                                  spatialize.gs.PLAIN_INTERPOLATOR,
                                                  lib_spatialize_facade.backend_options.IN_MEMORY)

    # get the argument list
    l_args = [np.float32(points), np.float32(values),
              rad, exp, np.float32(xi), callback]

    # run
    try:
        estimation = estimate(*l_args)
    except Exception as e:
        raise SpatializeError(e)

    return estimation
