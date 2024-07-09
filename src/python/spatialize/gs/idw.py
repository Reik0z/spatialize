import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import ParameterGrid

import spatialize.gs
from spatialize import SpatializeError, logging
from spatialize._math_util import flatten_grid_data
from spatialize.logging import default_singleton_callback, singleton_null_callback
from spatialize.gs import LibSpatializeFacade


class GridSearchResult:
    def __init__(self, search_result_data):
        self.search_result_data = search_result_data

        data = self.search_result_data
        self.cv_error = data[['cv_error']]
        min_error = self.cv_error.min()['cv_error']
        self.best_params = data[data.cv_error <= min_error]

    def cv_error_plot(self, **kwargs):
        fig = plt.figure(figsize=(10, 4), dpi=150)
        gs = fig.add_gridspec(1, 2, wspace=0.45)
        (ax1, ax2) = gs.subplots()
        fig.suptitle("Cross Validation Error")
        self.cv_error.plot(kind='hist', ax=ax1,
                           title="Histogram",
                           rot=25,
                           colormap="Accent",
                           legend=False)
        self.cv_error.plot(kind='line', ax=ax2,
                           y='cv_error',
                           xlabel="Search result data index",
                           ylabel="Error",
                           colormap="Accent",
                           legend=False)

    def best_result(self, **kwargs):
        pass


class IDWGridSearchResult(GridSearchResult):
    def __init__(self, search_result_data):
        super().__init__(search_result_data)

    def best_result(self, optimize_data_usage=False, **kwargs):
        b_param = self.best_params.sort_values(by='radius', ascending=optimize_data_usage)
        return b_param.iloc[0].to_dict()


def idw_hparams_search(points, values, xi,
                       k=10,
                       griddata=False,
                       radius=(10, 100, 1000, 10000, 100000, np.inf),
                       exponent=tuple(np.arange(1.0, 15.0, 1.0)),
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

    # create a dataframe with all results
    result_data = pd.DataFrame(columns=list(grid.keys()) + ["cv_error"])
    for k, v in results.items():
        d = {"cv_error": v}
        d.update(param_grid[k])
        result_data = pd.concat([result_data, pd.DataFrame(d, index=[k])])

    return IDWGridSearchResult(result_data)


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
