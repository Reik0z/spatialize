import tempfile
from concurrent.futures.thread import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
from sklearn.model_selection import ParameterGrid

from spatialize import SpatializeError, logging
import spatialize.gs.esi.aggfunction as af
import spatialize.gs.esi.precfunction as pf
from spatialize._util import signature_overload, get_progress_bar, in_notebook
from spatialize._math_util import flatten_grid_data
from spatialize.gs import LibSpatializeFacade
from spatialize.logging import MessageHandler, LogMessage, AsyncProgressCounter, log_message, \
    SingletonAsyncProgressCounter


def default_callback(msg):
    return MessageHandler([LogMessage(), SingletonAsyncProgressCounter()])(msg)

# ============================================= PUBLIC API ==========================================================
@signature_overload(pivot_arg=("base_interpolator", "idw", "base interpolator"),
                    common_args={"k": 10,
                                 "griddata": False,
                                 "n_partitions": [30],
                                 "alpha": list(np.flip(np.arange(0.70, 0.90, 0.01))),
                                 "agg_function": {"mean": af.mean, "median": af.median},
                                 "seed": np.random.randint(1000, 10000),
                                 "folding_seed": np.random.randint(1000, 10000),
                                 "callback": default_callback,
                                 "backend": None,  # it can be: None or one the LibSpatializeFacade.BackendOptions
                                 "cache_path": None,  # Needed if 'backend' is
                                                      # LibSpatializeFacade.BackendOptions.DISK_CACHED
                                 "show_progress": True},
                    specific_args={
                        "idw": {"exponent": list(np.arange(1.0, 15.0, 1.0))},
                        "kriging": {"model": ["spherical", "exponential", "cubic", "gaussian"],
                                    "nugget": [0.0, 0.5, 1.0],
                                    "range": [10.0, 50.0, 100.0, 200.0]}
                    })
def esi_hparams_search(points, values, xi, **kwargs):
    log_message(logging.logger.info(f'backend: {"auto" if kwargs["backend"] is None else kwargs["backend"]}'))

    method, k = "kfold", kwargs["k"]
    if k == points.shape[0] or k == -1:
        method = "loo"

    # get the cross validation function
    cross_validate = LibSpatializeFacade.get_operator(points, kwargs["base_interpolator"], method, kwargs["backend"])

    grid = {"n_partitions": kwargs["n_partitions"],
            "alpha": kwargs["alpha"]}

    if kwargs["base_interpolator"] == "idw":
        grid["exponent"] = kwargs["exponent"]

    if kwargs["base_interpolator"] == "kriging":
        grid["model"] = kwargs["model"]
        grid["nugget"] = kwargs["nugget"]
        grid["range"] = kwargs["range"]

    # get the actual parameter grid
    param_grid = ParameterGrid(grid)

    p_xi = xi
    if kwargs["griddata"]:
        p_xi, _ = flatten_grid_data(xi)

    # run the scenarios
    results = {}
    it = range(len(param_grid))
    if kwargs["show_progress"]:
        it = get_progress_bar(range(len(param_grid)), "searching the grid ...")

    def run_scenario(i):
        param_set = param_grid[i].copy()
        param_set["base_interpolator"] = kwargs["base_interpolator"]
        param_set["seed"] = kwargs["seed"]
        param_set["callback"] = kwargs["callback"]
        param_set["backend"] = kwargs["backend"]
        param_set["cache_path"] = kwargs["cache_path"]

        l_args = build_arg_list(points, values, p_xi, param_set)
        if method == "kfold":
            l_args.insert(-2, k)
            l_args.insert(-2, kwargs["folding_seed"])

        model, cv = cross_validate(*l_args)

        for agg_func_name, agg_func in kwargs["agg_function"].items():
            results[(agg_func_name, i)] = np.nanmean(np.abs(values - agg_func(cv)))

    for i in it:
        run_scenario(i)

    # sort results
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=False))

    # look for the best params combination
    best_key = list(results.keys())[0]
    best_params = param_grid[best_key[1]]
    best_params["agg_function"] = kwargs["agg_function"][best_key[0]]
    best_params["base_interpolator"] = kwargs["base_interpolator"]

    # this is just a reminder that a better way to return
    # search results needs to be implemented
    #
    # for k, v in results.items():
    #     print(f"({k[0], param_grid[k[1]]}) ---> {v}")
    return best_params


def esi_griddata(points, values, xi, **kwargs):
    ng_xi, original_shape = flatten_grid_data(xi)
    estimation, precision = _call_libspatialize(points, values, ng_xi, **kwargs)
    return estimation.reshape(original_shape), precision.reshape(original_shape)


def esi_nongriddata(points, values, xi, **kwargs):
    return _call_libspatialize(points, values, xi, **kwargs)


# =========================================== END of PUBLIC API ======================================================
@signature_overload(pivot_arg=("base_interpolator", "idw", "base interpolator"),
                    common_args={"n_partitions": 100,
                                 "alpha": 0.8,
                                 "agg_function": af.mean,
                                 "prec_function": pf.mse_precision,
                                 "seed": np.random.randint(1000, 10000),
                                 "callback": default_callback,
                                 "backend": None,  # it can be: None or one the LibSpatializeFacade.BackendOptions
                                 "cache_path": None  # Needed if 'backend' is
                                                     # LibSpatializeFacade.BackendOptions.DISK_CACHED
                                 },
                    specific_args={
                        "idw": {"exponent": 2.0},
                        "kriging": {"model": 1, "nugget": 0.1, "range": 5000.0}
                    })
def _call_libspatialize(points, values, xi, **kwargs):
    log_message(logging.logger.info(f'backend: {"auto" if kwargs["backend"] is None else kwargs["backend"]}'))

    # get the estimator function
    estimate = LibSpatializeFacade.get_operator(points, kwargs["base_interpolator"], "estimate", kwargs["backend"])

    # get the argument list
    l_args = build_arg_list(points, values, xi, kwargs)

    # run
    try:
        esi_model, esi_samples = estimate(*l_args)
    except Exception as e:
        raise SpatializeError(e)

    estimation = kwargs["agg_function"](esi_samples)
    precision = kwargs["prec_function"](estimation, esi_samples)

    return estimation, precision


def build_arg_list(points, values, xi, nonpos_args):
    # add initial common args
    l_args = [np.float32(points), np.float32(values),
              nonpos_args["n_partitions"], nonpos_args["alpha"], np.float32(xi), nonpos_args["callback"]]

    # add specific args
    if nonpos_args["base_interpolator"] == "idw":
        l_args.insert(-2, nonpos_args["exponent"])
        l_args.insert(-2, nonpos_args["seed"])

    if nonpos_args["base_interpolator"] == "kriging":
        l_args.insert(-2, LibSpatializeFacade.get_kriging_model_number(nonpos_args["model"]))
        l_args.insert(-2, nonpos_args["nugget"])
        l_args.insert(-2, nonpos_args["range"])
        l_args.insert(-2, nonpos_args["seed"])

    cached_disk = LibSpatializeFacade.BackendOptions.DISK_CACHED
    if nonpos_args["backend"] in set([None, cached_disk]):
        cache_path = nonpos_args["cache_path"]
        if cache_path is None:
            cache_path = tempfile.TemporaryDirectory().name + ".db"

        if nonpos_args["backend"] is None:  # setting the backend automatically
            if in_notebook():
                log_message(logging.logger.debug(f'cache path: {cache_path}'))
                l_args.insert(0, cache_path)
        else:
            log_message(logging.logger.debug(f'cache path: {cache_path}'))
            l_args.insert(0, cache_path)

    return l_args
