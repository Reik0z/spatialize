from concurrent.futures.thread import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
from sklearn.model_selection import ParameterGrid

from spatialize import SpatializeError
import spatialize.gs.esi.aggfunction as af
import spatialize.gs.esi.precfunction as pf
from spatialize._util import signature_overload, is_notebook, flatten_grid_data
from spatialize.gs import LibSpatializeFacade

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# ============================================= PUBLIC API ==========================================================
@signature_overload(pivot_arg=("base_interpolator", "idw", "base interpolator"),
                    common_args={"k": 10,
                                 "griddata": False,
                                 "n_partitions": [100],
                                 "alpha": list(np.flip(np.arange(0.70, 0.90, 0.01))),
                                 "agg_function": {"mean": af.mean, "median": af.median},
                                 "show_progress": True},
                    specific_args={
                        "idw": {"exponent": list(np.arange(1.0, 15.0, 1.0))},
                        "kriging": {"model": ["spherical", "exponential", "cubic", "gaussian"],
                                    "nugget": [0.0, 0.5, 1.0],
                                    "range": [10.0, 50.0, 100.0, 200.0]}
                    })
def esi_hparams_search(points, values, xi, **kwargs):
    method, k = "kfold", kwargs["k"]
    if k == points.shape[0] or k == -1:
        method = "loo"

    # get the cross validation function
    cross_validate = LibSpatializeFacade.get_operator(points, kwargs["base_interpolator"], method)

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
        it = tqdm(range(len(param_grid)), desc="searching the grid")

    def run_scenario(i):
        param_set = param_grid[i].copy()
        param_set["base_interpolator"] = kwargs["base_interpolator"]

        l_args = build_arg_list(points, values, p_xi, param_set)
        if method == "kfold":
            l_args.insert(-1, k)

        cv = cross_validate(*l_args)

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
                                 "prec_function": pf.mse_precision},
                    specific_args={
                        "idw": {"exponent": 2.0},
                        "kriging": {"model": 1, "nugget": 0.1, "range": 5000.0}
                    })
def _call_libspatialize(points, values, xi, **kwargs):
    # get the estimator function
    estimate = LibSpatializeFacade.get_operator(points, kwargs["base_interpolator"], "estimate")

    # get the argument list
    l_args = build_arg_list(points, values, xi, kwargs)

    # run
    try:
        esi_samples = estimate(*l_args)
    except Exception as e:
        raise SpatializeError(e)

    estimation = kwargs["agg_function"](esi_samples)
    precision = kwargs["prec_function"](estimation, esi_samples)

    return estimation, precision


def build_arg_list(points, values, xi, nonpos_args):
    # add common args
    l_args = [np.float32(points), np.float32(values), nonpos_args["n_partitions"], nonpos_args["alpha"], np.float32(xi)]

    # add specific args
    if nonpos_args["base_interpolator"] == "idw":
        l_args.insert(-1, nonpos_args["exponent"])

    if nonpos_args["base_interpolator"] == "kriging":
        l_args.insert(-1, LibSpatializeFacade.get_kriging_model_number(nonpos_args["model"]))
        l_args.insert(-1, nonpos_args["nugget"])
        l_args.insert(-1, nonpos_args["range"])

    return l_args
