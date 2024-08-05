import tempfile

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import ParameterGrid

from spatialize import SpatializeError, logging
import spatialize.gs.esi.aggfunction as af
import spatialize.gs.esi.precfunction as pf
from spatialize._util import signature_overload, GridSearchResult, EstimationResult
from spatialize._math_util import flatten_grid_data
from spatialize.gs import lib_spatialize_facade, partitioning_process, local_interpolator as li
from spatialize.logging import log_message, default_singleton_callback, singleton_null_callback


class ESIGridSearchResult(GridSearchResult):
    def __init__(self, search_result_data, agg_function_map, p_process):
        super().__init__(search_result_data)
        self.agg_func_map = agg_function_map
        self.p_process = p_process

    def best_result(self, **kwargs):
        b_param = self.best_params.sort_values(by='cv_error', ascending=True)
        row = pd.DataFrame(b_param.iloc[0]).to_dict(index=True)
        index = list(row.keys())[0]
        result = row[index]
        result.update({"result_data_index": index,
                       "agg_function": self.agg_func_map[result["agg_func_name"]],
                       "p_process": self.p_process})
        return result


class ESIResult(EstimationResult):
    def __init__(self, estimation, esi_samples, griddata=False, original_shape=None):
        super().__init__(estimation, griddata, original_shape)
        self.esi_samples = esi_samples
        self._precision = None

    def precision(self, prec_function=pf.mse_precision):
        log_message(logging.logger.debug(f'applying "{prec_function}" precision function'))
        prec = prec_function(self._estimation, self.esi_samples)

        if self.griddata:
            self._precision = prec.reshape(self.original_shape)
        else:
            self._precision = prec

        return self._precision

    def re_estimate(self, agg_function=af.mean):
        self._estimation = agg_function(self.esi_samples)
        return self.estimation()

    def plot_precision(self, ax=None, w=None, h=None, **figargs):
        if self._precision is None:
            self._precision = self.precision()
        if 'cmap' not in figargs:
            figargs['cmap'] = 'bwr'
        self._plot_data(self._precision, ax, w, h, **figargs)

    def quick_plot(self, w=None, h=None, **figargs):
        fig = plt.figure(dpi=150, **figargs)
        gs = fig.add_gridspec(1, 2, wspace=0.45)
        (ax1, ax2) = gs.subplots()

        ax1.set_title('Estimation')
        self.plot_estimation(ax1, w=w, h=h)
        ax1.set_aspect('auto')

        ax2.set_title('Precision')
        self.plot_precision(ax2, w=w, h=h)
        ax2.set_aspect('auto')

        return fig  # just in case you want to embed it somewhere else


# ============================================= PUBLIC API ==========================================================
@signature_overload(pivot_arg=("local_interpolator", li.IDW, "local interpolator"),
                    common_args={"k": 10,
                                 "griddata": False,
                                 "p_process": partitioning_process.MONDRIAN,  # partitioning process
                                 "data_cond": [True, False],  # whether to condition the partitioning process on samples
                                 # -- valid only when ‘p_process’ is ‘voronoi’.
                                 "n_partitions": [100],
                                 "alpha": list(np.flip(np.arange(0.70, 0.90, 0.01))),
                                 "agg_function": {"mean": af.mean, "median": af.median},
                                 "seed": np.random.randint(1000, 10000),
                                 "folding_seed": np.random.randint(1000, 10000),
                                 "callback": default_singleton_callback,
                                 "backend": None,  # it can be: None or one the lib_spatialize_facade.backend_options
                                 "cache_path": None,  # Needed if 'backend' is
                                 # lib_spatialize_facade.backend_options.DISK_CACHED
                                 },
                    specific_args={
                        li.IDW: {"exponent": list(np.arange(1.0, 15.0, 1.0))},
                        li.KRIGING: {"model": ["spherical", "exponential", "cubic", "gaussian"],
                                     "nugget": [0.0, 0.5, 1.0],
                                     "range": [10.0, 50.0, 100.0, 200.0]}
                    })
def esi_hparams_search(points, values, xi, **kwargs):
    log_message(logging.logger.debug(f"searching best params ..."))
    log_message(logging.logger.debug(f'backend: {"auto" if kwargs["backend"] is None else kwargs["backend"]}'))

    method, k = "kfold", kwargs["k"]
    if k == points.shape[0] or k == -1:
        method = "loo"

    # get the cross validation function
    cross_validate = lib_spatialize_facade.get_operator(points, kwargs["local_interpolator"],
                                                        method, kwargs["p_process"],
                                                        kwargs["backend"])

    grid = {"n_partitions": kwargs["n_partitions"],
            "alpha": kwargs["alpha"]}

    if kwargs["p_process"] == partitioning_process.VORONOI:
        grid["data_cond"] = kwargs["data_cond"]

    if kwargs["local_interpolator"] == li.IDW:
        grid["exponent"] = kwargs["exponent"]

    if kwargs["local_interpolator"] == li.KRIGING:
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

    def run_scenario(i):
        param_set = param_grid[i].copy()
        param_set["local_interpolator"] = kwargs["local_interpolator"]
        param_set["seed"] = kwargs["seed"]
        param_set["callback"] = singleton_null_callback
        param_set["backend"] = kwargs["backend"]
        param_set["cache_path"] = kwargs["cache_path"]
        param_set["p_process"] = kwargs["p_process"]

        if kwargs["p_process"] == partitioning_process.MONDRIAN:
            param_set["data_cond"] = True

        l_args = build_arg_list(points, values, p_xi, param_set)
        if method == "kfold":
            l_args.insert(-2, k)
            l_args.insert(-2, kwargs["folding_seed"])

        model, cv = cross_validate(*l_args)

        for agg_func_name, agg_func in kwargs["agg_function"].items():
            results[(agg_func_name, i)] = np.nanmean(np.abs(values - agg_func(cv)))

        kwargs["callback"](logging.progress.inform())

    it = range(len(param_grid))
    kwargs["callback"](logging.progress.init(len(param_grid), 1))
    for i in it:
        run_scenario(i)
    kwargs["callback"](logging.progress.stop())

    # create a dataframe with all results
    result_data = pd.DataFrame(columns=list(grid.keys()) + ["cv_error"])
    for k, v in results.items():
        d = {"agg_func_name": k[0],
             "cv_error": v,
             "local_interpolator": kwargs["local_interpolator"],
             }
        d.update(param_grid[k[1]])
        if not result_data.empty:
            result_data = pd.concat([result_data, pd.DataFrame(d, index=[k[1]])])
        else:
            result_data = pd.DataFrame(d, index=[k[1]])

    return ESIGridSearchResult(result_data, kwargs["agg_function"], kwargs["p_process"])


def esi_griddata(points, values, xi, **kwargs):
    ng_xi, original_shape = flatten_grid_data(xi)
    estimation, esi_samples = _call_libspatialize(points, values, ng_xi, **kwargs)
    return ESIResult(estimation, esi_samples, griddata=True, original_shape=original_shape)


def esi_nongriddata(points, values, xi, **kwargs):
    estimation, esi_samples = _call_libspatialize(points, values, xi, **kwargs)
    return ESIResult(estimation, esi_samples)


# =========================================== END of PUBLIC API ======================================================
@signature_overload(pivot_arg=("local_interpolator", li.IDW, "local interpolator"),
                    common_args={"n_partitions": 500,
                                 "p_process": partitioning_process.MONDRIAN,  # partitioning process
                                 "data_cond": True,  # whether to condition the partitioning process on samples
                                 # -- valid only when ‘p_process’ is ‘voronoi’.
                                 "alpha": 0.8,
                                 "agg_function": af.mean,
                                 "seed": np.random.randint(1000, 10000),
                                 "callback": default_singleton_callback,
                                 "backend": None,  # it can be: None or one the lib_spatialize_facade.backend_options
                                 "cache_path": None,  # Needed if 'backend' is
                                 # lib_spatialize_facade.backend_options.DISK_CACHED
                                 "best_params_found": None
                                 },
                    specific_args={
                        li.IDW: {"exponent": 2.0},
                        li.KRIGING: {"model": 1, "nugget": 0.1, "range": 5000.0}
                    })
def _call_libspatialize(points, values, xi, **kwargs):
    log_message(logging.logger.debug('calling libspatialize'))
    log_message(logging.logger.debug(f'backend: {"auto" if kwargs["backend"] is None else kwargs["backend"]}'))

    if not kwargs["best_params_found"] is None:
        try:
            del kwargs["best_params_found"]["n_partitions"]  # this param can be overwritten all cases
        except KeyError:
            pass
        log_message(logging.logger.debug(f"using best params found: {kwargs['best_params_found']}"))
        for k in kwargs["best_params_found"]:
            try:
                kwargs[k] = kwargs["best_params_found"][k]
            except KeyError:
                pass

    # get the estimator function
    estimate = lib_spatialize_facade.get_operator(points, kwargs["local_interpolator"],
                                                  "estimate", kwargs["p_process"],
                                                  kwargs["backend"])

    # get the argument list
    l_args = build_arg_list(points, values, xi, kwargs)

    # run
    try:
        esi_model, esi_samples = estimate(*l_args)
    except Exception as e:
        raise SpatializeError(e)

    estimation = kwargs["agg_function"](esi_samples)

    return estimation, esi_samples


def build_arg_list(points, values, xi, nonpos_args):
    alpha = nonpos_args["alpha"]
    if nonpos_args["p_process"] == partitioning_process.VORONOI and not nonpos_args["data_cond"]:
        alpha *= -1

    # add initial common args
    l_args = [np.float32(points), np.float32(values),
              nonpos_args["n_partitions"], alpha, np.float32(xi), nonpos_args["callback"]]

    # add specific args
    if nonpos_args["local_interpolator"] == li.IDW:
        l_args.insert(-2, nonpos_args["exponent"])
        l_args.insert(-2, nonpos_args["seed"])

    if nonpos_args["local_interpolator"] == li.KRIGING:
        l_args.insert(-2, lib_spatialize_facade.get_kriging_model_number(nonpos_args["model"]))
        l_args.insert(-2, nonpos_args["nugget"])
        l_args.insert(-2, nonpos_args["range"])
        l_args.insert(-2, nonpos_args["seed"])

    if nonpos_args["backend"] == lib_spatialize_facade.backend_options.DISK_CACHED:
        cache_path = nonpos_args["cache_path"]
        if cache_path is None:
            cache_path = tempfile.TemporaryDirectory().name + ".db"
            log_message(logging.logger.debug(f'cache path: {cache_path}'))
            l_args.insert(0, cache_path)

    return l_args
