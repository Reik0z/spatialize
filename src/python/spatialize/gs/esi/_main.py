import libspatialize as lsp
from sklearn.model_selection import ParameterGrid

import spatialize.gs.esi.aggfunction as af
import spatialize.gs.esi.precfunction as pf

import numpy as np

from spatialize import SpatializeError
from spatialize._util import signature_overload


# ============================================= PUBLIC API ==========================================================
@signature_overload(pivot_arg=("base_interpolator", "idw", "base interpolator"),
                    common_args={"k": 10,
                                 "griddata": False,
                                 "n_partitions": [100],
                                 "alpha": list(np.flip(np.arange(0.7, 0.98, 0.01))),
                                 "agg_function": [af.mean, af.median, af.MAP]},
                    specific_args={
                        "idw": {"exponent": list(np.arange(1.0, 10.0, 1.0))},
                        "kriging": {"model": 1, "nugget": 0.1, "range": 50.0}
                    })
def hparams_search(points, values, xi, **kwargs):
    method, k = "kfold", kwargs["k"]
    if k == points.shape[0] or k == -1:
        method = "loo"
    cross_validate = LibSpatializeFacade.get_operator(points, kwargs["base_interpolator"], method)
    print(cross_validate)

    grid = {"n_partitions": kwargs["n_partitions"],
            "alpha": kwargs["alpha"]}

    if kwargs["base_interpolator"] == "idw":
        grid["exponent"] = kwargs["exponent"]

    p_xi = xi
    if kwargs["griddata"]:
        p_xi, _ = flatten_grid_data(xi)

    param_grid = ParameterGrid(grid)
    for param_set in param_grid:
        print(param_set)
        if kwargs["base_interpolator"] == "idw":
            if method == "loo":
                cv = cross_validate(points, values, param_set["n_partitions"],
                                    param_set["alpha"], param_set["exponent"], p_xi)
            else:
                cv = cross_validate(points, values, param_set["n_partitions"],
                                    param_set["alpha"], param_set["exponent"], k, p_xi)


def griddata(points, values, xi, **kwargs):
    ng_xi, original_shape = flatten_grid_data(xi)
    estimation, precision = nongriddata(points, values, ng_xi, **kwargs)
    return estimation.reshape(original_shape), precision.reshape(original_shape)


def nongriddata(points, values, xi, **kwargs):
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
    estimate = LibSpatializeFacade.get_operator(points, kwargs["base_interpolator"], "estimate")

    # add common args
    l_args = [np.float32(points), np.float32(values), kwargs["n_partitions"], kwargs["alpha"], np.float32(xi)]

    # add specific args
    if kwargs["base_interpolator"] == "idw":
        l_args.insert(-1, kwargs["exponent"])

    if kwargs["base_interpolator"] == "kriging":
        l_args.insert(-1, LibSpatializeFacade.get_kriging_model_number(kwargs["model"]))
        l_args.insert(-1, kwargs["nugget"])
        l_args.insert(-1, kwargs["range"])

    try:
        esi_samples = estimate(*l_args)
    except Exception as e:
        raise SpatializeError(e)

    estimation = kwargs["agg_function"](esi_samples)
    precision = kwargs["prec_function"](estimation, esi_samples)

    return estimation, precision


class LibSpatializeFacade:
    hash_map = {
        2: {"idw": {"estimate": lsp.esi_idw_2d,
                    "loo": lsp.loo_esi_idw_2d,
                    "kfold": lsp.kfold_esi_idw_2d},
            "kriging": {"estimate": lsp.esi_kriging_2d,
                        "loo": lsp.loo_esi_kriging_2d,
                        "kfold": lsp.kfold_esi_kriging_2d},
            },
        3: {"idw": {"estimate": lsp.esi_idw_3d,
                    "loo": lsp.loo_esi_idw_3d,
                    "kfold": lsp.kfold_esi_idw_3d},
            "kriging": {"estimate": lsp.esi_kriging_3d,
                        "loo": lsp.loo_esi_kriging_3d,
                        "kfold": lsp.kfold_esi_kriging_3d},
            },
    }

    kriging_models = {
        "spherical": 1,
        "exponential": 2,
        "cubic": 3,
        "gaussian": 4
    }

    @classmethod
    def get_operator(cls, points, base_interpolator, operation):
        d = int(points.shape[1])

        if d not in LibSpatializeFacade.hash_map:
            raise SpatializeError(f"Points dimension must be in {list(LibSpatializeFacade.hash_map.keys)}")

        if base_interpolator not in LibSpatializeFacade.hash_map[d]:
            raise SpatializeError(f"Base interpolator '{base_interpolator}' not supported for {str(d).upper()}-D data")

        if operation not in LibSpatializeFacade.hash_map[d][base_interpolator]:
            raise SpatializeError(f"Operation '{operation}' not supported for '{base_interpolator}' and "
                                  f"{str(d).upper()}-D data")

        return LibSpatializeFacade.hash_map[d][base_interpolator][operation]

    @classmethod
    def get_kriging_model_number(cls, model):
        return LibSpatializeFacade.kriging_models[model]


def flatten_grid_data(xi):
    try:
        if len(xi) == 1:
            ng_xi = np.column_stack((xi.flatten()))
        elif len(xi) == 2:
            (grid_x, grid_y) = xi
            ng_xi = np.column_stack((grid_x.flatten(), grid_y.flatten()))
        elif len(xi) == 3:
            (grid_x, grid_y, grid_z) = xi
            ng_xi = np.column_stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()))
        elif len(xi) == 4:
            (grid_x, grid_y, grid_z, grid_t) = xi
            ng_xi = np.column_stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten(), grid_t.flatten()))
        else:
            raise Exception
    except:
        raise SpatializeError("No grid data positions found")
    return ng_xi, grid_x.shape
