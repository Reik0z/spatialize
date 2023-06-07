import libspatialize as lsp
import spatialize.gs.esi.aggfunction as af
import spatialize.gs.esi.precfunction as pf

import numpy as np

from spatialize import SpatializeError


# ============================================= PUBLIC API ==========================================================
def hparams_search(points, values, xi, base_interpolator='idw', n_partitions=500, exponent=2):
    loo = LibSpatializeFacade.get_operator(points, base_interpolator, "loo")
    alpha = 0.7
    print(loo)

    # res = loo(points, values, n_partitions, alpha, exponent, xi)


def griddata(points, values, xi, **kwargs):
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

    estimation, precision = nongriddata(points, values, ng_xi, **kwargs)

    if len(xi) == 1:
        estimation = estimation.reshape([grid_x.shape[0]])
        precision = precision.reshape([grid_x.shape[0]])
    elif len(xi) == 2:
        estimation = estimation.reshape([grid_x.shape[0], grid_x.shape[1]])
        precision = precision.reshape([grid_x.shape[0], grid_x.shape[1]])
    elif len(xi) == 3:
        estimation = estimation.reshape([grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]])
        precision = precision.reshape([grid_x.shape[0], grid_x.shape[1], grid_x.shape[2]])
    elif len(xi) == 4:
        estimation = estimation.reshape([grid_x.shape[0], grid_x.shape[1], grid_x.shape[2], grid_x.shape[3]])
        precision = precision.reshape([grid_x.shape[0], grid_x.shape[1], grid_x.shape[2], grid_x.shape[3]])

    return estimation, precision


def nongriddata(points, values, xi, **kwargs):
    return _call_libspatialize(points, values, xi, **kwargs)


# =========================================== END of PUBLIC API ======================================================
def signature_overload(default_base_interpolator, common_args, specific_args):
    def outer_function(func):
        def inner_function(*args, **kwargs):
            if "base_interpolator" not in kwargs:
                kwargs["base_interpolator"] = default_base_interpolator
            bi = kwargs["base_interpolator"]

            if bi not in specific_args:
                raise SpatializeError(f"Base interpolator '{bi}' not supported")

            # if the common argument is needed for the base interpolator
            # and is not in kwargs then add it with its declared
            # default value
            for arg in common_args.keys():
                if arg not in kwargs:
                    kwargs[arg] = common_args[arg]

            # get the specific args for the current base interpolator
            spec_args = specific_args[bi]

            # if the specific argument is needed for the base interpolator
            # and is not in kwargs then add it with its declared
            # default value
            for arg in spec_args.keys():
                if arg not in kwargs:
                    kwargs[arg] = spec_args[arg]

            # check that all arguments are consistent
            # with the base interpolator
            for arg in kwargs.keys():
                if arg != "base_interpolator" and arg not in spec_args and arg not in common_args:
                    raise SpatializeError(f"Argument '{arg}' not recognized for '{bi}' base interpolator")

            return func(*args, **kwargs)

        return inner_function

    return outer_function


@signature_overload(default_base_interpolator="idw",
                    common_args={"base_interpolator": "idw",
                                 "n_partitions": 100,
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
