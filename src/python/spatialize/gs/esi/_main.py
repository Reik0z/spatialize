import libspatialize as lsp
import spatialize.gs.esi.aggfunction as af
import spatialize.gs.esi.precfunction as pf

import numpy as np


class SpatializeError(Exception):
    pass


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

    @classmethod
    def get_operator(cls, points, base_interpolator, operation):
        d = int(points.shape[1])

        if d not in LibSpatializeFacade.hash_map:
            raise SpatializeError("Points dimension must be 2 or 3")

        if base_interpolator not in LibSpatializeFacade.hash_map[d]:
            raise SpatializeError(f"Base interpolator '{base_interpolator}' not supported for {str(d).upper()}-D data")

        if operation not in LibSpatializeFacade.hash_map[d][base_interpolator]:
            raise SpatializeError(f"Operation '{operation}' not supported for '{base_interpolator}' and "
                                  f"{str(d).upper()}-D data")

        return LibSpatializeFacade.hash_map[d][base_interpolator][operation]


def signature_overload(default_base_interpolator, base_interp_case):
    def outer_function(func):
        def inner_function(*args, **kwargs):
            # get the specific args for the current base interpolator
            if "base_interpolator" not in kwargs:
                kwargs["base_interpolator"] = default_base_interpolator
            bi = kwargs["base_interpolator"]

            if bi not in base_interp_case:
                raise SpatializeError(f"Base interpolator '{bi}' not supported")

            spec_args = base_interp_case[bi]
            # if the argument is needed for the base interpolator
            # and is not in kwargs then add it with its declared
            # default value
            for arg in spec_args.keys():
                if arg not in kwargs:
                    kwargs[arg] = spec_args[arg]
            # check that all arguments are consistent
            # with the base interpolator
            for arg in kwargs.keys():
                if arg != "base_interpolator" and arg not in spec_args:
                    raise SpatializeError(f"Argument '{arg}' not recognized for '{bi}' base interpolator")
            returned_value = func(*args, **kwargs)
            return returned_value
        return inner_function
    return outer_function


def hparams_search(points, values, xi, base_interpolator='idw', n_partitions=500, exponent=2):
    loo = LibSpatializeFacade.get_operator(points, base_interpolator, "loo")
    alpha = 0.7
    print(loo)

    # res = loo(points, values, n_partitions, alpha, exponent, xi)


def griddata(points, values, xi, base_interpolator='idw', n_partitions=500,
             alpha=0.7, agg_function=af.mean, prec_function=pf.mse_precision,
             exponent=2.0):

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

    estimation, precision = nongriddata(points, values, ng_xi, base_interpolator, n_partitions, alpha,
                                        agg_function, prec_function,exponent)

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


def nongriddata(points, values, xi, base_interpolator='idw', n_partitions=500, alpha=0.7,
                agg_function=af.mean, prec_function=pf.mse_precision,
                exponent=2.0):

    estimate = LibSpatializeFacade.get_operator(points, base_interpolator, "estimate")
    e_values = estimate(np.float32(points), np.float32(values), n_partitions, alpha, exponent, np.float32(xi))
    estimation = agg_function(e_values)
    precision = prec_function(estimation, e_values)

    return estimation, precision
