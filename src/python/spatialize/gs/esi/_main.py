import libspatialize as lsp

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


def loo_griddata(points, values, xi, base_interpolator='idw', n_partitions=500, exponent=2):
    loo = LibSpatializeFacade.get_operator(points, base_interpolator, "loo")
    alpha = 0.7

    res = loo(points, values, n_partitions, alpha, exponent, xi)
