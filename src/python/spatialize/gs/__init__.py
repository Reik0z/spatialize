import libspatialize as lsp
from spatialize import SpatializeError


class LibSpatializeFacade:
    esi_hash_map = {
        2: {"idw": {"estimate": lsp.estimation_esi_idw,
                    "loo": lsp.loo_esi_idw,
                    "kfold": lsp.kfold_esi_idw},
            "kriging": {"estimate": lsp.estimation_esi_kriging_2d,
                        "loo": lsp.loo_esi_kriging_2d,
                        "kfold": lsp.kfold_esi_kriging_2d},
            },
        3: {"idw": {"estimate": lsp.estimation_esi_idw,
                    "loo": lsp.loo_esi_idw,
                    "kfold": lsp.kfold_esi_idw},
            "kriging": {"estimate": lsp.estimation_esi_kriging_3d,
                        "loo": lsp.loo_esi_kriging_3d,
                        "kfold": lsp.kfold_esi_kriging_3d},
            },
    }

    esi_kriging_models = {
        "spherical": 1,
        "exponential": 2,
        "cubic": 3,
        "gaussian": 4
    }

    @classmethod
    def get_operator(cls, points, base_interpolator, operation):
        d = int(points.shape[1])

        if d not in LibSpatializeFacade.esi_hash_map:
            raise SpatializeError(f"Points dimension must be in {list(LibSpatializeFacade.esi_hash_map.keys)}")

        if base_interpolator not in LibSpatializeFacade.esi_hash_map[d]:
            raise SpatializeError(f"Base interpolator '{base_interpolator}' not supported for {str(d).upper()}-D data")

        if operation not in LibSpatializeFacade.esi_hash_map[d][base_interpolator]:
            raise SpatializeError(f"Operation '{operation}' not supported for '{base_interpolator}' and "
                                  f"{str(d).upper()}-D data")

        return LibSpatializeFacade.esi_hash_map[d][base_interpolator][operation]

    @classmethod
    def get_kriging_model_number(cls, model):
        return LibSpatializeFacade.esi_kriging_models[model]
