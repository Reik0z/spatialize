import libspatialize as lsp
import libspatialite as lsplt
from spatialize import SpatializeError
from spatialize._util import in_notebook

# backend constants
SQLITE_BACKEND = "lite"
[IDW, KRIGING] = ["idw", "kriging"]
[IDWSQLite, KRIGINSQLite] = [IDW + SQLITE_BACKEND, KRIGING + SQLITE_BACKEND]


class LibSpatializeFacade:
    # backend options
    class BackendOptions:
        [IN_MEMORY, DISK_CACHED] = ["in-memory", "disk-cached"]

    esi_hash_map = {
        2: {IDW: {"estimate": lsp.estimation_esi_idw,
                  "loo": lsp.loo_esi_idw,
                  "kfold": lsp.kfold_esi_idw},
            KRIGING: {"estimate": lsp.estimation_esi_kriging_2d,
                      "loo": lsp.loo_esi_kriging_2d,
                      "kfold": lsp.kfold_esi_kriging_2d},
            IDWSQLite: {"estimate": lsplt.estimation_esi_idw,
                        "loo": lsplt.loo_esi_idw,
                        "kfold": lsplt.kfold_esi_idw},
            KRIGINSQLite: {"estimate": lsplt.estimation_esi_kriging,
                           "loo": lsplt.loo_esi_kriging,
                           "kfold": lsplt.kfold_esi_kriging},
            },
        3: {IDW: {"estimate": lsp.estimation_esi_idw,
                  "loo": lsp.loo_esi_idw,
                  "kfold": lsp.kfold_esi_idw},
            KRIGING: {"estimate": lsp.estimation_esi_kriging_3d,
                      "loo": lsp.loo_esi_kriging_3d,
                      "kfold": lsp.kfold_esi_kriging_3d},
            IDWSQLite: {"estimate": lsplt.estimation_esi_idw,
                        "loo": lsplt.loo_esi_idw,
                        "kfold": lsplt.kfold_esi_idw},
            KRIGINSQLite: {"estimate": lsplt.estimation_esi_kriging,
                           "loo": lsplt.loo_esi_kriging,
                           "kfold": lsplt.kfold_esi_kriging},
            },
        4: {IDW: {"estimate": lsp.estimation_esi_idw,
                  "loo": lsp.loo_esi_idw,
                  "kfold": lsp.kfold_esi_idw},
            IDWSQLite: {"estimate": lsplt.estimation_esi_idw,
                        "loo": lsplt.loo_esi_idw,
                        "kfold": lsplt.kfold_esi_idw},
            },
        5: {IDW: {"estimate": lsp.estimation_esi_idw,
                  "loo": lsp.loo_esi_idw,
                  "kfold": lsp.kfold_esi_idw},
            IDWSQLite: {"estimate": lsplt.estimation_esi_idw,
                        "loo": lsplt.loo_esi_idw,
                        "kfold": lsplt.kfold_esi_idw},
            },
    }

    esi_kriging_models = {
        "spherical": 1,
        "exponential": 2,
        "cubic": 3,
        "gaussian": 4
    }

    @classmethod
    def get_operator(cls, points, base_interpolator, operation, backend):
        d = int(points.shape[1])

        if d not in LibSpatializeFacade.esi_hash_map:
            raise SpatializeError(f"Points dimension must be in {list(LibSpatializeFacade.esi_hash_map.keys)}")

        operator = LibSpatializeFacade.raw_operator(base_interpolator, backend)

        if operator not in LibSpatializeFacade.esi_hash_map[d]:
            raise SpatializeError(f"Base interpolator '{operator}' not supported for {str(d).upper()}-D data")

        if operation not in LibSpatializeFacade.esi_hash_map[d][operator]:
            raise SpatializeError(f"Operation '{operation}' not supported for '{operator}' and "
                                  f"{str(d).upper()}-D data")

        print(operator, operation)
        return LibSpatializeFacade.esi_hash_map[d][operator][operation]

    @classmethod
    def get_kriging_model_number(cls, model):
        return LibSpatializeFacade.esi_kriging_models[model]

    @classmethod
    def raw_operator(cls, base_interpolator, backend):
        if backend is None:  # setting the backend automatically
            if in_notebook():  # and base_interpolator in set(["idw", "kriging"]):
                print("in notebook ...")
                return base_interpolator + SQLITE_BACKEND
            else:
                print("out of notebook ...")
                return base_interpolator

        if backend == LibSpatializeFacade.BackendOptions.IN_MEMORY:
            return base_interpolator

        if backend == LibSpatializeFacade.BackendOptions.DISK_CACHED:
            if base_interpolator in set(["idw", "kriging"]):
                return base_interpolator + SQLITE_BACKEND

        raise SpatializeError(f"Backend '{backend}' not implemented for base interpolator '{base_interpolator}'")
