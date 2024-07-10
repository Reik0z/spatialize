import libspatialize as lsp
import libspatialite as lsplt

from spatialize import SpatializeError, logging
from spatialize._util import in_notebook
from spatialize.logging import log_message

# backend constants
SQLITE_BACKEND = "lite"
[PLAINIDW, ESIIDW, ESIKRIGING] = ["plain_idw", "idw", "kriging"]
[ESIIDWSQLite, ESIKRIGINSQLite] = [ESIIDW + SQLITE_BACKEND, ESIKRIGING + SQLITE_BACKEND]


class LibSpatializeFacade:
    # backend options
    class BackendOptions:
        [IN_MEMORY, DISK_CACHED] = ["in-memory", "disk-cached"]

    function_hash_map = {
        2: {ESIIDW: {"estimate": lsp.estimation_esi_idw,
                  "loo": lsp.loo_esi_idw,
                  "kfold": lsp.kfold_esi_idw},
            ESIKRIGING: {"estimate": lsp.estimation_esi_kriging_2d,
                      "loo": lsp.loo_esi_kriging_2d,
                      "kfold": lsp.kfold_esi_kriging_2d},
            ESIIDWSQLite: {"estimate": lsplt.estimation_esi_idw,
                        "loo": lsplt.loo_esi_idw,
                        "kfold": lsplt.kfold_esi_idw},
            ESIKRIGINSQLite: {"estimate": lsplt.estimation_esi_kriging,
                           "loo": lsplt.loo_esi_kriging,
                           "kfold": lsplt.kfold_esi_kriging},
            PLAINIDW: {"estimate": lsp.estimation_nn_idw,
                  "loo": lsp.loo_nn_idw,
                  "kfold": lsp.kfold_nn_idw},
            },
        3: {ESIIDW: {"estimate": lsp.estimation_esi_idw,
                  "loo": lsp.loo_esi_idw,
                  "kfold": lsp.kfold_esi_idw},
            ESIKRIGING: {"estimate": lsp.estimation_esi_kriging_3d,
                      "loo": lsp.loo_esi_kriging_3d,
                      "kfold": lsp.kfold_esi_kriging_3d},
            ESIIDWSQLite: {"estimate": lsplt.estimation_esi_idw,
                        "loo": lsplt.loo_esi_idw,
                        "kfold": lsplt.kfold_esi_idw},
            ESIKRIGINSQLite: {"estimate": lsplt.estimation_esi_kriging,
                           "loo": lsplt.loo_esi_kriging,
                           "kfold": lsplt.kfold_esi_kriging},
            PLAINIDW: {"estimate": lsp.estimation_nn_idw,
                       "loo": lsp.loo_nn_idw,
                       "kfold": lsp.kfold_nn_idw},
            },
        4: {ESIIDW: {"estimate": lsp.estimation_esi_idw,
                  "loo": lsp.loo_esi_idw,
                  "kfold": lsp.kfold_esi_idw},
            ESIIDWSQLite: {"estimate": lsplt.estimation_esi_idw,
                        "loo": lsplt.loo_esi_idw,
                        "kfold": lsplt.kfold_esi_idw},
            PLAINIDW: {"estimate": lsp.estimation_nn_idw,
                       "loo": lsp.loo_nn_idw,
                       "kfold": lsp.kfold_nn_idw},
            },
        5: {ESIIDW: {"estimate": lsp.estimation_esi_idw,
                  "loo": lsp.loo_esi_idw,
                  "kfold": lsp.kfold_esi_idw},
            ESIIDWSQLite: {"estimate": lsplt.estimation_esi_idw,
                        "loo": lsplt.loo_esi_idw,
                        "kfold": lsplt.kfold_esi_idw},
            PLAINIDW: {"estimate": lsp.estimation_nn_idw,
                       "loo": lsp.loo_nn_idw,
                       "kfold": lsp.kfold_nn_idw},
            },
    }

    esi_kriging_models = {
        "spherical": 1,
        "exponential": 2,
        "cubic": 3,
        "gaussian": 4
    }

    @classmethod
    def get_operator(cls, points, local_interpolator, operation, backend):
        d = int(points.shape[1])

        if d not in LibSpatializeFacade.function_hash_map:
            raise SpatializeError(f"Points dimension must be in {list(LibSpatializeFacade.function_hash_map.keys)}")

        operator = LibSpatializeFacade.raw_operator(local_interpolator, backend)

        if operator not in LibSpatializeFacade.function_hash_map[d]:
            raise SpatializeError(f"Local interpolator '{operator}' not supported for {str(d).upper()}-D data")

        if operation not in LibSpatializeFacade.function_hash_map[d][operator]:
            raise SpatializeError(f"Operation '{operation}' not supported for '{operator}' and "
                                  f"{str(d).upper()}-D data")

        log_message(logging.logger.debug(f"esi operation: {operation}; local operator: {operator}"))
        return LibSpatializeFacade.function_hash_map[d][operator][operation]

    @classmethod
    def get_kriging_model_number(cls, model):
        return LibSpatializeFacade.esi_kriging_models[model]

    @classmethod
    def raw_operator(cls, local_interpolator, backend):
        if backend is None:  # setting the backend automatically
            if in_notebook():
                log_message(logging.logger.debug("context: in notebook"))
                return local_interpolator + SQLITE_BACKEND
            else:
                log_message(logging.logger.debug("context: out of notebook"))
                return local_interpolator

        if backend == LibSpatializeFacade.BackendOptions.IN_MEMORY:
            return local_interpolator

        if backend == LibSpatializeFacade.BackendOptions.DISK_CACHED:
            if local_interpolator in set([ESIIDW, ESIKRIGING]):
                return local_interpolator + SQLITE_BACKEND

        raise SpatializeError(f"Backend '{backend}' not implemented for local interpolator '{local_interpolator}'")


