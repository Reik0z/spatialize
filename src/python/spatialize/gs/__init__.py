import libspatialize as lsp
import libspatialite as lsplt

from spatialize import SpatializeError, logging
from spatialize._util import in_notebook
from spatialize.logging import log_message

# backend constants
SQLITE_BACKEND = "lite"


class local_interpolator:
    [IDW, KRIGING] = ["idw", "kriging"]


class partitioning_process:
    [MONDRIAN, VORONOI] = ["mondrian", "voronoi"]


PLAIN_INTERPOLATOR = "plain"
PLAINIDW = PLAIN_INTERPOLATOR + local_interpolator.IDW
[MONDRIANIDW, MONDRIANKRIGING] = [partitioning_process.MONDRIAN + local_interpolator.IDW,
                                  partitioning_process.MONDRIAN + local_interpolator.KRIGING]
VORONOIIDW = partitioning_process.VORONOI + local_interpolator.IDW
[MONDRIANIDWSQLite, MONDRIANKRIGINSQLite] = [MONDRIANIDW + SQLITE_BACKEND, MONDRIANKRIGING + SQLITE_BACKEND]


class lib_spatialize_facade:
    # backend options
    class backend_options:
        [IN_MEMORY, DISK_CACHED] = ["in-memory", "disk-cached"]

    function_hash_map = {
        2: {MONDRIANIDW: {"estimate": lsp.estimation_esi_idw,
                          "loo": lsp.loo_esi_idw,
                          "kfold": lsp.kfold_esi_idw},
            MONDRIANKRIGING: {"estimate": lsp.estimation_esi_kriging_2d,
                              "loo": lsp.loo_esi_kriging_2d,
                              "kfold": lsp.kfold_esi_kriging_2d},
            MONDRIANIDWSQLite: {"estimate": lsplt.estimation_esi_idw,
                                "loo": lsplt.loo_esi_idw,
                                "kfold": lsplt.kfold_esi_idw},
            MONDRIANKRIGINSQLite: {"estimate": lsplt.estimation_esi_kriging,
                                   "loo": lsplt.loo_esi_kriging,
                                   "kfold": lsplt.kfold_esi_kriging},
            VORONOIIDW: {"estimate": lsp.estimation_voronoi_idw,
                         "loo": lsp.loo_voronoi_idw,
                         "kfold": lsp.kfold_voronoi_idw},
            PLAINIDW: {"estimate": lsp.estimation_nn_idw,
                       "loo": lsp.loo_nn_idw,
                       "kfold": lsp.kfold_nn_idw},
            },
        3: {MONDRIANIDW: {"estimate": lsp.estimation_esi_idw,
                          "loo": lsp.loo_esi_idw,
                          "kfold": lsp.kfold_esi_idw},
            MONDRIANKRIGING: {"estimate": lsp.estimation_esi_kriging_3d,
                              "loo": lsp.loo_esi_kriging_3d,
                              "kfold": lsp.kfold_esi_kriging_3d},
            MONDRIANIDWSQLite: {"estimate": lsplt.estimation_esi_idw,
                                "loo": lsplt.loo_esi_idw,
                                "kfold": lsplt.kfold_esi_idw},
            MONDRIANKRIGINSQLite: {"estimate": lsplt.estimation_esi_kriging,
                                   "loo": lsplt.loo_esi_kriging,
                                   "kfold": lsplt.kfold_esi_kriging},
            PLAINIDW: {"estimate": lsp.estimation_nn_idw,
                       "loo": lsp.loo_nn_idw,
                       "kfold": lsp.kfold_nn_idw},
            },
        4: {MONDRIANIDW: {"estimate": lsp.estimation_esi_idw,
                          "loo": lsp.loo_esi_idw,
                          "kfold": lsp.kfold_esi_idw},
            MONDRIANIDWSQLite: {"estimate": lsplt.estimation_esi_idw,
                                "loo": lsplt.loo_esi_idw,
                                "kfold": lsplt.kfold_esi_idw},
            PLAINIDW: {"estimate": lsp.estimation_nn_idw,
                       "loo": lsp.loo_nn_idw,
                       "kfold": lsp.kfold_nn_idw},
            },
        5: {MONDRIANIDW: {"estimate": lsp.estimation_esi_idw,
                          "loo": lsp.loo_esi_idw,
                          "kfold": lsp.kfold_esi_idw},
            MONDRIANIDWSQLite: {"estimate": lsplt.estimation_esi_idw,
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
    def get_operator(cls, points, local_interpolator, operation, partitioning_process, backend):
        d = int(points.shape[1])

        if d not in lib_spatialize_facade.function_hash_map:
            raise SpatializeError(f"Points dimension must be in {list(lib_spatialize_facade.function_hash_map.keys)}")

        operator = lib_spatialize_facade.raw_operator(local_interpolator, partitioning_process, backend)

        if operator not in lib_spatialize_facade.function_hash_map[d]:
            raise SpatializeError(f"Local interpolator '{operator}' not supported for {str(d).upper()}-D data")

        if operation not in lib_spatialize_facade.function_hash_map[d][operator]:
            raise SpatializeError(f"Operation '{operation}' not supported for '{operator}' and "
                                  f"{str(d).upper()}-D data")

        log_message(logging.logger.debug(f"esi operation: {operation}; local operator: {operator}"))
        return lib_spatialize_facade.function_hash_map[d][operator][operation]

    @classmethod
    def get_kriging_model_number(cls, model):
        return lib_spatialize_facade.esi_kriging_models[model]

    @classmethod
    def raw_operator(cls, local_interpolator, partitioning_process, backend):
        log_message(logging.logger.debug(f"partitioning process: {partitioning_process}"))

        if backend is None:  # setting the backend automatically
            if in_notebook():
                log_message(logging.logger.debug("context: in notebook"))

                # at the moment, runs IN_MEMORY, but when it's available,
                # we will have to check if there is GPU, or if we are
                # in google-colab, and run on GPU.
                return partitioning_process + local_interpolator
            else:
                # run IN_MEMORY
                log_message(logging.logger.debug("context: out of notebook"))
                return partitioning_process + local_interpolator

        if backend == lib_spatialize_facade.backend_options.IN_MEMORY:
            return partitioning_process + local_interpolator

        if backend == lib_spatialize_facade.backend_options.DISK_CACHED:
            if local_interpolator in set([MONDRIANIDW, MONDRIANKRIGING]):
                return partitioning_process + local_interpolator + SQLITE_BACKEND

        raise SpatializeError(f"Backend '{backend}' not implemented for local interpolator '{local_interpolator}'")
