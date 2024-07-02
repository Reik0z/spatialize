import numpy as np

import spatialize.gs
from spatialize import SpatializeError
from spatialize._math_util import flatten_grid_data
from spatialize.logging import default_singleton_callback
from spatialize.gs import LibSpatializeFacade


def idw_griddata(points, values, xi, **kwargs):
    ng_xi, original_shape = flatten_grid_data(xi)
    estimation = idw_nongriddata(points, values, ng_xi, **kwargs)
    return estimation.reshape(original_shape)


def idw_nongriddata(points, values, xi, radius=100, exponent=2.0,
                    callback=default_singleton_callback):
    # get the estimator function
    estimate = LibSpatializeFacade.get_operator(points,
                                                spatialize.gs.PLAINIDW,
                                                "estimate", LibSpatializeFacade.BackendOptions.IN_MEMORY)

    # get the argument list
    l_args = [np.float32(points), np.float32(values),
              radius, exponent, np.float32(xi), callback]

    # run
    try:
        estimation = estimate(*l_args)
    except Exception as e:
        raise SpatializeError(e)

    return estimation
