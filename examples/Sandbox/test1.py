import numpy as np

import spatialize.gs.esi.aggfunction as af
from spatialize import logging
from spatialize.gs.esi import esi_griddata
from spatialize.gs.esi.precfunction import loss


logging.log.setLevel("DEBUG")

def func(x, y):  # a kind of "cubic" function
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2


grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:, 0], points[:, 1])

print(values.shape)
print(points.shape)

result = esi_griddata(points, values, (grid_x, grid_y),
                      local_interpolator="idw",
                      p_process="mondrian",
                      data_cond=False,
                      exponent=1.0,
                      n_partitions=500, alpha=0.985,
                      agg_function=af.mean
                      )

def op_error_precision(estimation, esi_samples):
    dyn_range = np.abs(np.nanmin(esi_samples) - np.nanmax(esi_samples))

    @loss
    def _op_error(x, y):
        return np.abs(x - y) / dyn_range

    return _op_error(estimation, esi_samples)

#print(op_error_precision(result.estimation(), result.esi_samples()))
print(result.esi_samples().shape)
print(result.estimation().shape)
print(result.re_estimate(af.median).shape)