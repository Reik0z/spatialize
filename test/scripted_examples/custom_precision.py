import hvplot.xarray  # noqa: adds hvplot methods to xarray objects
import holoviews as hv

import numpy as np

import spatialize.gs.esi.aggfunction as af
from spatialize.gs.esi import esi_griddata
import xarray as xr

from spatialize.gs.esi.precfunction import precision

w, h = 500, 600

import hvplot.pandas  # noqa
import pandas

hv.extension('matplotlib')


def func(x, y):  # a kind of "cubic" function
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2


grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:, 0], points[:, 1])

ds = xr.DataArray(func(grid_x, grid_y).T)
ds_points = pandas.DataFrame({"X": points[:, 1] * 100, "Y": points[:, 0] * 200})


def op_error_precision(estimation, esi_samples):
    dyn_range = np.abs(np.min(estimation) - np.max(estimation))

    @precision
    def _op_error(x, y):
        return np.abs(x - y) / dyn_range

    return _op_error(estimation, esi_samples)


result = esi_griddata(points, values, (grid_x, grid_y),
                      local_interpolator="idw",
                      exponent=1.0,
                      n_partitions=500, alpha=0.95,
                      agg_function=af.median)

grid_z3, grid_z3p = result.estimation(), result.precision(op_error_precision)
ds3 = xr.DataArray(grid_z3.T)
ds3p = xr.DataArray(grid_z3p.T)

fig = ds.hvplot.image(title="original", width=w, height=h, xlabel='X', ylabel='Y')
fig += ds3.hvplot.image(title="esi idw", width=w, height=h, xlabel='X', ylabel='Y')
fig += ds3p.hvplot.image(title="esi idw operational error", width=w, height=h, xlabel='X', ylabel='Y', cmap='seismic') \
       * ds_points.hvplot.points(size=3.0, color="green")

hv.save(fig, 'op_error_idw.png', dpi=144)

result = esi_griddata(points, values, (grid_x, grid_y),
                      local_interpolator="kriging",
                      model="spherical", nugget=0.0, range=10.0,
                      n_partitions=100, alpha=0.9,
                      agg_function=af.median)

grid_z4, grid_z4p = result.estimation(), result.precision(op_error_precision)
ds4 = xr.DataArray(grid_z4.T)
ds4p = xr.DataArray(grid_z4p.T)

fig = ds.hvplot.image(title="original", width=w, height=h, xlabel='X', ylabel='Y')
fig += ds4.hvplot.image(title="esi kriging", width=w, height=h, xlabel='X', ylabel='Y')
fig += ds4p.hvplot.image(title="esi kriging operational error", width=w, height=h, xlabel='X', ylabel='Y',
                         cmap='seismic') \
       * ds_points.hvplot.points(size=3.0, color="green")

hv.save(fig, 'op_error_kriging.png', dpi=144)
