import numpy as np
import xarray as xr
import pandas
import hvplot.xarray  # noqa: adds hvplot methods to xarray objects
import hvplot.pandas  # noqa

import holoviews as hv
from scipy.interpolate import griddata

from spatialize import logging
from spatialize.gs.idw import idw_griddata

hv.extension('matplotlib')

logging.log.setLevel("DEBUG")


def func(x, y):  # a kind of "cubic" function
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2


grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:, 0], points[:, 1])

grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

ds_points = pandas.DataFrame({"X": points[:, 1] * 100, "Y": points[:, 0] * 200})
ds = xr.DataArray(func(grid_x, grid_y).T)
ds0 = xr.DataArray(grid_z0.T)
ds1 = xr.DataArray(grid_z1.T)
ds2 = xr.DataArray(grid_z2.T)

w, h = 500, 600

grid_z3 = idw_griddata(points, values, (grid_x, grid_y),
                       exponent=1.0,
                       radius=10
                       )

ds3 = xr.DataArray(grid_z3.T)

fig = ds3.hvplot.image(title="plain idw", width=w, height=h, xlabel='X', ylabel='Y', cmap='seismic')

hv.save(fig, 'plain_idw_figure.png', dpi=144)
