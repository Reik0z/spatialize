import os

import spatialize.gs.esi.aggfunction as af
import spatialize.gs.esi.precfunction as pf
from spatialize.gs import LibSpatializeFacade
from spatialize.gs.esi import esi_griddata

import numpy as np


def func(x, y):  # a kind of "cubic" function
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2


grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:, 0], points[:, 1])

from scipy.interpolate import griddata

grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

import hvplot.pandas  # noqa
import pandas
import xarray as xr

ds_points = pandas.DataFrame({"X": points[:, 1] * 100, "Y": points[:, 0] * 200})
ds = xr.DataArray(func(grid_x, grid_y).T)
ds0 = xr.DataArray(grid_z0.T)
ds1 = xr.DataArray(grid_z1.T)
ds2 = xr.DataArray(grid_z2.T)

w, h = 500, 600


def progress(s):
    print(f'processing ... {int(float(s.split()[1][:-1]))}%\r', end="")


grid_z3, grid_z3p = esi_griddata(points, values, (grid_x, grid_y),
                                 base_interpolator="idw",
                                 callback=progress,
                                 exponent=1.0,
                                 n_partitions=500, alpha=0.975,
                                 agg_function=af.median, prec_function=pf.mse_precision,
                                 backend=LibSpatializeFacade.BackendOptions.DISK_CACHED,
                                 cache_path="/Users/alvaro/Projects/GitHub/spatialize/test/testdata/output/griddata.db"
                                 )
ds3 = xr.DataArray(grid_z3.T)
ds3p = xr.DataArray(grid_z3p.T)

fig = ds3.hvplot.image(title="esi idw", width=w, height=h, xlabel='X', ylabel='Y')
fig += ds3p.hvplot.image(title="esi idw precision", width=w, height=h, xlabel='X', ylabel='Y', cmap='seismic')
fig
