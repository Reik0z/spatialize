import hvplot.xarray  # noqa: adds hvplot methods to xarray objects
import holoviews as hv

import numpy as np
import pandas as pd
from spatialize.gs.esi import esi_hparams_search, esi_nongriddata
import spatialize.gs.esi.aggfunction as af
import spatialize.gs.esi.precfunction as pf
import xarray as xr

hv.extension('matplotlib')

# the samples

# NOTE: modify this to make it download the CSV from
# the GitHub repo
samples = pd.read_csv('../../test/testdata/data.csv')
with open('../../test/testdata/grid.dat', 'r') as data:
    lines = data.readlines()
    lines = [l.strip().split() for l in lines[5:]]
    aux = np.float32(lines)
locations = pd.DataFrame(aux, columns=['X', 'Y', 'Z'])

w, h = 300, 200

points = samples[['x', 'y']].values
values = samples[['cu']].values[:, 0]
xi = locations[['X', 'Y']].values

# operational error function for the observed dynamic range
op_error_precision=pf.OpErrorPrecision(np.abs(np.min(values) - np.max(values)))

# operational error function for the observed mean law
# op_error_precision=pf.OpErrorPrecision(np.nanmean(values))


b_params = esi_hparams_search(points, values, xi,
                              base_interpolator="idw", griddata=False, k=10,
                              exponent=list(np.arange(1.0, 15.0, 1.0)),
                              alpha=(0.5, 0.6, 0.8, 0.9, 0.95))
b_params

grid_z4, grid_z4p = esi_nongriddata(points, values, xi,
                                    base_interpolator="idw",
                                    exponent=5.0,
                                    n_partitions=100, alpha=0.8,
                                    prec_function=op_error_precision)
ds4 = xr.DataArray(grid_z4.reshape(w, h))
ds4p = xr.DataArray(grid_z4p.reshape(w, h) * 100)

fig = ds4.hvplot.image(title="esi idw", width=w, height=h * 2, xlabel='X', ylabel='Y', cmap='bwr', clim=(0, 4.5))
fig += ds4p.hvplot.image(title="esi idw UQ", width=w, height=h * 2, xlabel='X', ylabel='Y', cmap='Spectral')

hv.save(fig, '../nongridata_idw.png', dpi=144)

b_params = esi_hparams_search(points, values, xi,
                              base_interpolator="kriging", griddata=False, k=10,
                              model=["spherical", "exponential", "cubic", "gaussian"],
                              nugget=[0.0, 0.5, 1.0],
                              range=[100.0, 500.0, 1000.0, 5000.0],
                              alpha=list(np.flip(np.arange(0.90, 0.95, 0.01))))
b_params

grid_z4, grid_z4p = esi_nongriddata(points, values, xi,
                                    base_interpolator="kriging",
                                    model="cubic", nugget=0.0, range=1000.0,
                                    n_partitions=100, alpha=0.93,
                                    agg_function=af.median, prec_function=op_error_precision)
ds4 = xr.DataArray(grid_z4.reshape(w, h))
ds4p = xr.DataArray(grid_z4p.reshape(w, h) * 100)

fig = ds4.hvplot.image(title="esi kriging", width=w, height=h * 2, xlabel='X', ylabel='Y', cmap='bwr', clim=(0, 4.5))
fig += ds4p.hvplot.image(title="esi kriging UQ", width=w, height=h * 2, xlabel='X', ylabel='Y', cmap='Spectral')

hv.save(fig, '../nongridata_kriging.png', dpi=144)