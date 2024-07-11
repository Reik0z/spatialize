from matplotlib import pyplot as plt
import xarray as xr
import hvplot.xarray  # noqa: adds hvplot methods to xarray objects
import hvplot.pandas  # noqa

from spatialize import logging
from spatialize.gs.esi import esi_hparams_search, esi_griddata
import spatialize.gs.esi.aggfunction as af
import numpy as np
from rich import print

import holoviews as hv

logging.log.setLevel("DEBUG")

def func(x, y):  # a kind of "cubic" function
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2


grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:, 0], points[:, 1])

# *** kriging as local interpolator ***
# b_params = esi_hparams_search(points, values, (grid_x, grid_y),
#                               local_interpolator="kriging", griddata=True, k=10,
#                               model=["spherical", "exponential", "cubic", "gaussian"],
#                               nugget=[0.0, 0.5, 1.0],
#                               range=[10.0, 50.0, 100.0, 200.0],
#                               alpha=[0.97, 0.96, 0.95])
# print(b_params)

# *** idw as local interpolator ***
result = esi_hparams_search(points, values, (grid_x, grid_y),
                              local_interpolator="idw", griddata=True, k=10,
                              exponent=list(np.arange(1.0, 15.0, 1.0)),
                              alpha=(0.5, 0.6, 0.8, 0.9))
result.plot_cv_error()
plt.show()

w, h = 500, 600

grid_z3, grid_z3p = esi_griddata(points, values, (grid_x, grid_y),
                                 best_params_found=result.best_result()
                                 )
ds3 = xr.DataArray(grid_z3.T)
ds3p = xr.DataArray(grid_z3p.T)

fig = ds3.hvplot.image(title="esi idw", width=w, height=h, xlabel='X', ylabel='Y')
fig += ds3p.hvplot.image(title="esi idw precision", width=w, height=h, xlabel='X', ylabel='Y', cmap='seismic')

hv.save(fig, 'gs_griddata_figure.png', dpi=144)

exit(0)

# *** refining iwd as local interpolator ***
b_params = esi_hparams_search(points, values, (grid_x, grid_y),
                              local_interpolator="idw", griddata=True, k=10,
                              exponent=list(np.arange(1.0, 15.0, 1.0)),
                              alpha=(0.91, 0.92, 0.93, 0.94, 0.95),
                              agg_function={"mean": af.mean,
                                            "median": af.median,
                                            "p25": af.Percentile(25),
                                            "p75": af.Percentile(75)
                                            })
print(b_params)
