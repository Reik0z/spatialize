from multiprocessing import freeze_support

import numpy as np
from rich import print
import matplotlib.pyplot as plt

from spatialize.gs.idw import idw_hparams_search, idw_griddata
from spatialize import logging
import hvplot.xarray  # noqa: adds hvplot methods to xarray objects

logging.log.setLevel("DEBUG")


def func(x, y):  # a kind of "cubic" function
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2

grid_cmap, prec_cmap = 'coolwarm', 'bwr'

grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

plt.imshow(func(grid_x, grid_y).T, origin='lower', cmap=grid_cmap)

rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:, 0], points[:, 1])

# running the grid search
search_result = idw_hparams_search(points, values, (grid_x, grid_y),
                                   griddata=True, k=10,
                                   radius=(0.07, 0.1, 1, 10),
                                   exponent=tuple(np.arange(0.01, 0.1, 0.02)),
                                   )

print(search_result.search_result_data)
print(search_result.best_params)
print(search_result.best_result(optimize_data_usage=True))
print(search_result.best_result(optimize_data_usage=False))

search_result.plot_cv_error()

# using result to estimate
result = idw_griddata(points, values, (grid_x, grid_y),
                      best_params_found=search_result.best_result(optimize_data_usage=False))

result.quick_plot()

plt.show()
