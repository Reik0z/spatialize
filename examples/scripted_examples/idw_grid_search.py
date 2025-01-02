import numpy as np
from rich import print
import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from spatialize.gs.idw import idw_hparams_search, idw_griddata
from spatialize import logging

logging.log.setLevel("DEBUG")


def func(x, y):  # a kind of "cubic" function
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2

grid_cmap, prec_cmap = 'coolwarm', 'bwr'

grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]


rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:, 0], points[:, 1])

# running the grid search
search_result = idw_hparams_search(points, values, (grid_x, grid_y),
                                   griddata=True, k=10,
                                   radius=[0.07, 0.08],
                                   exponent=(0.001, 0.01, 0.1, 1, 2)
                                   )

print(search_result.search_result_data)
print(search_result.best_params)
print(search_result.best_result(optimize_data_usage=True))
print(search_result.best_result(optimize_data_usage=False))

search_result.plot_cv_error()

# using result to estimate
result = idw_griddata(points, values, (grid_x, grid_y),
                      best_params_found=search_result.best_result(optimize_data_usage=False))

fig = plt.figure(dpi=150, figsize=(10,5))
gs = fig.add_gridspec(1, 2, wspace=0.3)
(ax1, ax2) = gs.subplots()

ax1.set_aspect('equal')
ax1.set_title('original data')
img1 = ax1.imshow(func(grid_x, grid_y).T, origin='lower', cmap=grid_cmap)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.1)
colorbar(img1, orientation='vertical', cax=cax)

ax2.set_aspect('equal')
ax2.set_title('idw best result')
img2 = result.plot_estimation(ax2)

plt.show()