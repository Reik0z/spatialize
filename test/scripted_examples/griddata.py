import numpy as np
from matplotlib import pyplot as plt

import spatialize.gs.esi.aggfunction as af
from spatialize import logging
from spatialize.gs.esi import esi_griddata
from scipy.interpolate import griddata

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

grid_cmap, prec_cmap = 'coolwarm', 'bwr'

plt.imshow(func(grid_x, grid_y).T, extent=(0, 1, 0, 1), origin='lower', cmap=grid_cmap)
plt.show()

result = esi_griddata(points, values, (grid_x, grid_y),
                      local_interpolator="idw",
                      p_process="mondrian",
                      data_cond=False,
                      exponent=1.0,
                      n_partitions=500, alpha=0.985,
                      agg_function=af.mean
                      )

esi_idw_est = result.estimation()

fig = plt.figure(dpi=150)
gs = fig.add_gridspec(2, 2, wspace=0.1, hspace=0.47)
(ax1, ax2) = gs.subplots()
ax1, ax2, ax3, ax4 = ax1[0], ax1[1], ax2[0], ax2[1]

# plot original
ax1.imshow(esi_idw_est.T, extent=(0, 1, 0, 1), origin='lower', cmap=grid_cmap)
ax1.set_title("esi idw")

ax2.imshow(grid_z0.T, extent=(0, 1, 0, 1), origin='lower', cmap=grid_cmap)
ax2.set_title("nearest")

ax3.imshow(grid_z1.T, extent=(0, 1, 0, 1), origin='lower', cmap=grid_cmap)
ax3.set_title("linear")

ax4.imshow(grid_z2.T, extent=(0, 1, 0, 1), origin='lower', cmap=grid_cmap)
ax4.set_title("cubic")

plt.show()