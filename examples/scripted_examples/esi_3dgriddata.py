import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from spatialize import logging
import spatialize.gs.esi.aggfunction as af
from spatialize.gs.esi import esi_griddata
from spatialize.data import load_drill_holes_andes_3D


# get the data
samples, locations = load_drill_holes_andes_3D()
locations = locations.sort_values(["z", "y", "x"])

logging.log.setLevel("DEBUG")

points = samples[['x', 'y', 'z']].values
x, y, z = samples[['x']].values, samples[['y']].values, samples[['z']].values
values = samples[['cu']].values[:, 0]


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter3D(
        x, y, z,
        c=values,
        cmap='coolwarm'
)

plt.show()

grid_x, grid_y, grid_z = locations[['x']].values, locations[['y']].values, locations[['z']].values

result = esi_griddata(points, values, (grid_x, grid_y, grid_z),
                      local_interpolator="idw",
                      p_process="mondrian",
                      data_cond=False,
                      exponent=1.0,
                      n_partitions=100, alpha=0.7,
                      agg_function=af.mean
                      )

print(result.estimation())

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter3D(
        grid_x, grid_y, grid_z,
        c=result.estimation(),
        cmap='coolwarm'
)

plt.show()