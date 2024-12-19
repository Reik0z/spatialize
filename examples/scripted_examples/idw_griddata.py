import numpy as np
from matplotlib import pyplot as plt

from spatialize import logging
from spatialize.gs.idw import idw_griddata


logging.log.setLevel("DEBUG")


def func(x, y):  # a kind of "cubic" function
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2


grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:, 0], points[:, 1])

result = idw_griddata(points, values, (grid_x, grid_y),
                      exponent=1.0,
                      radius=0.01
                      )

print(result.estimation().max())
result.quick_plot()
plt.show()