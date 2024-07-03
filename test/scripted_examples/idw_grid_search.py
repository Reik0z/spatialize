import numpy as np
from rich import print

from spatialize.gs.idw import idw_hparams_search
from spatialize import logging

logging.log.setLevel("DEBUG")


def func(x, y):  # a kind of "cubic" function
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2


grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:, 0], points[:, 1])

b_params, data = idw_hparams_search(points, values, (grid_x, grid_y),
                                    griddata=True, k=10,
                                    )

print(data)
# print(data.cv_error[not np.isfinite(data['cv_error'])])

import matplotlib.pyplot as plt

# plot = data.plot(y='cv_error', kind='line')
plot = data.plot.hist(column=['exponent'])
# plt.hist(data[np.isfinite(data['cv_error'])].values)
plt.show()
