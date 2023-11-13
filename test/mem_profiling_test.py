import os
import sys
import numpy as np

# if running from 'this' test directory then change to the
# project root directory
curr_dir = os.path.split(os.getcwd())[1]
if curr_dir == "test":
    os.chdir(".")

# load libspatialize
try:
    # check if it's already installed
    import libspatialize
except ImportError:
    # we are in dev env so the compiled library
    # must be in the project root directory.
    sys.path.append('.')

try:
    import spatialize
except ImportError:
    sys.path.append("./src/python")

import spatialize.gs.esi.aggfunction as af
import spatialize.gs.esi.precfunction as pf
from spatialize.gs.esi import esi_griddata
from spatialize.gs.esi import esi_hparams_search

from rich import print as rprint

def progress(s):
    print(f'processing ... {int(float(s.split()[1][:-1]))}%\r', end="")


def func(x, y):  # a kind of "cubic" function
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2


grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:, 0], points[:, 1])


def idw(points, values, grid):
    _, _ = esi_griddata(points, values, grid,
                        base_interpolator="idw",
                        callback=progress,
                        exponent=7.0,
                        n_partitions=100, alpha=0.985,
                        agg_function=af.mean, prec_function=pf.mae_precision)


def kriging(points, values, grid):
    _, _ = esi_griddata(points, values, grid,
                        base_interpolator="kriging",
                        callback=progress,
                        model="cubic", nugget=0.1, range=5000.0,
                        n_partitions=100, alpha=0.97,
                        agg_function=af.mean, prec_function=pf.mae_precision)


def gsearch_kriging(points, values, grid):
    b_params = esi_hparams_search(points, values, grid,
                                  base_interpolator="kriging", griddata=True, k=10,
                                  show_progress=False,
                                  callback=progress,
                                  alpha=(0.70, 0.65))
    print(b_params)


def gsearch_idw(points, values, grid):
    b_params = esi_hparams_search(points, values, grid,
                                  base_interpolator="idw", griddata=True, k=-1,
                                  show_progress=True,
                                  callback=progress,
                                  alpha=list(reversed((0.5, 0.6, 0.8, 0.9, 0.95))))
    print(b_params)


if __name__ == "__main__":
    idw(points, values, (grid_x, grid_y))
    # kriging(points, values, (grid_x, grid_y))
    # gsearch_kriging(points, values, (grid_x, grid_y))
    # gsearch_idw(points, values, (grid_x, grid_y))
