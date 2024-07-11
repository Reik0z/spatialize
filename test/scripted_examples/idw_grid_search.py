from multiprocessing import freeze_support

import numpy as np
from rich import print
import matplotlib.pyplot as plt

from spatialize.gs.idw import idw_hparams_search, idw_griddata
from spatialize import logging
import holoviews as hv
import hvplot.xarray  # noqa: adds hvplot methods to xarray objects
import xarray as xr

logging.log.setLevel("DEBUG")


def main():
    def func(x, y):  # a kind of "cubic" function
        return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2

    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

    rng = np.random.default_rng()
    points = rng.random((1000, 2))
    values = func(points[:, 0], points[:, 1])

    # running the grid search
    result = idw_hparams_search(points, values, (grid_x, grid_y),
                                griddata=True, k=10,
                                )

    print(result.search_result_data)
    print(result.best_params)
    print(result.best_result(optimize_data_usage=True))
    print(result.best_result(optimize_data_usage=False))

    result.plot_cv_error()

    # using result to estimate
    grid_z3 = idw_griddata(points, values, (grid_x, grid_y),
                           best_params_found=result.best_result(optimize_data_usage=False))

    # just plotting
    ds3 = xr.DataArray(grid_z3.T)

    w, h = 500, 600
    fig = ds3.hvplot.image(title="plain idw", width=w, height=h, xlabel='X', ylabel='Y', cmap='seismic')

    hv.save(fig, 'bp_plain_idw_figure.png', dpi=144)

    plt.show()


if __name__ == '__main__':
    freeze_support()
    main()
