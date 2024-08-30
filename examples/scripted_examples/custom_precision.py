import numpy as np
from matplotlib import pyplot as plt

import spatialize.gs.esi.aggfunction as af
from spatialize.gs.esi import esi_griddata
from spatialize.gs.esi.precfunction import loss


def func(x, y):  # a kind of "cubic" function
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2


grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:, 0], points[:, 1])


def op_error_precision(estimation, esi_samples):
    dyn_range = np.abs(np.nanmin(esi_samples) - np.nanmax(esi_samples))

    @loss(af.mean)
    def _op_error(x, y):
        return np.abs(x - y) / dyn_range

    return _op_error(estimation, esi_samples)


def plot_result(result, title):
    grid_cmap, prec_cmap = 'coolwarm', 'bwr'
    fig = plt.figure(dpi=150)
    gs = fig.add_gridspec(2, 2, wspace=0.1, hspace=0.47)
    (ax1, ax2) = gs.subplots()
    ax1, ax2, ax3, ax4 = ax1[0], ax1[1], ax2[0], ax2[1]

    # plot original
    ax1.imshow(func(grid_x, grid_y).T, extent=(0, 1, 0, 1), origin='lower', cmap=grid_cmap)
    ax1.set_title("original")

    # plot estimation
    result.plot_estimation(ax=ax2, cmap=grid_cmap)
    ax2.set_title(title)

    # plot the default mse precision
    result.plot_precision(ax=ax3, cmap=prec_cmap)
    ax3.set_title('mse')
    ax3.plot(points[:, 0], points[:, 1], 'y.', ms=0.5)

    # plot a custom precision
    result.precision(op_error_precision)
    result.plot_precision(ax=ax4, cmap=prec_cmap)
    ax4.set_title('op error')
    ax4.plot(points[:, 0], points[:, 1], 'y.', ms=0.5)

    plt.show()


def esi_idw():
    result = esi_griddata(points, values, (grid_x, grid_y),
                          local_interpolator="idw",
                          exponent=1.0,
                          n_partitions=500, alpha=0.98,
                          agg_function=af.median)

    plot_result(result, 'esi idw')


def esi_kriging():
    result = esi_griddata(points, values, (grid_x, grid_y),
                          local_interpolator="kriging",
                          model="spherical", nugget=0.0, range=10.0,
                          n_partitions=100, alpha=0.9,
                          agg_function=af.median)

    plot_result(result, 'esi kriging')


if __name__ == '__main__':
    esi_idw()
    # esi_kriging()
