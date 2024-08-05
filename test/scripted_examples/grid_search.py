from matplotlib import pyplot as plt
from spatialize import logging
from spatialize.gs.esi import esi_hparams_search, esi_griddata
import spatialize.gs.esi.aggfunction as af
import numpy as np

logging.log.setLevel("DEBUG")


def func(x, y):  # a kind of "cubic" function
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2


grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:, 0], points[:, 1])

# *** kriging as local interpolator ***
search_result = esi_hparams_search(points, values, (grid_x, grid_y),
                                   local_interpolator="kriging", griddata=True, k=10,
                                   model=["spherical", "exponential", "cubic", "gaussian"],
                                   nugget=[0.0, 0.5, 1.0],
                                   range=[10.0, 50.0, 100.0, 200.0],
                                   alpha=[0.97, 0.96, 0.95])

search_result.plot_cv_error()
plt.show()

result = esi_griddata(points, values, (grid_x, grid_y),
                      best_params_found=search_result.best_result()
                      )

result.quick_plot()
plt.show()

# *** idw as local interpolator ***

# run the grid search
search_result = esi_hparams_search(points, values, (grid_x, grid_y),
                                   local_interpolator="idw", griddata=True, k=10,
                                   p_process="mondrian",
                                   n_partitions=(30, 50, 100),
                                   exponent=list(np.arange(1.0, 5.0, 1.0)),
                                   alpha=(0.95, 0.97, 0.98, 0.985))

search_result.plot_cv_error()
plt.show()

result = esi_griddata(points, values, (grid_x, grid_y),
                      best_params_found=search_result.best_result()
                      )

result.quick_plot()
plt.show()

# *** refining iwd as local interpolator ***
search_result = esi_hparams_search(points, values, (grid_x, grid_y),
                                   local_interpolator="idw", griddata=True, k=10,
                                   exponent=list(np.arange(1.0, 15.0, 1.0)),
                                   alpha=(0.91, 0.92, 0.93, 0.94, 0.95),
                                   agg_function={"mean": af.mean,
                                                 "median": af.median,
                                                 "p25": af.Percentile(25),
                                                 "p75": af.Percentile(75)
                                                 })

result = esi_griddata(points, values, (grid_x, grid_y),
                      best_params_found=search_result.best_result()
                      )

result.quick_plot()
plt.show()
